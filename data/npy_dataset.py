from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .preprocessing import (
    extract_mfcc,
    extract_spectrogram,
    fix_waveform_length,
    maybe_pre_emphasize,
)


FIELD_ALIASES = {
    "waveform": ("waveform", "waveforms", "audio", "audios", "signal", "signals"),
    "mfcc": ("mfcc", "mfccs", "MFCC"),
    "spectrogram": ("spectrogram", "spectrograms", "spec", "specs", "stft"),
    "label": ("label", "labels", "emotion", "emotions", "target", "targets", "y", "Y"),
    "speaker_id": ("speaker_id", "speaker_ids", "speaker", "speakers", "subject", "subjects"),
    "file_path": ("file_path", "file_paths", "path", "paths", "filename", "filenames"),
    "wavlm_features": ("wavlm_features", "wavlm_feature", "wavlm", "ssl_features"),
}


def _unwrap_numpy_object(value):
    if isinstance(value, np.ndarray) and value.shape == () and value.dtype == object:
        return value.item()
    return value


def _decode_scalar(value):
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return value


class NpyDataStore:
    """Read common preprocessed .npy layouts without changing the wav backend."""

    def __init__(self, dataset_cfg):
        self.dataset_cfg = dataset_cfg
        self.npy_cfg = dataset_cfg.get("npy", {})
        npy_path = dataset_cfg.get("npy_path")
        if not npy_path:
            raise ValueError("dataset.npy_path is required when dataset.backend is 'npy'.")
        self.path = Path(npy_path)
        if not self.path.exists():
            raise FileNotFoundError(f"NPY dataset not found: {self.path}")

        mmap_mode = self.npy_cfg.get("mmap_mode")
        allow_pickle = bool(self.npy_cfg.get("allow_pickle", True))
        try:
            raw = np.load(self.path, allow_pickle=allow_pickle, mmap_mode=mmap_mode)
        except ValueError as error:
            if mmap_mode is not None and "Python objects" in str(error):
                raw = np.load(self.path, allow_pickle=allow_pickle, mmap_mode=None)
            else:
                raise
        self.raw = _unwrap_numpy_object(raw)
        self.records = self._resolve_records(self.raw)

    def _resolve_records(self, raw):
        samples_key = self.npy_cfg.get("samples_key")
        if isinstance(raw, dict) and samples_key:
            if samples_key not in raw:
                raise KeyError(f"Configured dataset.npy.samples_key '{samples_key}' not found in {self.path}")
            return _unwrap_numpy_object(raw[samples_key])
        if isinstance(raw, dict):
            for candidate in ("samples", "records", "items"):
                if candidate in raw and self._looks_like_records(raw[candidate]):
                    return raw[candidate]
        return raw

    @staticmethod
    def _looks_like_records(value):
        value = _unwrap_numpy_object(value)
        return isinstance(value, (list, tuple)) or (
            isinstance(value, np.ndarray) and value.dtype == object and value.ndim == 1
        )

    def __len__(self):
        if isinstance(self.records, dict):
            labels = self.get_array("label", required=True)
            return len(labels)
        if isinstance(self.records, np.ndarray) and self.records.dtype.names:
            return len(self.records)
        if isinstance(self.records, (list, tuple, np.ndarray)):
            return len(self.records)
        raise TypeError(
            f"Unsupported NPY root type {type(self.records).__name__}. "
            "Use a dict of arrays, structured array, or one-dimensional records array."
        )

    def configured_key(self, canonical_name):
        return self.npy_cfg.get("keys", {}).get(canonical_name)

    def candidate_keys(self, canonical_name):
        configured = self.configured_key(canonical_name)
        if configured:
            return (configured,)
        return FIELD_ALIASES[canonical_name]

    def get_array(self, canonical_name, required=False):
        if not isinstance(self.records, dict):
            return None
        for key in self.candidate_keys(canonical_name):
            if key in self.records:
                return self.records[key]
        if required:
            raise KeyError(
                f"Missing '{canonical_name}' in {self.path}. Available keys: {sorted(self.records.keys())}. "
                f"Set dataset.npy.keys.{canonical_name} in YAML if a custom key is used."
            )
        return None

    def get(self, index):
        if isinstance(self.records, dict):
            return self._get_from_mapping(index)
        record = _unwrap_numpy_object(self.records[index])
        if isinstance(record, np.void) and record.dtype.names:
            record = {name: record[name] for name in record.dtype.names}
        if isinstance(record, (tuple, list)):
            fields = self.npy_cfg.get("record_fields")
            if not fields:
                raise ValueError(
                    "Tuple/list NPY records require dataset.npy.record_fields, for example "
                    "[waveform, label, speaker_id]."
                )
            record = dict(zip(fields, record))
        if not isinstance(record, dict):
            raise TypeError(f"NPY sample {index} must be a dict/structured record, got {type(record).__name__}")
        return self._canonicalize_record(record, index)

    def _get_from_mapping(self, index):
        sample = {}
        for canonical_name in FIELD_ALIASES:
            array = self.get_array(canonical_name, required=canonical_name == "label")
            if array is not None:
                sample[canonical_name] = array[index]
        sample.setdefault("file_path", f"{self.path}#{index}")
        return sample

    def _canonicalize_record(self, record, index):
        canonical = {}
        for canonical_name in FIELD_ALIASES:
            for key in self.candidate_keys(canonical_name):
                if key in record:
                    canonical[canonical_name] = _unwrap_numpy_object(record[key])
                    break
        canonical.setdefault("file_path", f"{self.path}#{index}")
        return canonical


def normalize_npy_label(value, label_names, label_map=None):
    value_array = np.asarray(value)
    if value_array.ndim == 1 and value_array.size > 1:
        return int(np.argmax(value_array))
    value = _decode_scalar(value)
    if label_map and str(value) in label_map:
        value = label_map[str(value)]
    if isinstance(value, str) and not value.isdigit():
        if value not in label_names:
            raise ValueError(f"Unknown NPY label '{value}'. Configured labels: {label_names}")
        return label_names.index(value)
    return int(value)


def load_npy_items(dataset_cfg):
    store = NpyDataStore(dataset_cfg)
    label_names = dataset_cfg["label_names"]
    label_map = dataset_cfg.get("npy", {}).get("label_map")
    items = []
    for index in range(len(store)):
        record = store.get(index)
        speaker_id = record.get("speaker_id")
        if speaker_id is None:
            speaker_id = infer_speaker_id(index, len(store), dataset_cfg.get("npy", {}))
        items.append(
            {
                "backend": "npy",
                "index": index,
                "path": str(_decode_scalar(record.get("file_path"))),
                "label": normalize_npy_label(record["label"], label_names, label_map),
                "speaker_id": str(_decode_scalar(speaker_id)),
            }
        )
    return items


def infer_speaker_id(index, num_samples, npy_cfg):
    strategy = npy_cfg.get("speaker_id_strategy")
    if strategy == "block":
        num_speakers = int(npy_cfg.get("num_speakers", 0))
        if num_speakers <= 0 or num_samples % num_speakers != 0:
            raise ValueError(
                "dataset.npy.speaker_id_strategy=block requires num_speakers > 0 and an evenly divisible "
                f"sample count, got samples={num_samples}, speakers={num_speakers}."
            )
        block_size = num_samples // num_speakers
        return f"{npy_cfg.get('speaker_prefix', 'spk')}{index // block_size:02d}"
    if strategy == "modulo":
        num_speakers = int(npy_cfg.get("num_speakers", 0))
        if num_speakers <= 0:
            raise ValueError("dataset.npy.speaker_id_strategy=modulo requires num_speakers > 0.")
        return f"{npy_cfg.get('speaker_prefix', 'spk')}{index % num_speakers:02d}"
    raise ValueError(
        "NPY samples have no speaker_id. Configure dataset.npy.keys.speaker_id, or explicitly set "
        "dataset.npy.speaker_id_strategy to block/modulo with num_speakers. Do not guess this for LOSO."
    )


class NpySpeechEmotionDataset(Dataset):
    """Dataset for preprocessed NPY samples, returning the same fields as the wav backend."""

    def __init__(self, dataset_cfg, items=None):
        self.dataset_cfg = dataset_cfg
        self.preprocessing_cfg = dataset_cfg.get("preprocessing", {})
        self.sample_rate = int(dataset_cfg.get("sample_rate", 16000))
        self.duration_seconds = float(dataset_cfg.get("duration_seconds", 3.0))
        self.items = items if items is not None else load_npy_items(dataset_cfg)
        self._store = None

    @property
    def store(self):
        if self._store is None:
            self._store = NpyDataStore(self.dataset_cfg)
        return self._store

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        item = self.items[index]
        record = self.store.get(int(item["index"]))
        waveform = self._prepare_waveform(record)
        mfcc, spectrogram = self._prepare_handcrafted_features(record, waveform)
        wavlm_features = record.get("wavlm_features")
        if wavlm_features is None:
            wavlm_tensor = torch.empty(0)
        else:
            wavlm_tensor = torch.as_tensor(np.asarray(wavlm_features), dtype=torch.float32)

        return {
            "waveform": torch.from_numpy(waveform),
            "mfcc": torch.from_numpy(mfcc),
            "spectrogram": torch.from_numpy(spectrogram),
            "label": torch.tensor(item["label"], dtype=torch.long),
            "speaker_id": item["speaker_id"],
            "file_path": item["path"],
            "wavlm_features": wavlm_tensor,
        }

    def _prepare_waveform(self, record):
        waveform = record.get("waveform")
        if waveform is None:
            missing_policy = self.dataset_cfg.get("npy", {}).get("missing_feature_policy", "error")
            if record.get("wavlm_features") is None and missing_policy != "zeros":
                raise ValueError(
                    "NPY sample has neither waveform nor wavlm_features. This MFCC-only file cannot feed "
                    "WavLM_Att/CFIF-GF. Set dataset.npy.missing_feature_policy=zeros only for pipeline tests."
                )
            return np.zeros(int(self.sample_rate * self.duration_seconds), dtype=np.float32)
        if bool(self.dataset_cfg.get("npy", {}).get("enforce_fixed_waveform", True)):
            return fix_waveform_length(waveform, self.sample_rate, self.duration_seconds)
        return np.asarray(waveform, dtype=np.float32).reshape(-1)

    def _prepare_handcrafted_features(self, record, waveform):
        mfcc = record.get("mfcc")
        spectrogram = record.get("spectrogram")
        compute_missing = bool(self.dataset_cfg.get("npy", {}).get("compute_missing_features", True))
        missing_policy = self.dataset_cfg.get("npy", {}).get("missing_feature_policy", "error")
        if (mfcc is None or spectrogram is None) and not compute_missing and missing_policy != "zeros":
            raise ValueError(
                "NPY sample is missing MFCC/spectrogram and dataset.npy.compute_missing_features is false."
            )

        feature_waveform = maybe_pre_emphasize(
            waveform,
            enabled=bool(self.preprocessing_cfg.get("pre_emphasis", False)),
            coefficient=float(self.preprocessing_cfg.get("pre_emphasis_coeff", 0.97)),
        )
        if mfcc is None and compute_missing:
            mfcc = extract_mfcc(
                feature_waveform,
                sample_rate=self.sample_rate,
                n_mfcc=int(self.preprocessing_cfg.get("n_mfcc", 40)),
                window_ms=float(self.preprocessing_cfg.get("mfcc_window_ms", 40)),
                hop_ms=float(self.preprocessing_cfg.get("mfcc_hop_ms", 10)),
                window=self.preprocessing_cfg.get("window", "hamming"),
            )
        if spectrogram is None and compute_missing:
            spectrogram = extract_spectrogram(
                feature_waveform,
                sample_rate=self.sample_rate,
                n_fft=int(self.preprocessing_cfg.get("spectrogram_n_fft", 800)),
                bins=int(self.preprocessing_cfg.get("spectrogram_bins", 200)),
                hop_ms=float(self.preprocessing_cfg.get("spectrogram_hop_ms", 10)),
                window=self.preprocessing_cfg.get("window", "hamming"),
            )
        if mfcc is None and missing_policy == "zeros":
            mfcc_dim = int(self.dataset_cfg.get("npy", {}).get("mfcc_dim", 40))
            mfcc = np.zeros((mfcc_dim, 301), dtype=np.float32)
        if spectrogram is None and missing_policy == "zeros":
            spec_bins = int(self.preprocessing_cfg.get("spectrogram_bins", 200))
            spectrogram = np.zeros((spec_bins, 301), dtype=np.float32)

        mfcc_dim = int(self.dataset_cfg.get("npy", {}).get("mfcc_dim", self.preprocessing_cfg.get("n_mfcc", 40)))
        spec_bins = int(self.preprocessing_cfg.get("spectrogram_bins", 200))
        return self._orient_feature(mfcc, mfcc_dim, "MFCC"), self._orient_feature(
            spectrogram, spec_bins, "spectrogram"
        )

    @staticmethod
    def _orient_feature(value, expected_first_dim, name):
        array = np.asarray(value, dtype=np.float32).squeeze()
        if array.ndim != 2:
            raise ValueError(f"{name} must be 2-D after squeeze, got shape {array.shape}")
        if array.shape[0] == expected_first_dim:
            return array
        if array.shape[1] == expected_first_dim:
            return array.T
        raise ValueError(
            f"Cannot orient {name} shape {array.shape}; expected one dimension to equal {expected_first_dim}."
        )
