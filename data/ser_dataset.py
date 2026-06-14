import torch
from torch.utils.data import Dataset

from .manifests import discover_dataset_items, load_manifest, normalize_items
from .preprocessing import extract_mfcc, extract_spectrogram, load_audio_16k_fixed


class SpeechEmotionDataset(Dataset):
    """SER dataset returning waveform, MFCC, spectrogram, and metadata."""

    def __init__(self, dataset_cfg, split="train", items=None):
        self.dataset_cfg = dataset_cfg
        self.split = split
        self.mock = bool(dataset_cfg.get("mock", True))
        self.num_classes = int(dataset_cfg["num_classes"])
        self.sample_rate = int(dataset_cfg.get("sample_rate", 16000))
        self.duration_seconds = float(dataset_cfg.get("duration_seconds", 3.0))
        self.num_samples = int(self.sample_rate * self.duration_seconds)
        self.preprocessing_cfg = dataset_cfg.get("preprocessing", {})

        if items is not None:
            self.items = normalize_items(items, dataset_cfg["label_names"])
        elif self.mock:
            self.items = self._build_mock_items()
        else:
            manifest = dataset_cfg.get(f"{split}_manifest")
            if manifest:
                self.items = normalize_items(load_manifest(manifest), dataset_cfg["label_names"])
            elif split == "all":
                self.items = normalize_items(discover_dataset_items(dataset_cfg), dataset_cfg["label_names"])
            else:
                raise ValueError(
                    f"dataset.{split}_manifest is required for split='{split}'. "
                    "For LOSO training, use --loso with dataset.all_manifest or a supported dataset root."
                )

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        item = self.items[index]
        if self.mock:
            return self._mock_sample(index, item)

        waveform = load_audio_16k_fixed(
            item["path"],
            sample_rate=self.sample_rate,
            duration_seconds=self.duration_seconds,
        )
        mfcc = extract_mfcc(
            waveform,
            sample_rate=self.sample_rate,
            n_mfcc=int(self.preprocessing_cfg.get("n_mfcc", 40)),
            window_ms=float(self.preprocessing_cfg.get("mfcc_window_ms", 40)),
            hop_ms=float(self.preprocessing_cfg.get("mfcc_hop_ms", 10)),
        )
        spectrogram = extract_spectrogram(
            waveform,
            sample_rate=self.sample_rate,
            n_fft=int(self.preprocessing_cfg.get("spectrogram_n_fft", 800)),
            bins=int(self.preprocessing_cfg.get("spectrogram_bins", 200)),
            hop_ms=float(self.preprocessing_cfg.get("spectrogram_hop_ms", 10)),
        )

        return {
            "waveform": torch.from_numpy(waveform),
            "mfcc": torch.from_numpy(mfcc),
            "spectrogram": torch.from_numpy(spectrogram),
            "label": torch.tensor(item["label"], dtype=torch.long),
            "speaker_id": item["speaker_id"],
            "file_path": item["path"],
            "wavlm_features": self._load_offline_wavlm_features(item),
        }

    def _build_mock_items(self):
        num_samples = int(self.dataset_cfg.get("mock_num_samples", 32))
        speakers = [f"spk{index:02d}" for index in range(10)]
        return [
            {
                "path": f"mock://sample_{index:04d}.wav",
                "label": index % self.num_classes,
                "speaker_id": speakers[index % len(speakers)],
                "wavlm_path": "",
            }
            for index in range(num_samples)
        ]

    def _mock_sample(self, index, item):
        generator = torch.Generator().manual_seed(index)
        waveform = torch.randn(self.num_samples, generator=generator)
        hop_length = int(self.sample_rate * 0.01)
        frames = 1 + self.num_samples // hop_length
        mfcc = torch.randn(40, frames, generator=generator)
        spectrogram = torch.randn(200, frames, generator=generator)
        return {
            "waveform": waveform,
            "mfcc": mfcc,
            "spectrogram": spectrogram,
            "label": torch.tensor(item["label"], dtype=torch.long),
            "speaker_id": item["speaker_id"],
            "file_path": item["path"],
            "wavlm_features": torch.empty(0),
        }

    @staticmethod
    def _load_offline_wavlm_features(item):
        wavlm_path = item.get("wavlm_path")
        if not wavlm_path:
            return torch.empty(0)
        return torch.load(wavlm_path, map_location="cpu").float()
