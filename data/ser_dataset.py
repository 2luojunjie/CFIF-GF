import csv
from pathlib import Path

import torch
from torch.utils.data import Dataset


class SpeechEmotionDataset(Dataset):
    """Minimal SER dataset.

    In mock mode, returns random fixed-length waveforms so the training loop can
    run before IEMOCAP/EMODB parsing is implemented.
    """

    def __init__(self, dataset_cfg, split):
        self.dataset_cfg = dataset_cfg
        self.split = split
        self.mock = bool(dataset_cfg.get("mock", True))
        self.num_classes = int(dataset_cfg["num_classes"])
        self.sample_rate = int(dataset_cfg.get("sample_rate", 16000))
        self.duration_seconds = float(dataset_cfg.get("duration_seconds", 3.0))
        self.num_samples = int(self.sample_rate * self.duration_seconds)

        if self.mock:
            self.items = list(range(int(dataset_cfg.get("mock_num_samples", 32))))
        else:
            manifest = dataset_cfg.get(f"{split}_manifest")
            if not manifest:
                raise ValueError(f"dataset.{split}_manifest is required when dataset.mock is false")
            self.items = self._load_manifest(manifest)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        if self.mock:
            generator = torch.Generator().manual_seed(index)
            waveform = torch.randn(self.num_samples, generator=generator)
            label = index % self.num_classes
            return {"waveform": waveform, "label": torch.tensor(label, dtype=torch.long)}

        item = self.items[index]
        raise NotImplementedError(
            "Real audio loading is reserved for the next implementation step. "
            f"Received manifest item: {item}"
        )

    @staticmethod
    def _load_manifest(manifest_path):
        path = Path(manifest_path)
        if not path.exists():
            raise FileNotFoundError(f"Manifest not found: {path}")

        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            required = {"path", "label"}
            missing = required.difference(reader.fieldnames or [])
            if missing:
                raise ValueError(f"Manifest {path} is missing columns: {sorted(missing)}")
            return list(reader)

