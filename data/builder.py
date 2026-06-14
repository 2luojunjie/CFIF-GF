from torch.utils.data import DataLoader

from .ser_dataset import SpeechEmotionDataset


SUPPORTED_DATASETS = {"IEMOCAP", "EMODB"}


def build_dataset(config, split):
    dataset_cfg = config["dataset"]
    dataset_name = dataset_cfg["name"].upper()
    if dataset_name not in SUPPORTED_DATASETS:
        supported = ", ".join(sorted(SUPPORTED_DATASETS))
        raise ValueError(f"Unsupported dataset '{dataset_cfg['name']}'. Choose from: {supported}")
    return SpeechEmotionDataset(dataset_cfg, split=split)


def build_dataloader(config, split, shuffle=False):
    dataset = build_dataset(config, split)
    train_cfg = config["train"]
    return DataLoader(
        dataset,
        batch_size=train_cfg["batch_size"],
        shuffle=shuffle,
        num_workers=train_cfg.get("num_workers", 0),
        pin_memory=False,
    )

