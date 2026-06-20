from .manifests import discover_dataset_items, load_manifest, normalize_items
from .npy_dataset import load_npy_items


def load_all_items(dataset_cfg):
    if dataset_cfg.get("mock", True):
        return _mock_items(dataset_cfg)

    backend = dataset_cfg.get("backend", "wav").lower()
    if backend == "npy":
        return load_npy_items(dataset_cfg)
    if backend != "wav":
        raise ValueError(f"Unsupported dataset backend '{backend}'. Choose from: wav, npy")

    manifest = dataset_cfg.get("all_manifest")
    if manifest:
        items = load_manifest(manifest)
    else:
        items = discover_dataset_items(dataset_cfg)
    return normalize_items(items, dataset_cfg["label_names"])


def build_loso_folds(dataset_cfg):
    items = load_all_items(dataset_cfg)
    speakers = sorted({item["speaker_id"] for item in items})
    expected_folds = int(dataset_cfg.get("loso", {}).get("expected_folds", 10))
    if len(speakers) != expected_folds:
        raise ValueError(
            f"LOSO expected {expected_folds} speakers/folds, but found {len(speakers)}: {speakers}"
        )

    folds = []
    for fold_index, speaker_id in enumerate(speakers, start=1):
        train_items = [item for item in items if item["speaker_id"] != speaker_id]
        test_items = [item for item in items if item["speaker_id"] == speaker_id]
        folds.append(
            {
                "fold": fold_index,
                "test_speaker": speaker_id,
                "train_items": train_items,
                "test_items": test_items,
            }
        )
    return folds


def _mock_items(dataset_cfg):
    num_samples = int(dataset_cfg.get("mock_num_samples", 32))
    num_classes = int(dataset_cfg["num_classes"])
    speakers = [f"spk{index:02d}" for index in range(10)]
    return [
        {
            "path": f"mock://sample_{index:04d}.wav",
            "label": index % num_classes,
            "speaker_id": speakers[index % len(speakers)],
        }
        for index in range(num_samples)
    ]
