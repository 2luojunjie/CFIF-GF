import csv
import re
from pathlib import Path


IEMOCAP_LABEL_MAP = {
    "ang": "angry",
    "hap": "happy",
    "exc": "happy",
    "neu": "neutral",
    "sad": "sad",
}

EMODB_LABEL_MAP = {
    "W": "anger",
    "L": "boredom",
    "E": "disgust",
    "A": "fear",
    "F": "happiness",
    "N": "neutral",
    "T": "sadness",
}


def load_manifest(manifest_path):
    path = Path(manifest_path)
    if not path.exists():
        raise FileNotFoundError(f"Manifest not found: {path}")

    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"path", "label", "speaker_id"}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Manifest {path} is missing columns: {sorted(missing)}")
        return [dict(row) for row in reader]


def discover_dataset_items(dataset_cfg):
    dataset_name = dataset_cfg["name"].upper()
    root = Path(dataset_cfg["root"])
    label_names = dataset_cfg["label_names"]
    label_to_index = {name: index for index, name in enumerate(label_names)}

    if dataset_name == "IEMOCAP":
        return discover_iemocap_items(root, label_to_index)
    if dataset_name == "EMODB":
        return discover_emodb_items(root, label_to_index)
    raise ValueError(f"Unsupported dataset: {dataset_cfg['name']}")


def discover_iemocap_items(root, label_to_index):
    if not root.exists():
        raise FileNotFoundError(f"IEMOCAP root not found: {root}")

    utterance_labels = {}
    pattern = re.compile(r"^\[.+?\]\s+(\S+)\s+(\S+)\s+\[")
    for annotation_path in root.glob("Session*/dialog/EmoEvaluation/*.txt"):
        with annotation_path.open("r", encoding="utf-8", errors="ignore") as handle:
            for line in handle:
                match = pattern.match(line.strip())
                if not match:
                    continue
                utterance_id, raw_label = match.groups()
                label_name = IEMOCAP_LABEL_MAP.get(raw_label)
                if label_name in label_to_index:
                    utterance_labels[utterance_id] = label_to_index[label_name]

    items = []
    for wav_path in root.glob("Session*/sentences/wav/**/*.wav"):
        utterance_id = wav_path.stem
        if utterance_id not in utterance_labels:
            continue
        items.append(
            {
                "path": str(wav_path),
                "label": utterance_labels[utterance_id],
                "speaker_id": utterance_id[:6],
            }
        )
    return sorted(items, key=lambda item: item["path"])


def discover_emodb_items(root, label_to_index):
    if not root.exists():
        raise FileNotFoundError(f"EMODB root not found: {root}")

    items = []
    for wav_path in root.rglob("*.wav"):
        stem = wav_path.stem
        if len(stem) < 6:
            continue
        label_name = EMODB_LABEL_MAP.get(stem[5])
        if label_name not in label_to_index:
            continue
        items.append(
            {
                "path": str(wav_path),
                "label": label_to_index[label_name],
                "speaker_id": stem[:2],
            }
        )
    return sorted(items, key=lambda item: item["path"])


def normalize_items(items, label_names):
    label_to_index = {name: index for index, name in enumerate(label_names)}
    normalized = []
    for item in items:
        label = item["label"]
        if isinstance(label, str) and not label.isdigit():
            label = label_to_index[label]
        normalized.append(
            {
                "path": item["path"],
                "label": int(label),
                "speaker_id": str(item["speaker_id"]),
            }
        )
    return normalized

