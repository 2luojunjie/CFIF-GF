import argparse
from collections import Counter

from data.folds import build_loso_folds, load_all_items
from utils import load_config


def parse_args():
    parser = argparse.ArgumentParser(description="Check dataset samples, labels, speakers, and LOSO folds.")
    parser.add_argument("--config", required=True, help="YAML config path.")
    return parser.parse_args()


def count_labels(items, label_names):
    counts = Counter(int(item["label"]) for item in items)
    return {label_names[index]: counts.get(index, 0) for index in range(len(label_names))}


def print_distribution(title, distribution):
    print(title)
    for key, value in distribution.items():
        print(f"  {key}: {value}")


def main():
    args = parse_args()
    config = load_config(args.config)
    dataset_cfg = config["dataset"]
    dataset_name = dataset_cfg["name"].upper()
    label_names = dataset_cfg["label_names"]

    if dataset_name == "IEMOCAP":
        expected = {"angry", "happy", "sad", "neutral"}
        actual = set(label_names)
        if actual != expected:
            raise RuntimeError(f"IEMOCAP labels must be {sorted(expected)}, got {sorted(actual)}")
        print("IEMOCAP label check OK: excited is merged into happy; final labels are angry/happy/sad/neutral.")
    if dataset_name == "EMODB":
        print(f"EMODB label mapping is configurable via dataset.label_names: {label_names}")

    items = load_all_items(dataset_cfg)
    print(f"Total samples: {len(items)}")
    print_distribution("Samples per emotion:", count_labels(items, label_names))
    speaker_counts = Counter(item["speaker_id"] for item in items)
    print_distribution("Samples per speaker:", dict(sorted(speaker_counts.items())))

    folds = build_loso_folds(dataset_cfg)
    for fold in folds:
        train_speakers = {item["speaker_id"] for item in fold["train_items"]}
        test_speakers = {item["speaker_id"] for item in fold["test_items"]}
        overlap = train_speakers.intersection(test_speakers)
        if overlap:
            raise RuntimeError(f"Fold {fold['fold'] - 1} has overlapping train/test speakers: {sorted(overlap)}")
        print(f"\nFold {fold['fold'] - 1} | test speaker: {fold['test_speaker']}")
        print_distribution("  Train class distribution:", count_labels(fold["train_items"], label_names))
        print_distribution("  Test class distribution:", count_labels(fold["test_items"], label_names))
    print("\nOK: LOSO train/test speakers do not overlap.")


if __name__ == "__main__":
    main()

