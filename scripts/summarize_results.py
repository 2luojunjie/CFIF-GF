import argparse
import csv
import json
from pathlib import Path

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description="Summarize 10-fold SER results.")
    parser.add_argument("--results-dir", required=True, help="Directory containing fold_* result folders.")
    parser.add_argument("--output-dir", default=None, help="Output directory. Defaults to results-dir.")
    return parser.parse_args()


def load_fold_results(results_dir):
    rows = []
    confusion = None
    for result_path in sorted(Path(results_dir).glob("fold_*/result.json")):
        with result_path.open("r", encoding="utf-8") as handle:
            result = json.load(handle)
        metrics = result["final"]
        row = {
            "fold": result["fold"]["fold"],
            "test_speaker": result["fold"]["test_speaker"],
            "wa": metrics["wa"],
            "ua": metrics["ua"],
            "macro_f1": metrics["macro_f1"],
            "best_epoch": result["best_epoch"],
            "checkpoint": result["checkpoint"],
        }
        rows.append(row)
        matrix = np.asarray(metrics["confusion_matrix"], dtype=np.int64)
        confusion = matrix if confusion is None else confusion + matrix
    if not rows:
        raise FileNotFoundError(f"No fold result.json files found under {results_dir}")
    return rows, confusion


def save_csv(path, rows):
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def save_confusion_png(path, confusion):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 5))
    image = ax.imshow(confusion, cmap="Blues")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Aggregated Confusion Matrix")
    for i in range(confusion.shape[0]):
        for j in range(confusion.shape[1]):
            ax.text(j, i, str(int(confusion[i, j])), ha="center", va="center", color="black")
    fig.colorbar(image, ax=ax)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def summarize_results_dir(results_dir, output_dir=None):
    results_dir = Path(results_dir)
    output_dir = Path(output_dir) if output_dir else results_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    rows, confusion = load_fold_results(results_dir)
    metrics = {}
    for key in ("wa", "ua", "macro_f1"):
        values = np.asarray([row[key] for row in rows], dtype=np.float64)
        metrics[key] = {"mean": float(values.mean()), "std": float(values.std(ddof=0))}

    summary = {
        "folds": rows,
        "mean_std": metrics,
        "confusion_matrix": confusion.tolist(),
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
    save_csv(output_dir / "summary.csv", rows)
    save_confusion_png(output_dir / "confusion_matrix.png", confusion)
    return summary


def main():
    args = parse_args()
    summary = summarize_results_dir(args.results_dir, args.output_dir)
    for row in summary["folds"]:
        print(
            f"fold={row['fold']} speaker={row['test_speaker']} "
            f"WA={row['wa']:.4f} UA={row['ua']:.4f} F1={row['macro_f1']:.4f}"
        )
    print("Mean +/- Std")
    for key, value in summary["mean_std"].items():
        print(f"{key}: {value['mean']:.4f} +/- {value['std']:.4f}")


if __name__ == "__main__":
    main()

