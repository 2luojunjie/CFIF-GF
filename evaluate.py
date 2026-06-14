import argparse
import json
from pathlib import Path

import torch

from data import build_dataloader, build_loso_folds
from models import build_model
from utils import compute_classification_metrics, load_config, set_seed, setup_logger


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a speech emotion recognition model.")
    parser.add_argument("--config", default="configs/default.yaml", help="Path to YAML config.")
    parser.add_argument("--checkpoint", required=True, help="Checkpoint path, usually path/to/best.pt.")
    parser.add_argument("--fold", type=int, default=None, help="0-based fold index. Defaults to checkpoint fold.")
    parser.add_argument("--output", default=None, help="Optional JSON output path.")
    return parser.parse_args()


def resolve_device(config):
    requested = config.get("device", "auto")
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def move_optional_wavlm_features(batch, device):
    features = batch.get("wavlm_features")
    if features is None or features.numel() == 0:
        return None
    return features.to(device)


@torch.no_grad()
def evaluate(model, dataloader, device, num_classes, label_names):
    model.eval()
    all_logits = []
    all_labels = []
    for batch in dataloader:
        labels = batch["label"].to(device)
        logits = model(
            waveform=batch["waveform"].to(device),
            mfcc=batch["mfcc"].to(device),
            spectrogram=batch["spectrogram"].to(device),
            wavlm_features=move_optional_wavlm_features(batch, device),
        )
        all_logits.append(logits.cpu())
        all_labels.append(labels.cpu())
    return compute_classification_metrics(
        torch.cat(all_logits), torch.cat(all_labels), num_classes=num_classes, label_names=label_names
    )


def resolve_eval_items(config, checkpoint, requested_fold):
    if config["dataset"].get("test_manifest"):
        return None

    folds = build_loso_folds(config["dataset"])
    fold_index = requested_fold
    if fold_index is None and checkpoint.get("fold") is not None:
        fold_index = int(checkpoint["fold"]["fold"])
    if fold_index is None:
        raise ValueError("No test_manifest configured. Provide --fold or use a checkpoint saved from fold training.")
    if fold_index < 0 or fold_index >= len(folds):
        raise ValueError(f"--fold must be in [0, {len(folds) - 1}], got {fold_index}")
    return folds[fold_index]["test_items"]


def print_metrics(logger, metrics):
    logger.info("WA=%.4f", metrics["wa"])
    logger.info("UA=%.4f", metrics["ua"])
    logger.info("F1=%.4f", metrics["macro_f1"])
    logger.info("Confusion matrix:")
    for row in metrics["confusion_matrix"]:
        logger.info("%s", row)
    logger.info("Per-class metrics:")
    for item in metrics["per_class"]:
        logger.info(
            "%s | precision=%.4f recall=%.4f f1=%.4f support=%s",
            item["class_name"],
            item["precision"],
            item["recall"],
            item["f1"],
            item["support"],
        )


def main():
    args = parse_args()
    config = load_config(args.config)
    set_seed(int(config.get("seed", 42)))

    logger = setup_logger(log_dir=config.get("log_dir", "logs"), filename="evaluate.log")
    device = resolve_device(config)
    checkpoint = torch.load(args.checkpoint, map_location=device)

    model = build_model(config).to(device)
    model.load_state_dict(checkpoint["model"])
    logger.info("Loaded checkpoint: %s", args.checkpoint)

    eval_items = resolve_eval_items(config, checkpoint, args.fold)
    dataloader = build_dataloader(config, split="test", shuffle=False, items=eval_items)
    metrics = evaluate(
        model,
        dataloader,
        device,
        int(config["model"]["num_classes"]),
        config["dataset"].get("label_names"),
    )
    print_metrics(logger, metrics)

    output_path = Path(args.output) if args.output else Path(args.checkpoint).with_suffix(".eval.json")
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2, ensure_ascii=False)
    logger.info("Saved evaluation: %s", output_path)


if __name__ == "__main__":
    main()
