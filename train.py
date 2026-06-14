import argparse
import csv
import json
from pathlib import Path

import torch
from torch import nn

from data import build_dataloader, build_loso_folds
from models import build_model
from utils import compute_classification_metrics, load_config, set_seed, setup_logger

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    SummaryWriter = None


def parse_args():
    parser = argparse.ArgumentParser(description="Train a speech emotion recognition model.")
    parser.add_argument("--config", default="configs/default.yaml", help="Path to YAML config.")
    parser.add_argument("--fold", type=int, default=None, help="0-based LOSO fold index. Omit to train all folds.")
    parser.add_argument("--resume", default=None, help="Path to a checkpoint for resuming training.")
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


def forward_model(model, batch, device):
    # waveform: [B, samples]; mfcc: [B, 40, T_m]; spectrogram: [B, F, T_s].
    return model(
        waveform=batch["waveform"].to(device),
        mfcc=batch["mfcc"].to(device),
        spectrogram=batch["spectrogram"].to(device),
        wavlm_features=move_optional_wavlm_features(batch, device),
    )


def train_one_epoch(model, dataloader, criterion, optimizer, device, num_classes, label_names):
    model.train()
    total_loss = 0.0
    all_logits = []
    all_labels = []

    for batch in dataloader:
        labels = batch["label"].to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = forward_model(model, batch, device)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        all_logits.append(logits.detach().cpu())
        all_labels.append(labels.detach().cpu())

    metrics = compute_classification_metrics(
        torch.cat(all_logits), torch.cat(all_labels), num_classes=num_classes, label_names=label_names
    )
    metrics["loss"] = total_loss / max(len(dataloader), 1)
    return metrics


@torch.no_grad()
def evaluate_split(model, dataloader, criterion, device, num_classes, label_names):
    model.eval()
    total_loss = 0.0
    all_logits = []
    all_labels = []

    for batch in dataloader:
        labels = batch["label"].to(device)
        logits = forward_model(model, batch, device)
        loss = criterion(logits, labels)
        total_loss += loss.item()
        all_logits.append(logits.cpu())
        all_labels.append(labels.cpu())

    metrics = compute_classification_metrics(
        torch.cat(all_logits), torch.cat(all_labels), num_classes=num_classes, label_names=label_names
    )
    metrics["loss"] = total_loss / max(len(dataloader), 1)
    return metrics


def build_optimizer(model, config):
    optimizer_name = config["train"].get("optimizer", "AdamW")
    if optimizer_name != "AdamW":
        raise ValueError(f"Unsupported optimizer '{optimizer_name}'. Current implementation supports AdamW.")
    return torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=float(config["train"]["learning_rate"]),
        weight_decay=float(config["train"].get("weight_decay", 0.0)),
    )


def load_resume_checkpoint(resume_path, model, optimizer, device):
    checkpoint = torch.load(resume_path, map_location=device)
    model.load_state_dict(checkpoint["model"])
    if "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])
    start_epoch = int(checkpoint.get("epoch", 0)) + 1
    best_score = float(checkpoint.get("best_score", -1.0))
    history = checkpoint.get("history", [])
    return start_epoch, best_score, history


def save_checkpoint(path, model, optimizer, epoch, config, fold_info, metrics, best_score, history):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "config": config,
            "fold": fold_info,
            "metrics": metrics,
            "best_score": best_score,
            "history": history,
        },
        path,
    )


def append_csv_log(path, row):
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def create_summary_writer(config, output_dir):
    if config["train"].get("log_backend", "csv").lower() != "tensorboard":
        return None
    if SummaryWriter is None:
        raise ImportError("TensorBoard logging requested, but tensorboard is not installed.")
    return SummaryWriter(log_dir=str(output_dir / "tensorboard"))


def log_epoch(writer, epoch, train_metrics, eval_metrics):
    if writer is None:
        return
    for prefix, metrics in (("train", train_metrics), ("eval", eval_metrics)):
        for key in ("loss", "wa", "ua", "macro_f1"):
            writer.add_scalar(f"{prefix}/{key}", metrics[key], epoch)


def run_training(config, device, logger, output_dir, train_items, eval_items, fold_info, resume_path=None):
    train_loader = build_dataloader(config, split="train", shuffle=True, items=train_items)
    eval_loader = build_dataloader(config, split="test", shuffle=False, items=eval_items)

    model = build_model(config).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = build_optimizer(model, config)

    start_epoch = 1
    best_score = -1.0
    history = []
    if resume_path:
        start_epoch, best_score, history = load_resume_checkpoint(resume_path, model, optimizer, device)
        logger.info("Resumed from %s at epoch %s", resume_path, start_epoch)

    train_cfg = config["train"]
    epochs = int(train_cfg.get("epochs", 100))
    patience = int(train_cfg.get("early_stopping", {}).get("patience", 10))
    min_delta = float(train_cfg.get("early_stopping", {}).get("min_delta", 0.0))
    monitor = train_cfg.get("early_stopping", {}).get("monitor", "wa")
    num_classes = int(config["model"]["num_classes"])
    label_names = config["dataset"].get("label_names")
    output_dir.mkdir(parents=True, exist_ok=True)
    writer = create_summary_writer(config, output_dir)
    bad_epochs = 0

    try:
        for epoch in range(start_epoch, epochs + 1):
            train_metrics = train_one_epoch(
                model, train_loader, criterion, optimizer, device, num_classes, label_names
            )
            eval_metrics = evaluate_split(model, eval_loader, criterion, device, num_classes, label_names)
            score = float(eval_metrics[monitor])
            improved = score > best_score + min_delta
            if improved:
                best_score = score
                bad_epochs = 0
                save_checkpoint(
                    output_dir / "best.pt",
                    model,
                    optimizer,
                    epoch,
                    config,
                    fold_info,
                    eval_metrics,
                    best_score,
                    history,
                )
            else:
                bad_epochs += 1

            save_checkpoint(
                output_dir / "last.pt",
                model,
                optimizer,
                epoch,
                config,
                fold_info,
                eval_metrics,
                best_score,
                history,
            )

            row = {
                "epoch": epoch,
                "train_loss": train_metrics["loss"],
                "train_wa": train_metrics["wa"],
                "train_ua": train_metrics["ua"],
                "train_f1": train_metrics["macro_f1"],
                "eval_loss": eval_metrics["loss"],
                "eval_wa": eval_metrics["wa"],
                "eval_ua": eval_metrics["ua"],
                "eval_f1": eval_metrics["macro_f1"],
                "best_score": best_score,
            }
            append_csv_log(output_dir / "train_log.csv", row)
            log_epoch(writer, epoch, train_metrics, eval_metrics)
            history.append({"epoch": epoch, "train": train_metrics, "eval": eval_metrics})

            logger.info(
                "epoch=%03d train_loss=%.4f train_WA=%.4f train_UA=%.4f train_F1=%.4f "
                "eval_loss=%.4f eval_WA=%.4f eval_UA=%.4f eval_F1=%.4f%s",
                epoch,
                train_metrics["loss"],
                train_metrics["wa"],
                train_metrics["ua"],
                train_metrics["macro_f1"],
                eval_metrics["loss"],
                eval_metrics["wa"],
                eval_metrics["ua"],
                eval_metrics["macro_f1"],
                " *best*" if improved else "",
            )

            if bad_epochs >= patience:
                logger.info("Early stopping at epoch %s. Best %s=%.4f", epoch, monitor, best_score)
                break
    finally:
        if writer is not None:
            writer.close()

    best_checkpoint = torch.load(output_dir / "best.pt", map_location="cpu")
    result = {
        "fold": fold_info,
        "best_epoch": best_checkpoint["epoch"],
        "best_score": best_checkpoint["best_score"],
        "final": best_checkpoint["metrics"],
        "checkpoint": str(output_dir / "best.pt"),
    }
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, ensure_ascii=False)
    return result


def summarize_results(results):
    keys = ["wa", "ua", "macro_f1"]
    return {key: sum(result["final"][key] for result in results) / len(results) for key in keys}


def select_folds(config, requested_fold):
    folds = build_loso_folds(config["dataset"])
    if requested_fold is None:
        return folds
    if requested_fold < 0 or requested_fold >= len(folds):
        raise ValueError(f"--fold must be in [0, {len(folds) - 1}], got {requested_fold}")
    return [folds[requested_fold]]


def main():
    args = parse_args()
    config = load_config(args.config)
    set_seed(int(config.get("seed", 42)))

    logger = setup_logger(log_dir=config.get("log_dir", "logs"), filename="train.log")
    device = resolve_device(config)
    logger.info("Using device: %s", device)
    logger.info("Dataset: %s | Model: %s", config["dataset"]["name"], config["model"]["name"])

    folds = select_folds(config, args.fold)
    output_root = Path(config.get("output_dir", "outputs")) / config["dataset"]["name"] / config["model"]["name"]
    results = []
    for fold in folds:
        fold_index_zero_based = int(fold["fold"]) - 1
        fold_info = {"fold": fold_index_zero_based, "test_speaker": fold["test_speaker"]}
        fold_output = output_root / f"fold_{fold_index_zero_based:02d}_{fold['test_speaker']}"
        logger.info(
            "Starting fold=%s test_speaker=%s train=%s eval=%s",
            fold_index_zero_based,
            fold["test_speaker"],
            len(fold["train_items"]),
            len(fold["test_items"]),
        )
        results.append(
            run_training(
                config=config,
                device=device,
                logger=logger,
                output_dir=fold_output,
                train_items=fold["train_items"],
                eval_items=fold["test_items"],
                fold_info=fold_info,
                resume_path=args.resume if len(folds) == 1 else None,
            )
        )

    if len(results) > 1:
        summary = summarize_results(results)
        summary_path = output_root / "cross_validation_summary.json"
        with summary_path.open("w", encoding="utf-8") as handle:
            json.dump({"folds": results, "average": summary}, handle, indent=2, ensure_ascii=False)
        logger.info("10-fold average | WA=%.4f UA=%.4f F1=%.4f", summary["wa"], summary["ua"], summary["macro_f1"])
        logger.info("Saved summary: %s", summary_path)


if __name__ == "__main__":
    main()
