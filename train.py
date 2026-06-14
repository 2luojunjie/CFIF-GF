import argparse
import json
from pathlib import Path

import torch
from torch import nn

from data import build_dataloader, build_loso_folds
from models import build_model
from utils import compute_classification_metrics, load_config, set_seed, setup_logger


def parse_args():
    parser = argparse.ArgumentParser(description="Train a speech emotion recognition model.")
    parser.add_argument("--config", default="configs/default.yaml", help="Path to YAML config.")
    parser.add_argument("--loso", action="store_true", help="Run 10-fold leave-one-speaker-out training.")
    return parser.parse_args()


def resolve_device(config):
    requested = config.get("device", "auto")
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def forward_model(model, batch, device):
    waveform = batch["waveform"].to(device)
    mfcc = batch["mfcc"].to(device)
    spectrogram = batch["spectrogram"].to(device)
    return model(waveform=waveform, mfcc=mfcc, spectrogram=spectrogram)


def train_one_epoch(model, dataloader, criterion, optimizer, device, num_classes, logger, log_interval):
    model.train()
    total_loss = 0.0
    all_logits = []
    all_labels = []

    for step, batch in enumerate(dataloader, start=1):
        labels = batch["label"].to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = forward_model(model, batch, device)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        all_logits.append(logits.detach().cpu())
        all_labels.append(labels.detach().cpu())

        if step % log_interval == 0:
            logger.info("step=%s loss=%.4f", step, loss.item())

    metrics = compute_classification_metrics(
        torch.cat(all_logits), torch.cat(all_labels), num_classes=num_classes
    )
    metrics["loss"] = total_loss / max(len(dataloader), 1)
    return metrics


@torch.no_grad()
def evaluate_split(model, dataloader, criterion, device, num_classes):
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
        torch.cat(all_logits), torch.cat(all_labels), num_classes=num_classes
    )
    metrics["loss"] = total_loss / max(len(dataloader), 1)
    return metrics


def build_optimizer(model, config):
    return torch.optim.AdamW(
        model.parameters(),
        lr=float(config["train"]["learning_rate"]),
        weight_decay=float(config["train"].get("weight_decay", 0.0)),
    )


def run_training(config, device, logger, output_dir, train_items=None, test_items=None, fold_info=None):
    train_loader = build_dataloader(config, split="train", shuffle=True, items=train_items)
    test_loader = None
    if test_items is not None:
        test_loader = build_dataloader(config, split="test", shuffle=False, items=test_items)

    model = build_model(config).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = build_optimizer(model, config)

    epochs = int(config["train"]["epochs"])
    num_classes = int(config["model"]["num_classes"])
    history = []
    last_test_metrics = None

    for epoch in range(1, epochs + 1):
        train_metrics = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            num_classes,
            logger,
            log_interval=int(config["train"].get("log_interval", 10)),
        )
        record = {"epoch": epoch, "train": train_metrics}

        if test_loader is not None:
            last_test_metrics = evaluate_split(model, test_loader, criterion, device, num_classes)
            record["test"] = last_test_metrics
            logger.info(
                "epoch=%s train_loss=%.4f test_wa=%.4f test_ua=%.4f test_f1=%.4f",
                epoch,
                train_metrics["loss"],
                last_test_metrics["wa"],
                last_test_metrics["ua"],
                last_test_metrics["macro_f1"],
            )
        else:
            logger.info(
                "epoch=%s loss=%.4f wa=%.4f ua=%.4f f1=%.4f",
                epoch,
                train_metrics["loss"],
                train_metrics["wa"],
                train_metrics["ua"],
                train_metrics["macro_f1"],
            )
        history.append(record)

    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "checkpoint.pt"
    torch.save({"model": model.state_dict(), "config": config, "fold": fold_info}, checkpoint_path)

    result = {
        "fold": fold_info,
        "history": history,
        "final": last_test_metrics or history[-1]["train"],
        "checkpoint": str(checkpoint_path),
    }
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, ensure_ascii=False)
    logger.info("Saved result: %s", output_dir / "metrics.json")
    return result


def run_single_training(config, device, logger):
    output_dir = Path(config.get("output_dir", "outputs"))
    return run_training(config, device, logger, output_dir=output_dir)


def run_loso_training(config, device, logger):
    folds = build_loso_folds(config["dataset"])
    output_root = Path(config.get("output_dir", "outputs")) / "loso"
    results = []

    for fold in folds:
        logger.info(
            "Starting fold %s/%s | test_speaker=%s | train=%s test=%s",
            fold["fold"],
            len(folds),
            fold["test_speaker"],
            len(fold["train_items"]),
            len(fold["test_items"]),
        )
        fold_output = output_root / f"fold_{fold['fold']:02d}_{fold['test_speaker']}"
        result = run_training(
            config,
            device,
            logger,
            output_dir=fold_output,
            train_items=fold["train_items"],
            test_items=fold["test_items"],
            fold_info={"fold": fold["fold"], "test_speaker": fold["test_speaker"]},
        )
        results.append(result)

    summary = summarize_loso_results(results)
    output_root.mkdir(parents=True, exist_ok=True)
    with (output_root / "loso_summary.json").open("w", encoding="utf-8") as handle:
        json.dump({"folds": results, "average": summary}, handle, indent=2, ensure_ascii=False)

    logger.info(
        "LOSO average | WA=%.4f UA=%.4f F1=%.4f",
        summary["wa"],
        summary["ua"],
        summary["macro_f1"],
    )
    return summary


def summarize_loso_results(results):
    keys = ["wa", "ua", "macro_f1"]
    summary = {}
    for key in keys:
        values = [result["final"][key] for result in results]
        summary[key] = sum(values) / len(values) if values else 0.0
    return summary


def main():
    args = parse_args()
    config = load_config(args.config)
    set_seed(int(config.get("seed", 42)))

    Path(config.get("output_dir", "outputs")).mkdir(parents=True, exist_ok=True)
    logger = setup_logger(log_dir=config.get("log_dir", "logs"), filename="train.log")
    device = resolve_device(config)
    logger.info("Using device: %s", device)
    logger.info("Dataset: %s | Model: %s", config["dataset"]["name"], config["model"]["name"])

    if args.loso or bool(config["dataset"].get("loso", {}).get("enabled", False)):
        run_loso_training(config, device, logger)
    else:
        run_single_training(config, device, logger)


if __name__ == "__main__":
    main()
