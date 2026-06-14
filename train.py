import argparse
from pathlib import Path

import torch
from torch import nn

from data import build_dataloader
from models import build_model
from utils import compute_classification_metrics, load_config, set_seed, setup_logger


def parse_args():
    parser = argparse.ArgumentParser(description="Train a speech emotion recognition model.")
    parser.add_argument("--config", default="configs/default.yaml", help="Path to YAML config.")
    return parser.parse_args()


def resolve_device(config):
    requested = config.get("device", "auto")
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def train_one_epoch(model, dataloader, criterion, optimizer, device, num_classes, logger, log_interval):
    model.train()
    total_loss = 0.0
    all_logits = []
    all_labels = []

    for step, batch in enumerate(dataloader, start=1):
        waveform = batch["waveform"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(waveform)
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


def main():
    args = parse_args()
    config = load_config(args.config)
    set_seed(int(config.get("seed", 42)))

    Path(config.get("output_dir", "outputs")).mkdir(parents=True, exist_ok=True)
    logger = setup_logger(log_dir=config.get("log_dir", "logs"), filename="train.log")
    device = resolve_device(config)
    logger.info("Using device: %s", device)
    logger.info("Dataset: %s | Model: %s", config["dataset"]["name"], config["model"]["name"])

    train_loader = build_dataloader(config, split="train", shuffle=True)
    model = build_model(config).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["train"]["learning_rate"]),
        weight_decay=float(config["train"].get("weight_decay", 0.0)),
    )

    epochs = int(config["train"]["epochs"])
    num_classes = int(config["model"]["num_classes"])
    for epoch in range(1, epochs + 1):
        metrics = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            num_classes,
            logger,
            log_interval=int(config["train"].get("log_interval", 10)),
        )
        logger.info(
            "epoch=%s loss=%.4f accuracy=%.4f macro_f1=%.4f",
            epoch,
            metrics["loss"],
            metrics["accuracy"],
            metrics["macro_f1"],
        )

    checkpoint_path = Path(config.get("output_dir", "outputs")) / "last.pt"
    torch.save({"model": model.state_dict(), "config": config}, checkpoint_path)
    logger.info("Saved checkpoint: %s", checkpoint_path)


if __name__ == "__main__":
    main()

