import argparse

import torch

from data import build_dataloader
from models import build_model
from utils import compute_classification_metrics, load_config, set_seed, setup_logger


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a speech emotion recognition model.")
    parser.add_argument("--config", default="configs/default.yaml", help="Path to YAML config.")
    parser.add_argument("--checkpoint", default=None, help="Optional checkpoint path.")
    return parser.parse_args()


def resolve_device(config):
    requested = config.get("device", "auto")
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


@torch.no_grad()
def evaluate(model, dataloader, device, num_classes):
    model.eval()
    all_logits = []
    all_labels = []
    for batch in dataloader:
        labels = batch["label"].to(device)
        logits = model(
            waveform=batch["waveform"].to(device),
            mfcc=batch["mfcc"].to(device),
            spectrogram=batch["spectrogram"].to(device),
        )
        all_logits.append(logits.cpu())
        all_labels.append(labels.cpu())
    return compute_classification_metrics(torch.cat(all_logits), torch.cat(all_labels), num_classes)


def main():
    args = parse_args()
    config = load_config(args.config)
    set_seed(int(config.get("seed", 42)))

    logger = setup_logger(log_dir=config.get("log_dir", "logs"), filename="evaluate.log")
    device = resolve_device(config)
    logger.info("Using device: %s", device)

    model = build_model(config).to(device)
    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(checkpoint["model"])
        logger.info("Loaded checkpoint: %s", args.checkpoint)

    dataloader = build_dataloader(config, split="test", shuffle=False)
    metrics = evaluate(model, dataloader, device, int(config["model"]["num_classes"]))
    logger.info("WA=%.4f UA=%.4f F1=%.4f", metrics["wa"], metrics["ua"], metrics["macro_f1"])


if __name__ == "__main__":
    main()
