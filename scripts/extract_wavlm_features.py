import argparse
import csv
import hashlib
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import WavLMModel

from data.folds import load_all_items
from data.preprocessing import load_audio_16k_fixed
from utils import load_config, set_seed


def parse_args():
    parser = argparse.ArgumentParser(description="Extract offline WavLM sequence features.")
    parser.add_argument("--config", required=True, help="Experiment YAML config.")
    parser.add_argument("--output-dir", required=True, help="Directory for .pt WavLM feature files.")
    parser.add_argument("--output-manifest", required=True, help="CSV manifest with wavlm_path column.")
    parser.add_argument("--device", default="auto", help="auto, cuda, or cpu.")
    return parser.parse_args()


def resolve_device(requested):
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def feature_name(audio_path):
    digest = hashlib.md5(audio_path.encode("utf-8")).hexdigest()
    return f"{digest}.pt"


@torch.no_grad()
def main():
    args = parse_args()
    config = load_config(args.config)
    set_seed(int(config.get("seed", 42)))
    device = resolve_device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model = WavLMModel.from_pretrained(config["model"].get("wavlm_name", "microsoft/wavlm-base"))
    model.eval().to(device)

    items = load_all_items(config["dataset"])
    rows = []
    for item in tqdm(items, desc="Extracting WavLM"):
        waveform = load_audio_16k_fixed(
            item["path"],
            sample_rate=int(config["dataset"].get("sample_rate", 16000)),
            duration_seconds=float(config["dataset"].get("duration_seconds", 3.0)),
        )
        input_values = torch.from_numpy(waveform).unsqueeze(0).to(device)
        # last_hidden_state: [1, T_w, D_w], saved as [T_w, D_w].
        features = model(input_values=input_values).last_hidden_state.squeeze(0).cpu()
        feature_path = output_dir / feature_name(item["path"])
        torch.save(features, feature_path)
        rows.append(
            {
                "path": item["path"],
                "label": item["label"],
                "speaker_id": item["speaker_id"],
                "wavlm_path": str(feature_path),
            }
        )

    with Path(args.output_manifest).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["path", "label", "speaker_id", "wavlm_path"])
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
