import argparse

import torch

from models.cfif_gf import CFIFGF
from models.wavlm_att import WavLMAtt
from utils import load_config, set_seed


def parse_args():
    parser = argparse.ArgumentParser(description="Debug model forward shapes with random tensors.")
    parser.add_argument("--config", default="configs/iemocap_cfif_gf.yaml", help="YAML config path.")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size.")
    parser.add_argument("--num-classes", type=int, default=None, help="Override class count.")
    parser.add_argument("--wavlm-frames", type=int, default=149, help="Random WavLM sequence length.")
    parser.add_argument("--mfcc-frames", type=int, default=301, help="Random MFCC frame count.")
    parser.add_argument("--spec-frames", type=int, default=301, help="Random spectrogram frame count.")
    return parser.parse_args()


def shape(tensor):
    return list(tensor.shape)


def assert_shape(name, actual, expected):
    if list(actual) != list(expected):
        raise RuntimeError(f"{name} shape mismatch: expected {list(expected)}, got {list(actual)}")


def debug_wavlm_att(config, waveform, mfcc, spectrogram, wavlm_features):
    model_cfg = dict(config["model"])
    model_cfg.update({"name": "WavLM_Att", "use_offline_wavlm_features": True})
    model = WavLMAtt(model_cfg)
    model.eval()

    with torch.no_grad():
        wavlm_sequence = wavlm_features
        mfcc_features = model.mfcc_encoder(mfcc)
        cnn_features = model.spectrogram_encoder(spectrogram)
        attended_wavlm, attention = model.co_attention(wavlm_sequence, mfcc_features, cnn_features)
        fusion_features = torch.cat([attended_wavlm, mfcc_features, cnn_features], dim=-1)
        logits = model(waveform, mfcc, spectrogram, wavlm_features=wavlm_features)

    print("\n[WavLM_Att]")
    print(f"waveform: {shape(waveform)}")
    print(f"mfcc: {shape(mfcc)}")
    print(f"spectrogram: {shape(spectrogram)}")
    print(f"WavLM feature: {shape(wavlm_sequence)}")
    print(f"BiLSTM feature: {shape(mfcc_features)}")
    print(f"CNN feature: {shape(cnn_features)}")
    print(f"attention: {shape(attention)}")
    print(f"fusion feature: {shape(fusion_features)}")
    print(f"logits: {shape(logits)}")
    return logits


def debug_cfif_gf(config, waveform, mfcc, spectrogram, wavlm_features):
    model_cfg = dict(config["model"])
    model_cfg.update({"name": "CFIF-GF", "use_offline_wavlm_features": True})
    model = CFIFGF(model_cfg)
    model.eval()

    with torch.no_grad():
        wavlm_sequence = wavlm_features
        mfcc_sequence = model.mfcc_encoder(mfcc)
        tfcnn_sequence = model.spectrogram_encoder(spectrogram)
        x_mw, a_mw = model.mfcc_to_wavlm(mfcc_sequence, wavlm_sequence)
        x_sw, a_sw = model.spec_to_wavlm(tfcnn_sequence, wavlm_sequence)
        x_c = torch.cat([x_mw, x_sw], dim=1)
        fusion_sequence = model.global_fusion(x_c)
        fusion_features = fusion_sequence.mean(dim=1)
        logits = model(waveform, mfcc, spectrogram, wavlm_features=wavlm_features)

    print("\n[CFIF-GF]")
    print(f"waveform: {shape(waveform)}")
    print(f"mfcc: {shape(mfcc)}")
    print(f"spectrogram: {shape(spectrogram)}")
    print(f"WavLM feature: {shape(wavlm_sequence)}")
    print(f"BiLSTM feature: {shape(mfcc_sequence)}")
    print(f"TFCNN feature: {shape(tfcnn_sequence)}")
    print(f"A_mw: {shape(a_mw)}")
    print(f"A_sw: {shape(a_sw)}")
    print(f"X_mw: {shape(x_mw)}")
    print(f"X_sw: {shape(x_sw)}")
    print(f"fusion feature: {shape(fusion_features)}")
    print(f"logits: {shape(logits)}")
    return logits


def main():
    args = parse_args()
    config = load_config(args.config)
    set_seed(int(config.get("seed", 42)))

    batch_size = args.batch_size or int(config["train"].get("batch_size", 4))
    num_classes = args.num_classes or int(config["model"]["num_classes"])
    sample_rate = int(config["dataset"].get("sample_rate", 16000))
    duration = float(config["dataset"].get("duration_seconds", 3.0))
    samples = int(sample_rate * duration)
    spec_bins = int(config["dataset"].get("preprocessing", {}).get("spectrogram_bins", 200))
    wavlm_dim = int(config["model"].get("offline_wavlm_dim", 768))

    # waveform: [B, samples], mfcc: [B, 40, T_m], spectrogram: [B, F, T_s].
    waveform = torch.randn(batch_size, samples)
    mfcc = torch.randn(batch_size, 40, args.mfcc_frames)
    spectrogram = torch.randn(batch_size, spec_bins, args.spec_frames)
    wavlm_features = torch.randn(batch_size, args.wavlm_frames, wavlm_dim)

    wavlm_logits = debug_wavlm_att(config, waveform, mfcc, spectrogram, wavlm_features)
    cfif_logits = debug_cfif_gf(config, waveform, mfcc, spectrogram, wavlm_features)

    expected = [batch_size, num_classes]
    assert_shape("WavLM_Att logits", wavlm_logits.shape, expected)
    assert_shape("CFIF-GF logits", cfif_logits.shape, expected)
    print(f"\nOK: logits shape matches [batch_size, num_classes] = {expected}")


if __name__ == "__main__":
    main()

