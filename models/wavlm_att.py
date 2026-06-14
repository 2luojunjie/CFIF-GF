import torch
from torch import nn
from transformers import WavLMModel


class MFCCBiLSTMEncoder(nn.Module):
    """Encode MFCC from [B, 40, T_m] to utterance feature [B, 2H]."""
    def __init__(self, input_dim=40, hidden_size=256, num_layers=1, dropout=0.5):
        super().__init__()
        lstm_dropout = dropout if num_layers > 1 else 0.0
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=lstm_dropout,
            bidirectional=True,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.output_dim = hidden_size * 2

    def forward(self, mfcc):
        if mfcc.ndim != 3:
            raise ValueError(f"Expected MFCC shape [batch, n_mfcc, frames], got {tuple(mfcc.shape)}")
        mfcc = mfcc.transpose(1, 2).float()
        _, (hidden, _) = self.lstm(mfcc)
        forward_last = hidden[-2]
        backward_last = hidden[-1]
        features = torch.cat([forward_last, backward_last], dim=-1)
        return self.dropout(features)


class SpectrogramAlexNetEncoder(nn.Module):
    """Encode spectrogram from [B, F, T_s] to utterance feature [B, D_s]."""
    def __init__(self, output_dim=512, dropout=0.5):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.proj = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(256, output_dim),
            nn.ReLU(inplace=True),
        )
        self.output_dim = output_dim

    def forward(self, spectrogram):
        if spectrogram.ndim != 3:
            raise ValueError(
                f"Expected spectrogram shape [batch, freq_bins, frames], got {tuple(spectrogram.shape)}"
            )
        x = spectrogram.unsqueeze(1).float()
        return self.proj(self.features(x))


class CoAttentionFusion(nn.Module):
    """Fuse WavLM sequence [B, T_w, D_w] with utterance features [B, D]."""
    def __init__(self, wavlm_dim, mfcc_dim, spec_dim, attention_dim=256):
        super().__init__()
        aux_dim = mfcc_dim + spec_dim
        self.aux_proj = nn.Linear(aux_dim, attention_dim)
        self.wavlm_proj = nn.Linear(wavlm_dim, attention_dim)
        self.score = nn.Linear(attention_dim, 1)

    def forward(self, wavlm_sequence, mfcc_features, spectrogram_features, attention_mask=None):
        aux_features = torch.cat([mfcc_features, spectrogram_features], dim=-1)
        aux_context = self.aux_proj(aux_features).unsqueeze(1)
        scores = self.score(torch.tanh(self.wavlm_proj(wavlm_sequence) + aux_context)).squeeze(-1)

        if attention_mask is not None:
            mask = attention_mask.to(dtype=torch.bool, device=scores.device)
            scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)

        weights = torch.softmax(scores, dim=1)
        attended_wavlm = torch.sum(wavlm_sequence * weights.unsqueeze(-1), dim=1)
        return attended_wavlm, weights


class WavLMAtt(nn.Module):
    """Chapter 3 WavLM + MFCC BiLSTM + spectrogram CNN with co-attention fusion."""

    def __init__(self, model_cfg):
        super().__init__()
        self.use_offline_wavlm_features = bool(model_cfg.get("use_offline_wavlm_features", False))
        if self.use_offline_wavlm_features:
            self.wavlm = None
            self.wavlm_dim = int(model_cfg.get("offline_wavlm_dim", 768))
        else:
            wavlm_name = model_cfg.get("wavlm_name", "microsoft/wavlm-base")
            self.wavlm = WavLMModel.from_pretrained(wavlm_name)
            self.wavlm_dim = int(getattr(self.wavlm.config, "hidden_size", 768))

        self.freeze_wavlm = bool(model_cfg.get("freeze_wavlm", True))
        if self.wavlm is not None and self.freeze_wavlm:
            for parameter in self.wavlm.parameters():
                parameter.requires_grad = False

        dropout = float(model_cfg.get("dropout", 0.5))
        self.mfcc_encoder = MFCCBiLSTMEncoder(
            input_dim=int(model_cfg.get("mfcc_dim", 40)),
            hidden_size=int(model_cfg.get("mfcc_hidden_size", 256)),
            num_layers=int(model_cfg.get("mfcc_num_layers", 1)),
            dropout=dropout,
        )
        self.spectrogram_encoder = SpectrogramAlexNetEncoder(
            output_dim=int(model_cfg.get("spectrogram_feature_dim", 512)),
            dropout=dropout,
        )
        self.co_attention = CoAttentionFusion(
            wavlm_dim=self.wavlm_dim,
            mfcc_dim=self.mfcc_encoder.output_dim,
            spec_dim=self.spectrogram_encoder.output_dim,
            attention_dim=int(model_cfg.get("attention_dim", 256)),
        )

        classifier_input_dim = (
            self.wavlm_dim + self.mfcc_encoder.output_dim + self.spectrogram_encoder.output_dim
        )
        classifier_hidden_dim = int(model_cfg.get("classifier_hidden_dim", 256))
        num_classes = int(model_cfg["num_classes"])
        self.classifier = nn.Sequential(
            nn.Linear(classifier_input_dim, classifier_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(classifier_hidden_dim, num_classes),
        )

    def forward(self, waveform, mfcc, spectrogram, attention_mask=None, wavlm_features=None):
        # waveform: [B, samples]; optional wavlm_features: [B, T_w, D_w].
        # mfcc: [B, 40, T_m]; spectrogram: [B, F, T_s].
        if wavlm_features is not None and wavlm_features.numel() > 0:
            wavlm_sequence = wavlm_features.float()
        elif self.use_offline_wavlm_features:
            raise ValueError("model.use_offline_wavlm_features is true, but batch wavlm_features is empty.")
        elif self.freeze_wavlm:
            self.wavlm.eval()
            with torch.no_grad():
                wavlm_outputs = self.wavlm(input_values=waveform.float(), attention_mask=attention_mask)
            wavlm_sequence = wavlm_outputs.last_hidden_state
        else:
            wavlm_outputs = self.wavlm(input_values=waveform.float(), attention_mask=attention_mask)
            wavlm_sequence = wavlm_outputs.last_hidden_state
        mfcc_features = self.mfcc_encoder(mfcc)
        spectrogram_features = self.spectrogram_encoder(spectrogram)
        attended_wavlm, _ = self.co_attention(
            wavlm_sequence=wavlm_sequence,
            mfcc_features=mfcc_features,
            spectrogram_features=spectrogram_features,
            attention_mask=None,
        )
        fused_features = torch.cat([attended_wavlm, mfcc_features, spectrogram_features], dim=-1)
        return self.classifier(fused_features)

    @torch.no_grad()
    def predict_proba(self, waveform, mfcc, spectrogram, attention_mask=None, wavlm_features=None):
        logits = self.forward(
            waveform, mfcc, spectrogram, attention_mask=attention_mask, wavlm_features=wavlm_features
        )
        return torch.softmax(logits, dim=-1)
