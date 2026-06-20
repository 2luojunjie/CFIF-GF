import torch
import torch.nn.functional as F
from torch import nn
from transformers import AutoModel


class MFCCBiLSTMEncoder(nn.Module):
    """Encode MFCC [B, 40, T_m] as the sequence X'_M [B, T_m, 2H]."""

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
        sequence = mfcc.transpose(1, 2).float()  # [B, T_m, 40]
        sequence, _ = self.lstm(sequence)  # [B, T_m, 2H]
        return self.dropout(sequence)


class SpectrogramAlexNetEncoder(nn.Module):
    """Encode spectrogram [B, F, T_s] as the sequence X'_S [B, T'_s, D_s]."""

    def __init__(self, output_dim=512, fc1_dim=1024, fc2_dim=512, output_frames=8, dropout=0.5):
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
        )
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, int(output_frames)))
        # The paper specifies three fully connected layers after five convolutions.
        # They are applied to every remaining temporal position to retain X'_S as a sequence.
        self.fc = nn.Sequential(
            nn.Linear(256, fc1_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(fc1_dim, fc2_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(fc2_dim, output_dim),
        )
        self.output_dim = output_dim

    def forward(self, spectrogram):
        if spectrogram.ndim != 3:
            raise ValueError(
                f"Expected spectrogram shape [batch, freq_bins, frames], got {tuple(spectrogram.shape)}"
            )
        feature_map = self.features(spectrogram.unsqueeze(1).float())  # [B, 256, F', T'_s]
        feature_map = self.adaptive_pool(feature_map)  # [B, 256, 1, T_a]
        sequence = feature_map.squeeze(2).transpose(1, 2)  # [B, T_a, 256]
        return self.fc(sequence)  # [B, T'_s, D_s]


class CoAttentionFusion(nn.Module):
    """Implement equations 3.4-3.5 using X'_M and X'_S to weight X'_W."""

    def __init__(
        self,
        wavlm_dim,
        mfcc_dim,
        spec_dim,
        expected_wavlm_frames=149,
        source_frames=8,
        output_dim=512,
        dropout=0.5,
        normalization="softmax",
    ):
        super().__init__()
        if expected_wavlm_frames <= 0 or source_frames <= 0:
            raise ValueError("expected_wavlm_frames and source_frames must be positive")
        if normalization not in {"softmax", "sigmoid", "none"}:
            raise ValueError("attention normalization must be one of: softmax, sigmoid, none")

        self.source_frames = int(source_frames)
        self.expected_wavlm_frames = int(expected_wavlm_frames)
        self.normalization = normalization
        attention_input_dim = self.source_frames * (mfcc_dim + spec_dim)

        # Figure 3.6: Concat -> Dropout -> Linear -> X'_att.
        self.attention_dropout = nn.Dropout(dropout)
        self.attention_linear = nn.Linear(attention_input_dim, self.expected_wavlm_frames)
        # Figure 3.6: weighted WavLM -> Dropout -> Linear -> X''_W.
        self.wavlm_dropout = nn.Dropout(dropout)
        self.wavlm_linear = nn.Linear(wavlm_dim, output_dim)
        self.output_dim = output_dim

    @staticmethod
    def _pool_time(sequence, output_frames):
        # [B, T, D] -> [B, output_frames, D]
        return F.adaptive_avg_pool1d(sequence.transpose(1, 2), output_frames).transpose(1, 2)

    def forward(self, wavlm_sequence, mfcc_sequence, spectrogram_sequence, attention_mask=None):
        if wavlm_sequence.ndim != 3 or mfcc_sequence.ndim != 3 or spectrogram_sequence.ndim != 3:
            raise ValueError("Co-attention inputs must all have shape [batch, frames, feature_dim]")

        mfcc_aligned = self._pool_time(mfcc_sequence, self.source_frames)
        spec_aligned = self._pool_time(spectrogram_sequence, self.source_frames)
        auxiliary = torch.cat([mfcc_aligned, spec_aligned], dim=-1)  # [B, T_a, D_m + D_s]
        scores = self.attention_linear(self.attention_dropout(auxiliary.flatten(start_dim=1)))

        wavlm_frames = wavlm_sequence.size(1)
        if scores.size(1) != wavlm_frames:
            scores = F.interpolate(
                scores.unsqueeze(1), size=wavlm_frames, mode="linear", align_corners=False
            ).squeeze(1)

        if attention_mask is not None:
            mask = attention_mask.to(dtype=torch.bool, device=scores.device)
            if mask.shape != scores.shape:
                mask = F.interpolate(mask.float().unsqueeze(1), size=wavlm_frames, mode="nearest").squeeze(1).bool()
        else:
            mask = None

        if self.normalization == "softmax":
            if mask is not None:
                scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
            weights = torch.softmax(scores, dim=1)
        elif self.normalization == "sigmoid":
            weights = torch.sigmoid(scores)
            if mask is not None:
                weights = weights * mask
            weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-8)
        else:
            weights = scores if mask is None else scores * mask

        weighted_wavlm = torch.sum(wavlm_sequence * weights.unsqueeze(-1), dim=1)  # [B, D_w]
        attended_wavlm = self.wavlm_linear(self.wavlm_dropout(weighted_wavlm))  # [B, D'_w]
        return attended_wavlm, weights


class WavLMAtt(nn.Module):
    """Chapter 3 WavLM, BiLSTM, AlexNet and co-attention SER model."""

    def __init__(self, model_cfg):
        super().__init__()
        self.use_offline_wavlm_features = bool(model_cfg.get("use_offline_wavlm_features", False))
        if self.use_offline_wavlm_features:
            self.wavlm = None
            self.wavlm_dim = int(model_cfg.get("offline_wavlm_dim", 768))
        else:
            wavlm_name = model_cfg.get("wavlm_name", "microsoft/wavlm-base")
            self.wavlm = AutoModel.from_pretrained(wavlm_name)
            self.wavlm_dim = int(getattr(self.wavlm.config, "hidden_size", 768))

        self.freeze_wavlm = bool(model_cfg.get("freeze_wavlm", False))
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
            fc1_dim=int(model_cfg.get("spectrogram_fc1_dim", 1024)),
            fc2_dim=int(model_cfg.get("spectrogram_fc2_dim", 512)),
            output_frames=int(model_cfg.get("spectrogram_output_frames", 8)),
            dropout=dropout,
        )
        self.co_attention = CoAttentionFusion(
            wavlm_dim=self.wavlm_dim,
            mfcc_dim=self.mfcc_encoder.output_dim,
            spec_dim=self.spectrogram_encoder.output_dim,
            expected_wavlm_frames=int(model_cfg.get("expected_wavlm_frames", 149)),
            source_frames=int(model_cfg.get("attention_source_frames", 8)),
            output_dim=int(model_cfg.get("attended_wavlm_dim", 512)),
            dropout=dropout,
            normalization=model_cfg.get("attention_normalization", "none"),
        )

        self.fusion_mode = model_cfg.get("fusion_mode", "co_attention")
        fusion_dim = int(model_cfg.get("fusion_dim", 512))
        self.wavlm_proj = nn.Linear(self.wavlm_dim, fusion_dim)
        self.mfcc_proj = nn.Linear(self.mfcc_encoder.output_dim, fusion_dim)
        self.spec_proj = nn.Linear(self.spectrogram_encoder.output_dim, fusion_dim)
        self.mha = nn.MultiheadAttention(
            embed_dim=fusion_dim,
            num_heads=int(model_cfg.get("mha_num_heads", 4)),
            dropout=dropout,
            batch_first=True,
        )

        if self.fusion_mode == "wavlm_only":
            classifier_input_dim = self.wavlm_dim
        elif self.fusion_mode == "mha_fusion":
            classifier_input_dim = fusion_dim
        else:
            wavlm_fusion_dim = (
                self.wavlm_dim if self.fusion_mode == "concat_fusion" else self.co_attention.output_dim
            )
            classifier_input_dim = (
                wavlm_fusion_dim + self.mfcc_encoder.output_dim + self.spectrogram_encoder.output_dim
            )

        # Figure 3.6: final Concat -> Dropout -> Linear. Softmax is applied only for inference.
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(classifier_input_dim, int(model_cfg["num_classes"])),
        )

    def _extract_wavlm(self, waveform, attention_mask, wavlm_features):
        if wavlm_features is not None and wavlm_features.numel() > 0:
            return wavlm_features.float()
        if self.use_offline_wavlm_features:
            raise ValueError("model.use_offline_wavlm_features is true, but batch wavlm_features is empty.")
        if self.freeze_wavlm:
            self.wavlm.eval()
            with torch.no_grad():
                return self.wavlm(
                    input_values=waveform.float(), attention_mask=attention_mask
                ).last_hidden_state
        return self.wavlm(input_values=waveform.float(), attention_mask=attention_mask).last_hidden_state

    def forward(self, waveform, mfcc, spectrogram, attention_mask=None, wavlm_features=None):
        # waveform [B, samples], MFCC [B, 40, T_m], spectrogram [B, F, T_s].
        wavlm_sequence = self._extract_wavlm(waveform, attention_mask, wavlm_features)  # [B, T_w, D_w]
        mfcc_sequence = self.mfcc_encoder(mfcc)  # [B, T_m, D_m]
        spectrogram_sequence = self.spectrogram_encoder(spectrogram)  # [B, T'_s, D_s]
        fused_features = self._fuse_features(
            wavlm_sequence, mfcc_sequence, spectrogram_sequence, attention_mask=None
        )
        return self.classifier(fused_features)  # [B, num_classes]

    def _fuse_features(self, wavlm_sequence, mfcc_sequence, spectrogram_sequence, attention_mask=None):
        mfcc_features = mfcc_sequence.mean(dim=1)  # [B, D_m]
        spectrogram_features = spectrogram_sequence.mean(dim=1)  # [B, D_s]

        if self.fusion_mode == "wavlm_only":
            return wavlm_sequence.mean(dim=1)
        if self.fusion_mode == "concat_fusion":
            return torch.cat(
                [wavlm_sequence.mean(dim=1), mfcc_features, spectrogram_features], dim=-1
            )
        if self.fusion_mode == "mha_fusion":
            tokens = torch.cat(
                [
                    self.wavlm_proj(wavlm_sequence.mean(dim=1, keepdim=True)),
                    self.mfcc_proj(mfcc_features).unsqueeze(1),
                    self.spec_proj(spectrogram_features).unsqueeze(1),
                ],
                dim=1,
            )
            fused, _ = self.mha(tokens, tokens, tokens)
            return fused.mean(dim=1)
        if self.fusion_mode != "co_attention":
            raise ValueError(f"Unsupported WavLM_Att fusion mode: {self.fusion_mode}")

        attended_wavlm, _ = self.co_attention(
            wavlm_sequence,
            mfcc_sequence,
            spectrogram_sequence,
            attention_mask=attention_mask,
        )
        return torch.cat([attended_wavlm, mfcc_features, spectrogram_features], dim=-1)

    @torch.no_grad()
    def predict_proba(self, waveform, mfcc, spectrogram, attention_mask=None, wavlm_features=None):
        logits = self.forward(
            waveform, mfcc, spectrogram, attention_mask=attention_mask, wavlm_features=wavlm_features
        )
        return torch.softmax(logits, dim=-1)
