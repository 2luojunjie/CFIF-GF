import torch
from torch import nn
from transformers import WavLMModel


class MFCCBiLSTMSequenceEncoder(nn.Module):
    """Encode MFCC from [B, 40, T_m] to sequence features [B, T_m, 2H]."""
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
        sequence = mfcc.transpose(1, 2).float()
        outputs, _ = self.lstm(sequence)
        return self.dropout(outputs)


class TFCNNBranch(nn.Module):
    """One time/frequency Conv2d branch, input [B, 1, F, T]."""
    def __init__(self, kernel_size, padding, out_channels=64):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(1, out_channels, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

    def forward(self, x):
        return self.block(x)


class TFCNNSpectrogramEncoder(nn.Module):
    """Encode spectrogram [B, F, T_s] to sequence features [B, T_s', D_s]."""

    def __init__(self, branch_channels=64, conv_channels=128, output_dim=512, dropout=0.5):
        super().__init__()
        self.time_branch = TFCNNBranch(kernel_size=(5, 1), padding=(2, 0), out_channels=branch_channels)
        self.freq_branch = TFCNNBranch(kernel_size=(1, 4), padding=(0, 2), out_channels=branch_channels)
        self.post_conv = nn.Sequential(
            nn.Conv2d(branch_channels * 2, conv_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(conv_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(conv_channels, conv_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(conv_channels),
            nn.ReLU(inplace=True),
        )
        self.dropout = nn.Dropout(dropout)
        self.proj = nn.Linear(conv_channels, output_dim)
        self.output_dim = output_dim

    def forward(self, spectrogram):
        if spectrogram.ndim != 3:
            raise ValueError(
                f"Expected spectrogram shape [batch, freq_bins, frames], got {tuple(spectrogram.shape)}"
            )
        x = spectrogram.unsqueeze(1).float()
        t_features = self.time_branch(x)
        f_features = self.freq_branch(x)
        min_freq = min(t_features.shape[2], f_features.shape[2])
        min_time = min(t_features.shape[3], f_features.shape[3])
        t_features = t_features[:, :, :min_freq, :min_time]
        f_features = f_features[:, :, :min_freq, :min_time]

        features = torch.cat([t_features, f_features], dim=1)
        features = self.post_conv(features)
        features = nn.functional.adaptive_avg_pool2d(features, (1, features.shape[-1]))
        features = features.squeeze(2).transpose(1, 2)
        return self.proj(self.dropout(features))


class CrossFeatureInteractionBranch(nn.Module):
    """Source [B, T_src, D_src] attends to WavLM [B, T_w, D_w]."""
    def __init__(self, source_dim, wavlm_dim, hidden_dim, output_dim):
        super().__init__()
        self.source_proj = nn.Linear(source_dim, hidden_dim)
        self.wavlm_proj = nn.Linear(wavlm_dim, hidden_dim)
        self.score = nn.Linear(hidden_dim, 1)
        self.output_proj = nn.Linear(source_dim + wavlm_dim, output_dim)

    def forward(self, source_sequence, wavlm_sequence):
        source_hidden = self.source_proj(source_sequence).unsqueeze(2)
        wavlm_hidden = self.wavlm_proj(wavlm_sequence).unsqueeze(1)
        scores = self.score(torch.tanh(source_hidden + wavlm_hidden)).squeeze(-1)
        attention = torch.softmax(scores, dim=-1)
        attended_wavlm = torch.matmul(attention, wavlm_sequence)
        fused = torch.cat([attended_wavlm, source_sequence], dim=-1)
        return self.output_proj(fused), attention


class GlobalFusionBlock(nn.Module):
    """gMLP-style global fusion for sequence input [B, T_c, D_c]."""

    def __init__(self, input_dim, hidden_dim, dropout=0.5):
        super().__init__()
        if hidden_dim % 2 != 0:
            raise ValueError("global_fusion_hidden_dim must be divisible by 2")
        self.norm = nn.LayerNorm(input_dim)
        self.fc_in = nn.Linear(input_dim, hidden_dim)
        self.act = nn.GELU()
        half_dim = hidden_dim // 2
        self.gate_norm = nn.LayerNorm(half_dim)
        self.gate_conv = nn.Conv1d(half_dim, half_dim, kernel_size=1)
        self.fc_out = nn.Linear(half_dim, input_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        x = self.norm(x)
        x = self.act(self.fc_in(x))
        z1, z2 = x.chunk(2, dim=-1)
        z2 = self.gate_norm(z2)
        z2 = self.gate_conv(z2.transpose(1, 2)).transpose(1, 2)
        x = z1 * z2
        x = self.dropout(self.fc_out(x))
        return residual + x


class CFIFGF(nn.Module):
    """Chapter 4 cross-feature interactive fusion and global fusion model."""

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
        self.mfcc_encoder = MFCCBiLSTMSequenceEncoder(
            input_dim=int(model_cfg.get("mfcc_dim", 40)),
            hidden_size=int(model_cfg.get("mfcc_hidden_size", 256)),
            num_layers=int(model_cfg.get("mfcc_num_layers", 1)),
            dropout=dropout,
        )
        self.spectrogram_encoder = TFCNNSpectrogramEncoder(
            branch_channels=int(model_cfg.get("tfcnn_branch_channels", 64)),
            conv_channels=int(model_cfg.get("tfcnn_conv_channels", 128)),
            output_dim=int(model_cfg.get("spectrogram_feature_dim", 512)),
            dropout=dropout,
        )

        cfif_hidden_dim = int(model_cfg.get("cfif_hidden_dim", 256))
        cfif_output_dim = int(model_cfg.get("cfif_output_dim", 512))
        self.mfcc_to_wavlm = CrossFeatureInteractionBranch(
            source_dim=self.mfcc_encoder.output_dim,
            wavlm_dim=self.wavlm_dim,
            hidden_dim=cfif_hidden_dim,
            output_dim=cfif_output_dim,
        )
        self.spec_to_wavlm = CrossFeatureInteractionBranch(
            source_dim=self.spectrogram_encoder.output_dim,
            wavlm_dim=self.wavlm_dim,
            hidden_dim=cfif_hidden_dim,
            output_dim=cfif_output_dim,
        )

        self.global_fusion = GlobalFusionBlock(
            input_dim=cfif_output_dim,
            hidden_dim=int(model_cfg.get("global_fusion_hidden_dim", cfif_output_dim * 2)),
            dropout=dropout,
        )
        classifier_hidden_dim = int(model_cfg.get("classifier_hidden_dim", 256))
        num_classes = int(model_cfg["num_classes"])
        self.classifier = nn.Sequential(
            nn.Linear(cfif_output_dim, classifier_hidden_dim),
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
        mfcc_sequence = self.mfcc_encoder(mfcc)
        spectrogram_sequence = self.spectrogram_encoder(spectrogram)

        x_mw, _ = self.mfcc_to_wavlm(mfcc_sequence, wavlm_sequence)
        x_sw, _ = self.spec_to_wavlm(spectrogram_sequence, wavlm_sequence)
        x_c = torch.cat([x_mw, x_sw], dim=1)

        fused_sequence = self.global_fusion(x_c)
        fused_features = fused_sequence.mean(dim=1)
        return self.classifier(fused_features)

    @torch.no_grad()
    def predict_proba(self, waveform, mfcc, spectrogram, attention_mask=None, wavlm_features=None):
        logits = self.forward(
            waveform, mfcc, spectrogram, attention_mask=attention_mask, wavlm_features=wavlm_features
        )
        return torch.softmax(logits, dim=-1)
