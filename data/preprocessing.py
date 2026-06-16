import librosa
import numpy as np


def load_audio_16k_fixed(path, sample_rate=16000, duration_seconds=3.0):
    target_length = int(sample_rate * duration_seconds)
    waveform, _ = librosa.load(path, sr=sample_rate, mono=True)
    waveform = waveform.astype(np.float32)

    if waveform.shape[0] < target_length:
        waveform = np.pad(waveform, (0, target_length - waveform.shape[0]), mode="constant")
    else:
        waveform = waveform[:target_length]
    return waveform


def apply_pre_emphasis(waveform, coefficient=0.97):
    if waveform.size == 0:
        return waveform
    emphasized = np.empty_like(waveform)
    emphasized[0] = waveform[0]
    emphasized[1:] = waveform[1:] - coefficient * waveform[:-1]
    return emphasized.astype(np.float32)


def maybe_pre_emphasize(waveform, enabled=False, coefficient=0.97):
    if not enabled:
        return waveform
    return apply_pre_emphasis(waveform, coefficient=coefficient)


def extract_mfcc(
    waveform,
    sample_rate=16000,
    n_mfcc=40,
    window_ms=40,
    hop_ms=10,
    window="hamming",
):
    win_length = int(sample_rate * window_ms / 1000)
    hop_length = int(sample_rate * hop_ms / 1000)
    mfcc = librosa.feature.mfcc(
        y=waveform,
        sr=sample_rate,
        n_mfcc=n_mfcc,
        n_fft=win_length,
        win_length=win_length,
        hop_length=hop_length,
        window=window,
    )
    return mfcc.astype(np.float32)


def extract_spectrogram(waveform, sample_rate=16000, n_fft=800, bins=200, hop_ms=10, window="hamming"):
    hop_length = int(sample_rate * hop_ms / 1000)
    stft = librosa.stft(y=waveform, n_fft=n_fft, hop_length=hop_length, window=window)
    spectrogram = np.abs(stft)[:bins]
    return spectrogram.astype(np.float32)
