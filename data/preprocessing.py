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


def extract_mfcc(waveform, sample_rate=16000, n_mfcc=40, window_ms=40, hop_ms=10):
    win_length = int(sample_rate * window_ms / 1000)
    hop_length = int(sample_rate * hop_ms / 1000)
    mfcc = librosa.feature.mfcc(
        y=waveform,
        sr=sample_rate,
        n_mfcc=n_mfcc,
        n_fft=win_length,
        win_length=win_length,
        hop_length=hop_length,
    )
    return mfcc.astype(np.float32)


def extract_spectrogram(waveform, sample_rate=16000, n_fft=800, bins=200, hop_ms=10):
    hop_length = int(sample_rate * hop_ms / 1000)
    stft = librosa.stft(y=waveform, n_fft=n_fft, hop_length=hop_length)
    spectrogram = np.abs(stft)[:bins]
    return spectrogram.astype(np.float32)

