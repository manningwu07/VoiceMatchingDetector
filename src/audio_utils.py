# src/audio_utils.py
import torch
import torchaudio
import librosa
import numpy as np
from silero_vad import load_silero_vad, read_audio
from src.config import SAMPLE_RATE

def load_wav(path, sr=SAMPLE_RATE, mono=True):
    """Load audio and resample to target sample rate."""
    wav, file_sr = torchaudio.load(path)
    if mono and wav.shape[0] > 1:
        wav = torch.mean(wav, dim=0, keepdim=True)
    if file_sr != sr:
        wav = torchaudio.functional.resample(wav, file_sr, sr)
    return wav.squeeze(0).numpy(), sr

def normalize_loudness(audio, sr=SAMPLE_RATE, target_loudness=-23.0):
    """Loudness normalization using pyloudnorm."""
    import pyloudnorm
    meter = pyloudnorm.Meter(sr)
    try:
        loudness = meter.integrated_loudness(audio)
        if np.isfinite(loudness):
            audio = pyloudnorm.normalize.loudness(audio, loudness, target_loudness)
        else:
            audio = audio / (np.max(np.abs(audio)) + 1e-8)
    except:
        audio = audio / (np.max(np.abs(audio)) + 1e-8)
    return audio.astype(np.float32)

def apply_vad(audio, sr=SAMPLE_RATE, frame_ms=30, threshold=0.5):
    """
    Apply Silero VAD. Returns trimmed audio.
    """
    model = load_silero_vad(onnx=False, force_onnx=False)
    wav_torch = torch.from_numpy(audio).unsqueeze(0).float()
    
    speeches = model(wav_torch, sr, return_seconds=False)
    if not speeches:
        return audio
    
    # Concatenate voiced segments
    voiced = []
    for (start, end) in speeches:
        voiced.append(audio[start:end])
    return np.concatenate(voiced) if voiced else audio

def preprocess_audio(path, sr=SAMPLE_RATE, apply_vad_flag=True):
    """
    Full pipeline: load -> normalize -> VAD -> return numpy array.
    """
    audio, sr = load_wav(path, sr=sr)
    audio = normalize_loudness(audio, sr=sr)
    if apply_vad_flag:
        audio = apply_vad(audio, sr=sr)
    return audio

def mel_spectrogram(audio, sr=SAMPLE_RATE, n_mels=64, n_fft=400, hop_length=160):
    """Compute mel-spectrogram."""
    spec = librosa.feature.melspectrogram(
        y=audio, sr=sr, n_mels=n_mels, n_fft=n_fft, hop_length=hop_length
    )
    return librosa.power_to_db(spec, ref=np.max)