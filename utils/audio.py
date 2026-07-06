import io
import numpy as np
import soundfile as sf
import librosa


SUPPORTED_FORMATS = {"wav", "flac", "ogg", "mp3", "m4a", "webm"}


def load_audio(file_bytes: bytes, target_sr: int = 16000) -> np.ndarray:
    """Load arbitrary audio bytes, return mono float32 PCM at target_sr."""
    try:
        data, sr = sf.read(io.BytesIO(file_bytes), dtype="float32", always_2d=False)
    except Exception:
        data, sr = librosa.load(io.BytesIO(file_bytes), sr=None, mono=False)
        data = data.astype(np.float32)

    if data.ndim > 1:
        data = np.mean(data, axis=1 if data.shape[1] < data.shape[0] else 0)

    if sr != target_sr:
        data = librosa.resample(data, orig_sr=sr, target_sr=target_sr)

    return data.astype(np.float32)
