"""faster-whisper による文字起こし（任意。`uv sync --extra asr` で有効）。"""

from __future__ import annotations

import os
from functools import lru_cache

import numpy as np

DEFAULT_MODEL = os.environ.get("MYVOICE_ASR_MODEL", "large-v3")


def available() -> bool:
    try:
        import faster_whisper  # noqa: F401
    except ImportError:
        return False
    return True


@lru_cache(maxsize=2)
def _model(name: str, device: str):
    from faster_whisper import WhisperModel

    if device == "auto":
        try:
            import ctranslate2

            device = "cuda" if ctranslate2.get_cuda_device_count() > 0 else "cpu"
        except Exception:
            device = "cpu"
    compute = "float16" if device == "cuda" else "int8"
    return WhisperModel(name, device=device, compute_type=compute)


def _resample_to_16k(x: np.ndarray, sr: int) -> np.ndarray:
    if sr == 16000:
        return x.astype(np.float32)
    n = int(round(x.size * 16000 / sr))
    t_old = np.arange(x.size) / sr
    t_new = np.arange(n) / 16000
    return np.interp(t_new, t_old, x).astype(np.float32)


def transcribe(clips: list[np.ndarray], sr: int, model: str = DEFAULT_MODEL, device: str = "auto") -> list[str]:
    if not available():
        raise RuntimeError("faster-whisper が入っていません: uv sync --extra asr")
    m = _model(model, device)
    out = []
    for x in clips:
        segs, _ = m.transcribe(_resample_to_16k(x, sr), language="ja", beam_size=5, vad_filter=False)
        out.append("".join(s.text for s in segs).strip())
    return out
