"""音声の読み込み・書き出し・基本的な信号処理。

デコードは ffmpeg に任せる（iPhone の .m4a など何でも読めるようにするため）。
"""

from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf

SR = 48000  # Irodori-TTS のコーデックは 48kHz
AUDIO_EXTS = {".wav", ".m4a", ".mp3", ".flac", ".aac", ".caf", ".aif", ".aiff", ".ogg", ".opus", ".mp4", ".mov"}


class AudioError(RuntimeError):
    pass


def _require(binary: str) -> str:
    path = shutil.which(binary)
    if not path:
        raise AudioError(f"{binary} が見つかりません（Mac: brew install ffmpeg / Ubuntu: apt install ffmpeg）")
    return path


@dataclass
class ProbeInfo:
    codec: str | None
    sample_rate: int | None
    channels: int | None
    bit_rate: int | None
    duration: float | None
    bits_per_sample: int | None

    def describe(self) -> str:
        br = f"{self.bit_rate // 1000}kb/s" if self.bit_rate else "?"
        return (
            f"codec={self.codec} sr={self.sample_rate}Hz ch={self.channels} "
            f"bitrate={br} dur={self.duration:.2f}s" if self.duration is not None else f"codec={self.codec}"
        )


def probe(path: str | Path) -> ProbeInfo:
    """ffprobe で元ファイルの形式を調べる（サンプルレート・チャンネル数・ビットレートなど）。"""
    ffprobe = _require("ffprobe")
    proc = subprocess.run(
        [ffprobe, "-v", "error", "-print_format", "json", "-show_streams", "-show_format", str(path)],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        raise AudioError(f"ffprobe に失敗: {path}: {proc.stderr.strip()}")
    info = json.loads(proc.stdout or "{}")
    streams = [s for s in info.get("streams", []) if s.get("codec_type") == "audio"]
    if not streams:
        raise AudioError(f"音声ストリームがありません: {path}")
    s = streams[0]
    fmt = info.get("format", {})

    def _int(v):
        try:
            return int(v)
        except (TypeError, ValueError):
            return None

    def _float(v):
        try:
            return float(v)
        except (TypeError, ValueError):
            return None

    return ProbeInfo(
        codec=s.get("codec_name"),
        sample_rate=_int(s.get("sample_rate")),
        channels=_int(s.get("channels")),
        bit_rate=_int(s.get("bit_rate")) or _int(fmt.get("bit_rate")),
        duration=_float(s.get("duration")) or _float(fmt.get("duration")),
        bits_per_sample=_int(s.get("bits_per_raw_sample")) or _int(s.get("bits_per_sample")) or None,
    )


def load(path: str | Path, sr: int = SR) -> np.ndarray:
    """任意形式の音声をモノラル float32・指定サンプルレートで読み込む。"""
    ffmpeg = _require("ffmpeg")
    proc = subprocess.run(
        [ffmpeg, "-nostdin", "-v", "error", "-i", str(path), "-ac", "1", "-ar", str(sr), "-f", "f32le", "-"],
        capture_output=True,
    )
    if proc.returncode != 0:
        raise AudioError(f"デコードに失敗（壊れている可能性）: {path}: {proc.stderr.decode(errors='replace').strip()}")
    x = np.frombuffer(proc.stdout, dtype=np.float32).copy()
    if x.size == 0:
        raise AudioError(f"音声が空です: {path}")
    return x


def save_wav(path: str | Path, x: np.ndarray, sr: int = SR) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), np.clip(x, -1.0, 1.0), sr, subtype="PCM_24")


def to_db(v: np.ndarray | float, floor: float = 1e-10) -> np.ndarray:
    return 20.0 * np.log10(np.maximum(np.abs(v), floor))


def frame_rms_db(x: np.ndarray, sr: int, win_sec: float = 0.025, hop_sec: float = 0.010) -> np.ndarray:
    """短時間 RMS (dBFS) の系列。"""
    win = max(1, int(sr * win_sec))
    hop = max(1, int(sr * hop_sec))
    if x.size < win:
        return to_db(np.sqrt(np.mean(x.astype(np.float64) ** 2) + 1e-20))[None]
    n = 1 + (x.size - win) // hop
    # 累積和で高速に RMS を求める
    sq = np.concatenate([[0.0], np.cumsum(x.astype(np.float64) ** 2)])
    starts = np.arange(n) * hop
    energy = (sq[starts + win] - sq[starts]) / win
    return to_db(np.sqrt(energy + 1e-20))


def hop_samples(sr: int, hop_sec: float = 0.010) -> int:
    return max(1, int(sr * hop_sec))


def is_audio_file(p: Path) -> bool:
    return p.is_file() and p.suffix.lower() in AUDIO_EXTS and not p.name.startswith(".")
