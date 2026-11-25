"""無音区間で長い録音を文ごとに切り分ける（雑音レベルに合わせた自動しきい値）。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .audio import frame_rms_db, hop_samples


@dataclass
class Chunk:
    start: int  # サンプル位置（パディング込み）
    end: int
    speech_start: int  # パディング前の発話区間
    speech_end: int

    def duration(self, sr: int) -> float:
        return (self.end - self.start) / sr


@dataclass
class LevelStats:
    noise_db: float  # 雑音床（静かなフレームの 10 パーセンタイル）
    speech_db: float  # 発話レベル（95 パーセンタイル）
    threshold_db: float


def level_stats(db: np.ndarray, threshold_db: float | None = None) -> LevelStats:
    noise = float(np.percentile(db, 10))
    speech = float(np.percentile(db, 95))
    if threshold_db is None:
        threshold_db = max(noise + 8.0, noise + 0.3 * (speech - noise))
    return LevelStats(noise, speech, float(threshold_db))


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """True が連続する区間 [s, e) のリスト。"""
    if mask.size == 0:
        return []
    d = np.diff(np.concatenate([[0], mask.astype(np.int8), [0]]))
    starts = np.flatnonzero(d == 1)
    ends = np.flatnonzero(d == -1)
    return list(zip(starts.tolist(), ends.tolist()))


def detect_chunks(
    x: np.ndarray,
    sr: int,
    min_silence_sec: float = 0.9,
    min_speech_sec: float = 0.35,
    pad_sec: float = 0.15,
    threshold_db: float | None = None,
) -> tuple[list[Chunk], LevelStats]:
    db = frame_rms_db(x, sr)
    hop = hop_samples(sr)
    stats = level_stats(db, threshold_db)
    voiced = db > stats.threshold_db

    runs = _runs(voiced)
    # 短い無音（文中の息継ぎ・読点）でつながっている区間を結合
    gap = int(round(min_silence_sec / (hop / sr)))
    merged: list[list[int]] = []
    for s, e in runs:
        if merged and s - merged[-1][1] < gap:
            merged[-1][1] = e
        else:
            merged.append([s, e])

    # 発話量が少なすぎる区間（クリック音・物音）は捨てる
    min_frames = int(round(min_speech_sec / (hop / sr)))
    kept = []
    for s, e in merged:
        if int(voiced[s:e].sum()) >= min_frames:
            kept.append((s, e))

    pad = int(pad_sec * sr)
    win = int(0.025 * sr)
    chunks: list[Chunk] = []
    for i, (s, e) in enumerate(kept):
        ss = s * hop
        se = min(x.size, e * hop + win)
        lo = max(0, ss - pad)
        hi = min(x.size, se + pad)
        # 隣の区間とパディングが重ならないよう中点で切る
        if i > 0:
            prev_end = min(x.size, kept[i - 1][1] * hop + win)
            lo = max(lo, (prev_end + ss) // 2)
        if i + 1 < len(kept):
            next_start = kept[i + 1][0] * hop
            hi = min(hi, (se + next_start) // 2)
        chunks.append(Chunk(lo, hi, ss, se))
    return chunks, stats


def fade(x: np.ndarray, sr: int, sec: float = 0.01) -> np.ndarray:
    """切り口のプチノイズを防ぐ短いフェード。"""
    n = min(int(sr * sec), x.size // 2)
    if n <= 0:
        return x
    y = x.copy()
    ramp = np.linspace(0.0, 1.0, n, dtype=y.dtype)
    y[:n] *= ramp
    y[-n:] *= ramp[::-1]
    return y
