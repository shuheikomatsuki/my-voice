"""音声データの品質チェック。

- 壊れていないか（デコード可否・NaN・空）
- 形式（サンプルレート・長さ）
- 音割れ・音量不足・雑音 (S/N)・前後の無音・DC オフセット
- 台本とのずれ（モーラ/秒の異常、任意で ASR の一致度）
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from . import audio
from .segment import level_stats

ERROR, WARN = "error", "warn"


@dataclass
class Issue:
    level: str
    code: str
    message: str

    def __str__(self) -> str:
        mark = "✗" if self.level == ERROR else "!"
        return f"{mark} {self.code}: {self.message}"


@dataclass
class Report:
    path: str
    metrics: dict[str, Any] = field(default_factory=dict)
    issues: list[Issue] = field(default_factory=list)

    @property
    def status(self) -> str:
        if any(i.level == ERROR for i in self.issues):
            return ERROR
        if self.issues:
            return WARN
        return "ok"

    def add(self, level: str, code: str, message: str) -> None:
        self.issues.append(Issue(level, code, message))


DEFAULTS: dict[str, float] = {
    "min_duration_sec": 0.5,
    "max_duration_sec": 20.0,
    "error_duration_sec": 30.0,
    "min_sample_rate": 44100,
    "min_snr_db": 15.0,
    "warn_snr_db": 25.0,
    "min_level_dbfs": -40.0,
    "warn_level_dbfs": -32.0,
    "clip_ratio_warn": 1e-4,
    "clip_ratio_error": 1e-3,
    "edge_silence_warn_sec": 1.2,
    "mora_rate_min": 3.5,
    "mora_rate_max": 11.0,
    "asr_min_similarity": 0.55,
}


def thresholds(cfg: dict[str, Any] | None) -> dict[str, float]:
    t = dict(DEFAULTS)
    t.update(cfg or {})
    return t


def signal_metrics(x: np.ndarray, sr: int, noise_db: float | None = None) -> dict[str, Any]:
    """noise_db: 録音全体から求めた雑音床。切り出し後の短い音声は無音が少なく
    雑音を正しく測れないため、分かっていれば渡す。"""
    db = audio.frame_rms_db(x, sr)
    hop = audio.hop_samples(sr)
    st = level_stats(db)
    if noise_db is not None:
        st = level_stats(db, threshold_db=max(noise_db + 8.0, noise_db + 0.3 * (st.speech_db - noise_db)))
        st.noise_db = float(noise_db)
    snr: float | None = st.speech_db - st.noise_db
    if noise_db is None and snr < 12.0 and st.speech_db > -45.0:
        # 無音部分がほぼ無い（ぴったり切り出された）音声: 全体を発話とみなし、S/N は測れない
        st = level_stats(db, threshold_db=st.speech_db - 30.0)
        snr = None
    voiced = db > st.threshold_db
    idx = np.flatnonzero(voiced)
    if idx.size:
        lead = idx[0] * hop / sr
        trail = max(0.0, (x.size - (idx[-1] * hop + int(0.025 * sr))) / sr)
        speech_sec = idx.size * hop / sr
        active = db[voiced]
        level = float(10 * np.log10(np.mean(10 ** (active / 10))))
    else:
        lead = trail = x.size / sr
        speech_sec = 0.0
        level = float(np.max(db))
    absx = np.abs(x)
    clip_ratio = float(np.mean(absx >= 0.989))
    return {
        "duration_sec": round(x.size / sr, 3),
        "speech_sec": round(speech_sec, 3),
        "peak_dbfs": round(float(audio.to_db(absx.max())), 2),
        "level_dbfs": round(level, 2),
        "noise_dbfs": round(st.noise_db, 2) if snr is not None else None,
        "snr_db": round(snr, 2) if snr is not None else None,
        "clip_ratio": clip_ratio,
        "lead_silence_sec": round(float(lead), 3),
        "trail_silence_sec": round(float(trail), 3),
        "dc_offset": round(float(np.mean(x)), 5),
    }


def check_signal(
    x: np.ndarray, sr: int, t: dict[str, float], rep: Report, *, segmented: bool = True, noise_db: float | None = None
) -> None:
    if not np.all(np.isfinite(x)):
        rep.add(ERROR, "nan", "NaN/Inf を含む（壊れている）")
        x = np.nan_to_num(x)
    m = signal_metrics(x, sr, noise_db)
    rep.metrics.update(m)

    d = m["duration_sec"]
    if segmented:
        if d < t["min_duration_sec"]:
            rep.add(ERROR, "too_short", f"{d:.2f}s と短すぎる")
        elif d > t["error_duration_sec"]:
            rep.add(ERROR, "too_long", f"{d:.1f}s（学習の上限を超える。分割に失敗している可能性）")
        elif d > t["max_duration_sec"]:
            rep.add(WARN, "long", f"{d:.1f}s と長い（2 文がつながっていないか確認）")
        for key, label in (("lead_silence_sec", "先頭"), ("trail_silence_sec", "末尾")):
            if m[key] > t["edge_silence_warn_sec"]:
                rep.add(WARN, "edge_silence", f"{label}の無音が {m[key]:.1f}s")

    if m["speech_sec"] == 0:
        rep.add(ERROR, "no_speech", "発話が検出できない（無音・雑音のみ）")
    if m["clip_ratio"] >= t["clip_ratio_error"]:
        rep.add(ERROR, "clipping", f"音割れ {m['clip_ratio'] * 100:.2f}%（離れて録り直し）")
    elif m["clip_ratio"] >= t["clip_ratio_warn"] or m["peak_dbfs"] > -0.1:
        rep.add(WARN, "clipping", f"ピーク {m['peak_dbfs']}dBFS、音割れの疑い")
    if m["level_dbfs"] < t["min_level_dbfs"]:
        rep.add(ERROR, "too_quiet", f"声が小さすぎる（{m['level_dbfs']}dBFS）")
    elif m["level_dbfs"] < t["warn_level_dbfs"]:
        rep.add(WARN, "quiet", f"声がやや小さい（{m['level_dbfs']}dBFS）")
    if m["snr_db"] is None:
        pass  # 無音が無く測定不能（録音全体の S/N は split 時・check コマンドで確認）
    elif m["snr_db"] < t["min_snr_db"]:
        rep.add(ERROR, "noisy", f"S/N {m['snr_db']}dB（雑音が大きい）")
    elif m["snr_db"] < t["warn_snr_db"]:
        rep.add(WARN, "noisy", f"S/N {m['snr_db']}dB（やや雑音が多い）")
    if abs(m["dc_offset"]) > 0.01:
        rep.add(WARN, "dc_offset", f"DC オフセット {m['dc_offset']}")


def check_text(rep: Report, text: str, reading: str, t: dict[str, float]) -> None:
    from .corpus import count_morae

    if not text.strip():
        rep.add(ERROR, "empty_text", "テキストが空")
        return
    if reading:
        morae = count_morae(reading)
        sp = rep.metrics.get("speech_sec") or 0.0
        if sp > 0 and morae > 0:
            rate = morae / sp
            rep.metrics["mora_rate"] = round(rate, 2)
            if not (t["mora_rate_min"] <= rate <= t["mora_rate_max"]):
                rep.add(WARN, "mora_rate", f"{rate:.1f} モーラ/秒（台本とずれている／読み飛ばしの疑い）")


def check_file(
    path: str | Path,
    t: dict[str, float],
    *,
    text: str = "",
    reading: str = "",
    segmented: bool = True,
    noise_db: float | None = None,
    sr: int = audio.SR,
) -> tuple[Report, np.ndarray | None]:
    rep = Report(str(path))
    try:
        info = audio.probe(path)
        rep.metrics.update(
            {
                "codec": info.codec,
                "orig_sample_rate": info.sample_rate,
                "orig_channels": info.channels,
                "bit_rate": info.bit_rate,
            }
        )
        if info.sample_rate and info.sample_rate < t["min_sample_rate"]:
            lvl = ERROR if info.sample_rate < 24000 else WARN
            rep.add(lvl, "low_sample_rate", f"{info.sample_rate}Hz（44.1kHz 以上推奨）")
        if info.bit_rate and info.codec not in ("pcm_s16le", "pcm_s24le", "pcm_f32le", "flac", "alac") and info.bit_rate < 96000:
            rep.add(WARN, "low_bitrate", f"{info.bit_rate // 1000}kb/s（圧縮が強い）")
        x = audio.load(path, sr)
    except audio.AudioError as e:
        rep.add(ERROR, "decode", str(e))
        return rep, None
    check_signal(x, sr, t, rep, segmented=segmented, noise_db=noise_db)
    if text or reading:
        check_text(rep, text, reading, t)
    return rep, x


def summarize(reports: list[Report]) -> dict[str, Any]:
    n = len(reports)
    by = {"ok": 0, WARN: 0, ERROR: 0}
    codes: dict[str, int] = {}
    total = 0.0
    for r in reports:
        by[r.status] += 1
        total += float(r.metrics.get("duration_sec") or 0)
        for i in r.issues:
            codes[f"{i.level}:{i.code}"] = codes.get(f"{i.level}:{i.code}", 0) + 1
    snrs = [r.metrics["snr_db"] for r in reports if r.metrics.get("snr_db") is not None]
    return {
        "files": n,
        "ok": by["ok"],
        "warn": by[WARN],
        "error": by[ERROR],
        "total_minutes": round(total / 60, 2),
        "median_snr_db": round(float(np.median(snrs)), 1) if snrs else None,
        "issues": dict(sorted(codes.items(), key=lambda kv: -kv[1])),
    }
