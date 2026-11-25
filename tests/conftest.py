from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pytest

SR = 48000
REPO = Path(__file__).resolve().parents[1]


def voice(sec: float, f0: float = 140.0, amp: float = 0.3, seed: int = 0) -> np.ndarray:
    """発話っぽい合成音（倍音 + 音節ごとの振幅変調）。"""
    rng = np.random.default_rng(seed)
    t = np.arange(int(sec * SR)) / SR
    f = f0 * (1 + 0.05 * np.sin(2 * np.pi * 0.7 * t))
    phase = 2 * np.pi * np.cumsum(f) / SR
    x = sum(np.sin(k * phase) / k for k in range(1, 8))
    env = 0.6 + 0.4 * np.abs(np.sin(2 * np.pi * 3.5 * t + rng.uniform(0, 3)))
    x = x * env
    return (amp * x / np.max(np.abs(x))).astype(np.float32)


def silence(sec: float, noise_db: float = -65.0, seed: int = 1) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return (rng.standard_normal(int(sec * SR)) * 10 ** (noise_db / 20)).astype(np.float32)


def recording(lengths: list[float], gap: float = 1.6) -> np.ndarray:
    parts = [silence(1.0)]
    for i, L in enumerate(lengths):
        parts += [voice(L, seed=i), silence(gap, seed=100 + i)]
    x = np.concatenate(parts)
    rng = np.random.default_rng(7)
    return x + (rng.standard_normal(x.size) * 10 ** (-65 / 20)).astype(np.float32)


@pytest.fixture
def project_root(tmp_path, monkeypatch) -> Path:
    """3 文 × 1 セッションの小さなプロジェクト。"""
    root = tmp_path / "proj"
    (root / "configs").mkdir(parents=True)
    cfg = (REPO / "configs" / "project.toml").read_text(encoding="utf-8")
    cfg = cfg.replace("sentences_per_session = 25", "sentences_per_session = 3")
    (root / "configs" / "project.toml").write_text(cfg, encoding="utf-8")
    c = root / "data" / "corpus"
    c.mkdir(parents=True)
    (c / "emotion_transcript_utf8.txt").write_text(
        "EMOTION100_001:えっ嘘でしょ。,エッウソデショ。\n"
        "EMOTION100_002:シュヴァイツァーは見習うべき人間です。,シュヴァイツァーワミナラウベキニンゲンデス。\n"
        "EMOTION100_003:彼が解雇されるとは妙な話だ。,カレガカイコサレルトワミョーナハナシダ。\n",
        encoding="utf-8",
    )
    (c / "recitation_transcript_utf8.txt").write_text(
        "RECITATION324_001:女の子がキッキッ嬉しそう。,オンナノコガキッキッウレシソー。\n", encoding="utf-8"
    )
    monkeypatch.setenv("MYVOICE_ROOT", str(root))
    return root


@pytest.fixture(autouse=True)
def _need_ffmpeg():
    if not shutil.which("ffmpeg"):
        pytest.skip("ffmpeg が必要")
