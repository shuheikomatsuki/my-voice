"""録音ファイル → 文ごとの WAV（data/processed/<session>/）への変換。"""

from __future__ import annotations

import json
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from . import audio, corpus
from .align import align, similarity
from .project import Project
from .segment import detect_chunks, fade


@dataclass
class Segment:
    utt_id: str
    session: str
    corpus: str
    style: str
    corpus_id: str
    text: str
    reading: str
    file: str  # セッションディレクトリからの相対パス
    source: str  # 元の録音ファイル
    start_sec: float
    end_sec: float
    duration_sec: float
    noise_db: float  # 録音全体の雑音床
    align: str  # "count" | "asr"
    asr_text: str | None = None
    asr_similarity: float | None = None


class SplitMismatch(RuntimeError):
    pass


def session_dir(project: Project, sid: str) -> Path:
    return project.processed_dir / sid


def load_segments(project: Project, sid: str) -> list[Segment]:
    p = session_dir(project, sid) / "segments.jsonl"
    if not p.is_file():
        return []
    return [Segment(**json.loads(l)) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


def processed_sessions(project: Project) -> list[str]:
    if not project.processed_dir.is_dir():
        return []
    return sorted(p.name for p in project.processed_dir.iterdir() if (p / "segments.jsonl").is_file())


def _write_review(project: Project, sid: str, x: np.ndarray, chunks, sr: int, texts: list[str] | None) -> Path:
    d = project.review_dir / sid
    if d.exists():
        shutil.rmtree(d)
    d.mkdir(parents=True)
    rows = []
    for k, c in enumerate(chunks):
        name = f"chunk_{k + 1:03d}.wav"
        audio.save_wav(d / name, fade(x[c.start : c.end], sr), sr)
        rows.append(f"{name}\t{c.start / sr:8.2f}s\t{c.duration(sr):6.2f}s" + (f"\t{texts[k]}" if texts else ""))
    (d / "chunks.tsv").write_text("file\tstart\tduration\tasr\n" + "\n".join(rows) + "\n", encoding="utf-8")
    return d


def split_recording(
    project: Project,
    recording: Path,
    sid: str | None = None,
    start: int = 1,
    count: int | None = None,
    use_asr: str = "auto",  # "auto" | "always" | "never"
    min_silence_sec: float | None = None,
    threshold_db: float | None = None,
    append: bool = False,
) -> list[Segment]:
    """録音 1 ファイルを分割してセッションの文に対応付ける。

    start/count: 録音がセッションの一部（例: 13 文目から）しかない場合に指定（1 始まり）。
    append: 既存の segments.jsonl に追記（1 セッションを複数ファイルで録った場合）。
    """
    sid = sid or recording.stem
    utts = corpus.get_session(project, sid)
    sel = utts[start - 1 : (start - 1 + count) if count else None]
    if not sel:
        raise ValueError(f"start={start} count={count} で対象の文がありません（セッションは {len(utts)} 文）")

    cfg = project.section("split")
    sr = audio.SR
    x = audio.load(recording, sr)
    chunks, stats = detect_chunks(
        x,
        sr,
        min_silence_sec=min_silence_sec or float(cfg.get("min_silence_sec", 0.9)),
        min_speech_sec=float(cfg.get("min_speech_sec", 0.35)),
        pad_sec=float(cfg.get("pad_sec", 0.15)),
        threshold_db=threshold_db,
    )
    from . import asr as asr_mod

    want_asr = use_asr == "always" or (use_asr == "auto" and len(chunks) != len(sel) and asr_mod.available())
    asr_texts: list[str] | None = None
    if want_asr:
        asr_texts = asr_mod.transcribe([x[c.speech_start : c.speech_end] for c in chunks], sr)

    groups: list[tuple[list[int], float | None]]
    method = "count"
    if len(chunks) == len(sel) and not (use_asr == "always"):
        groups = [([k], None) for k in range(len(sel))]
        if asr_texts:
            groups = [([k], similarity(asr_texts[k], u.text, u.reading)) for k, u in enumerate(sel)]
    elif asr_texts is not None:
        method = "asr"
        try:
            res = align(asr_texts, [(u.text, u.reading) for u in sel])
        except ValueError as e:
            review = _write_review(project, sid, x, chunks, sr, asr_texts)
            raise SplitMismatch(f"{e}。区間は {review} に出力しました。") from None
        groups = [(a.chunk_indices, a.score) for a in res]
    else:
        review = _write_review(project, sid, x, chunks, sr, None)
        hint = (
            "faster-whisper を入れると自動で対応付けできます（uv sync --extra asr）。"
            if not asr_mod.available()
            else ""
        )
        raise SplitMismatch(
            f"区間数 {len(chunks)} ≠ 文数 {len(sel)}。{review} を聞いて確認してください。\n"
            f"  区間が多い → 言い直し/物音/文中の長い間。--min-silence を大きく（例 1.3）\n"
            f"  区間が少ない → 文間の間が短い。--min-silence を小さく（例 0.6）または録り直し\n"
            f"  雑音床 {stats.noise_db:.1f}dB / 発話 {stats.speech_db:.1f}dB / しきい値 {stats.threshold_db:.1f}dB"
            f"（--threshold-db で変更可）\n  {hint}"
        )

    d = session_dir(project, sid)
    wav_dir = d / "wavs"
    existing: dict[str, Segment] = {}
    if append:
        existing = {s.utt_id: s for s in load_segments(project, sid)}
    elif d.exists():
        shutil.rmtree(d)
    wav_dir.mkdir(parents=True, exist_ok=True)

    new: list[Segment] = []
    for u, (idxs, score) in zip(sel, groups):
        c0, c1 = chunks[idxs[0]], chunks[idxs[-1]]
        clip = fade(x[c0.start : c1.end], sr)
        rel = f"wavs/{u.utt_id}.wav"
        audio.save_wav(d / rel, clip, sr)
        new.append(
            Segment(
                utt_id=u.utt_id,
                session=sid,
                corpus=u.corpus,
                style=u.style,
                corpus_id=u.corpus_id,
                text=u.text,
                reading=u.reading,
                file=rel,
                source=str(recording),
                start_sec=round(c0.start / sr, 3),
                end_sec=round(c1.end / sr, 3),
                duration_sec=round((c1.end - c0.start) / sr, 3),
                noise_db=round(stats.noise_db, 2),
                align=method,
                asr_text="".join(asr_texts[k] for k in idxs) if asr_texts else None,
                asr_similarity=round(score, 3) if score is not None else None,
            )
        )
    existing.update({s.utt_id: s for s in new})
    order = {u.utt_id: i for i, u in enumerate(utts)}
    merged = sorted(existing.values(), key=lambda s: order.get(s.utt_id, 1 << 30))
    (d / "segments.jsonl").write_text(
        "".join(json.dumps(asdict(s), ensure_ascii=False) + "\n" for s in merged), encoding="utf-8"
    )
    review = project.review_dir / sid
    if review.exists():
        shutil.rmtree(review)
    return new
