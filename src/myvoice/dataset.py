"""処理済みセッションから学習用データセット（HF audiofolder 形式）を作る。

出力: data/datasets/<name>/
    *.wav
    metadata.csv  (file_name,text,caption,speaker)  ← Irodori-TTS prepare_manifest.py が読む
    stats.json / excluded.tsv
"""

from __future__ import annotations

import csv
import json
import shutil
from dataclasses import dataclass
from pathlib import Path

from . import qc
from .project import Project
from .sessions import Segment, load_segments, processed_sessions, session_dir


@dataclass
class QcRow:
    seg: Segment
    report: qc.Report


def qc_sessions(project: Project, sids: list[str] | None = None, use_asr: bool = False) -> list[QcRow]:
    t = qc.thresholds(project.section("qc"))
    rows = []
    sids = sids or processed_sessions(project)
    asr_texts: dict[str, str] = {}
    if use_asr:
        from . import asr

        from .audio import SR, load

        segs = [s for sid in sids for s in load_segments(project, sid) if s.asr_text is None]
        if segs:
            texts = asr.transcribe([load(session_dir(project, s.session) / s.file, SR) for s in segs], SR)
            asr_texts = {s.utt_id: tx for s, tx in zip(segs, texts)}
    from .align import similarity

    for sid in sids:
        for s in load_segments(project, sid):
            rep, _ = qc.check_file(
                session_dir(project, sid) / s.file, t, text=s.text, reading=s.reading, noise_db=s.noise_db
            )
            rep.metrics["utt_id"] = s.utt_id
            hyp = s.asr_text if s.asr_text is not None else asr_texts.get(s.utt_id)
            if hyp is not None:
                sim = s.asr_similarity if s.asr_similarity is not None else similarity(hyp, s.text, s.reading)
                rep.metrics["asr_text"] = hyp
                rep.metrics["asr_similarity"] = round(sim, 3)
                if sim < t["asr_min_similarity"]:
                    rep.add(qc.WARN, "asr_mismatch", f"文字起こしと台本の一致度 {sim:.2f}: 「{hyp}」")
            rows.append(QcRow(s, rep))
    return rows


def write_qc_report(project: Project, rows: list[QcRow], name: str = "qc") -> Path:
    project.reports_dir.mkdir(parents=True, exist_ok=True)
    path = project.reports_dir / f"{name}.tsv"
    cols = ["utt_id", "status", "duration_sec", "level_dbfs", "snr_db", "peak_dbfs", "mora_rate", "asr_similarity"]
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow([*cols, "issues", "text", "file"])
        for r in rows:
            m = r.report.metrics
            w.writerow(
                [r.seg.utt_id, r.report.status, *[m.get(c, "") for c in cols[2:]]]
                + ["; ".join(str(i) for i in r.report.issues), r.seg.text, r.report.path]
            )
    return path


def build(
    project: Project,
    name: str,
    phases: list[str],
    styles: list[str] | None = None,
    allow_warn: bool = True,
    use_asr: bool = False,
) -> dict:
    """phases に含まれる (corpus, style) の処理済み音声を集め、QC エラーと reject.txt を除外して書き出す。"""
    wanted: set[tuple[str, str]] = set()
    for ph in phases:
        p = project.phase(ph)
        for st in p["styles"]:
            if styles is None or st in styles:
                wanted.add((p["corpus"], st))

    sids = [sid for sid in processed_sessions(project) if any(sid.startswith(f"{c}-{s}-") for c, s in wanted)]
    if not sids:
        raise RuntimeError(f"対象の処理済みセッションがありません（phases={phases}）。先に myvoice split を実行してください。")
    rows = qc_sessions(project, sids, use_asr=use_asr)
    rejected = project.rejected_ids()

    out = project.datasets_dir / name
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)

    kept: list[QcRow] = []
    excluded: list[tuple[str, str]] = []
    for r in rows:
        if r.seg.utt_id in rejected:
            excluded.append((r.seg.utt_id, "reject.txt"))
        elif r.report.status == qc.ERROR or (not allow_warn and r.report.status == qc.WARN):
            excluded.append((r.seg.utt_id, "; ".join(str(i) for i in r.report.issues)))
        else:
            kept.append(r)
    if not kept:
        raise RuntimeError("QC を通過した音声がありません。data/reports を確認してください。")

    with open(out / "metadata.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["file_name", "text", "caption", "speaker", "style", "utt_id"])
        for r in kept:
            fname = f"{r.seg.utt_id}.wav"
            shutil.copy2(session_dir(project, r.seg.session) / r.seg.file, out / fname)
            w.writerow([fname, r.seg.text, project.caption_for(r.seg.style), project.speaker_id, r.seg.style, r.seg.utt_id])

    with open(out / "excluded.tsv", "w", encoding="utf-8") as f:
        f.write("utt_id\treason\n" + "".join(f"{u}\t{why}\n" for u, why in excluded))

    total = sum(float(r.report.metrics.get("duration_sec", 0)) for r in kept)
    by_style: dict[str, int] = {}
    for r in kept:
        by_style[r.seg.style] = by_style.get(r.seg.style, 0) + 1
    stats = {
        "name": name,
        "phases": phases,
        "utterances": len(kept),
        "excluded": len(excluded),
        "total_minutes": round(total / 60, 2),
        "by_style": by_style,
        "sessions": sids,
    }
    (out / "stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    write_qc_report(project, rows, f"qc_{name}")
    return stats


def _reference_candidates(
    project: Project, dataset: str, style: str, min_sec: float, max_sec: float
) -> list[tuple[float, float, Path]]:
    """(score, duration, path) を score の高い順に返す。score は S/N から問題の数を引いたもの。"""
    d = project.datasets_dir / dataset
    meta = d / "metadata.csv"
    if not meta.is_file():
        raise FileNotFoundError(f"データセットがありません: {d}")
    t = qc.thresholds(project.section("qc"))
    exclude = set(project.section("infer").get("ref_exclude", []))
    cands: list[tuple[float, float, Path]] = []
    with open(meta, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("style", "normal") != style or row.get("utt_id", Path(row["file_name"]).stem) in exclude:
                continue
            p = d / row["file_name"]
            rep, _ = qc.check_file(p, t)
            dur = float(rep.metrics.get("duration_sec", 0))
            if rep.status == qc.ERROR or not (min_sec <= dur <= max_sec):
                continue
            cands.append((rep.metrics.get("snr_db", 0) - 5 * len(rep.issues), dur, p))
    if not cands:
        raise RuntimeError(f"style={style} で参照音声に使える {min_sec}〜{max_sec}s の音声がありません")
    cands.sort(key=lambda c: (-c[0], c[2].name))
    return cands


def pick_reference(project: Project, dataset: str, style: str = "normal", min_sec: float = 3.0, max_sec: float = 10.0) -> Path:
    """推論用の参照音声を 1 つ自動選択（指定スタイル・長さ適度・S/N 最良）。"""
    return _reference_candidates(project, dataset, style, min_sec, max_sec)[0][2]


def pick_references(
    project: Project,
    dataset: str,
    style: str = "normal",
    total_sec: float | None = None,
    min_sec: float = 2.0,
    max_sec: float = 10.0,
) -> list[Path]:
    """推論用の参照音声を複数選ぶ（S/N の良い順に、合計 total_sec に達するまで）。

    Irodori-TTS v4 は学習時に同じ話者の別の文をつなげた長い参照（1〜120 秒）を使っており、
    公式も「短い文を複数、合計 30 秒程度」を推奨している。既定は [infer] ref_total_sec。
    """
    if total_sec is None:
        total_sec = float(project.section("infer").get("ref_total_sec", 30.0))
    chosen: list[Path] = []
    total = 0.0
    for _, dur, p in _reference_candidates(project, dataset, style, min_sec, max_sec):
        chosen.append(p)
        total += dur
        if total >= total_sec:
            break
    return chosen
