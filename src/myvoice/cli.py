"""myvoice コマンド。`uv run myvoice --help` で一覧。"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

from . import audio, corpus, qc
from .project import Project


def _p() -> Project:
    return Project.load()


def _print_report(r: qc.Report, verbose: bool = False) -> None:
    mark = {"ok": "✓", "warn": "!", "error": "✗"}[r.status]
    m = r.metrics
    keys = ["duration_sec", "level_dbfs", "noise_dbfs", "snr_db", "peak_dbfs"]
    summary = "  ".join(f"{k.replace('_sec', '').replace('_dbfs', '').replace('_db', '')}={m[k]}" for k in keys if k in m)
    print(f"{mark} {r.path}\n    {summary}")
    if verbose and m.get("codec"):
        print(f"    codec={m.get('codec')} sr={m.get('orig_sample_rate')} ch={m.get('orig_channels')} bitrate={m.get('bit_rate')}")
    for i in r.issues:
        print(f"    {i}")


# ---------------------------------------------------------------------------
def cmd_doctor(a) -> int:
    p = _p()
    ok = True
    print(f"root: {p.root}")
    for b in ("uv", "ffmpeg", "ffprobe", "git"):
        path = shutil.which(b)
        print(f"{'✓' if path else '✗'} {b}: {path or '見つかりません'}")
        ok &= bool(path)
    from . import asr, irodori

    print(f"{'✓' if asr.available() else '-'} faster-whisper (任意): {'あり' if asr.available() else 'なし（uv sync --extra asr）'}")
    try:
        irodori.ensure_installed(p)
        print(f"✓ Irodori-TTS: {p.irodori_dir}")
        print(f"    {irodori.torch_info(p)}")
    except irodori.IrodoriError as e:
        print(f"- {e}")
    print(f"  推奨 device: {irodori.default_device()} / backend: {irodori.default_backend()}")
    return 0 if ok else 1


def cmd_corpus(a) -> int:
    p = _p()
    paths = corpus.fetch(p, force=a.force)
    for name, path in paths.items():
        print(f"{name}: {len(corpus.load_corpus(p, name))} 文 → {path}")
    return 0


def cmd_script(a) -> int:
    p = _p()
    phases = [a.phase] if a.phase else [ph["name"] for ph in p.phases]
    for ph in phases:
        try:
            paths = corpus.write_scripts(p, ph)
        except FileNotFoundError as e:
            print(f"[{ph}] スキップ: {e}")
            continue
        print(f"[{ph}] {len(paths)} セッション → {p.scripts_dir / ph}")
        for path in paths:
            print(f"    {path.stem}.m4a  ← {path.name} を読む")
    print(f"\n録音したら recordings/ に置き、uv run myvoice split recordings/<セッションID>.m4a")
    return 0


def cmd_check(a) -> int:
    """任意の音声ファイル（テスト録音など）を検査。"""
    p = _p()
    t = qc.thresholds(p.section("qc"))
    files: list[Path] = []
    for x in a.paths:
        x = Path(x)
        files += sorted(f for f in x.rglob("*") if audio.is_audio_file(f)) if x.is_dir() else [x]
    reports = []
    for f in files:
        rep, _ = qc.check_file(f, t, segmented=a.segmented)
        _print_report(rep, verbose=True)
        reports.append(rep)
    if len(reports) > 1:
        print(json.dumps(qc.summarize(reports), ensure_ascii=False, indent=2))
    return 1 if any(r.status == qc.ERROR for r in reports) else 0


def cmd_split(a) -> int:
    from .sessions import SplitMismatch, split_recording

    p = _p()
    rc = 0
    t = qc.thresholds(p.section("qc"))
    for rec in a.recordings:
        rec = Path(rec)
        try:
            segs = split_recording(
                p,
                rec,
                sid=a.session,
                start=a.start,
                count=a.count,
                use_asr="always" if a.asr else ("never" if a.no_asr else "auto"),
                min_silence_sec=a.min_silence,
                threshold_db=a.threshold_db,
                append=a.append,
            )
        except SplitMismatch as e:
            print(f"✗ {rec}: {e}", file=sys.stderr)
            rc = 1
            continue
        from .sessions import session_dir

        n_err = n_warn = 0
        for s in segs:
            rep, _ = qc.check_file(session_dir(p, s.session) / s.file, t, text=s.text, reading=s.reading, noise_db=s.noise_db)
            if s.asr_similarity is not None and s.asr_similarity < t["asr_min_similarity"]:
                rep.add(qc.WARN, "asr_mismatch", f"一致度 {s.asr_similarity:.2f}: 「{s.asr_text}」")
            if rep.issues:
                n_err += rep.status == qc.ERROR
                n_warn += rep.status == qc.WARN
                print(f"  {s.utt_id} ({s.duration_sec:.1f}s) {s.text[:24]}")
                for i in rep.issues:
                    print(f"      {i}")
        print(f"✓ {rec} → {len(segs)} 文（{segs[0].align}）  error={n_err} warn={n_warn}")
    return rc


def cmd_status(a) -> int:
    from .sessions import load_segments

    p = _p()
    for ph in p.phases:
        try:
            sess = corpus.sessions_for_phase(p, ph["name"])
        except FileNotFoundError as e:
            print(f"[{ph['name']}] {e}\n")
            continue
        done = 0
        total = 0
        minutes = 0.0
        print(f"[{ph['name']}] {ph['corpus']} × {', '.join(ph['styles'])}")
        for sid, utts in sess.items():
            segs = load_segments(p, sid)
            total += len(utts)
            done += len(segs)
            minutes += sum(s.duration_sec for s in segs) / 60
            mark = "✓" if len(segs) == len(utts) else ("…" if segs else " ")
            rec = [r.name for r in p.recordings_dir.glob(f"{sid}*")] if p.recordings_dir.is_dir() else []
            print(f"  {mark} {sid}  {len(segs):>3}/{len(utts)}  {'録音: ' + ', '.join(rec) if rec else ''}")
        print(f"  → {done}/{total} 文, {minutes:.1f} 分\n")
    return 0


def cmd_qc(a) -> int:
    from .dataset import qc_sessions, write_qc_report

    p = _p()
    rows = qc_sessions(p, a.sessions or None, use_asr=a.asr)
    if not rows:
        print("処理済みの音声がありません")
        return 1
    for r in rows:
        if r.report.issues or a.verbose:
            print(f"{'✗' if r.report.status == qc.ERROR else '!' if r.report.issues else '✓'} {r.seg.utt_id}  {r.seg.text[:28]}")
            for i in r.report.issues:
                print(f"      {i}")
    path = write_qc_report(p, rows, f"qc_{datetime.now():%Y%m%d_%H%M%S}")
    print(json.dumps(qc.summarize([r.report for r in rows]), ensure_ascii=False, indent=2))
    print(f"詳細: {path}\n除外したい音声は data/reject.txt に utt_id を 1 行ずつ書く")
    return 0


def cmd_build(a) -> int:
    from .dataset import build

    p = _p()
    stats = build(
        p,
        a.name,
        phases=a.phases.split(","),
        styles=a.styles.split(",") if a.styles else None,
        allow_warn=not a.strict,
        use_asr=a.asr,
    )
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    print(f"→ {p.datasets_dir / a.name}  （除外理由: excluded.tsv）")
    return 0


def cmd_irodori_setup(a) -> int:
    from . import irodori

    irodori.setup(_p(), backend=a.backend, update=a.update)
    return 0


def cmd_irodori_download(a) -> int:
    from . import irodori

    p = _p()
    print(f"ベースモデル: {irodori.download_base_model(p)}")
    irodori.download_codec(p)
    return 0


def _resolve_refs(p: Project, ref: list[str] | None, dataset: str | None, style: str, total_sec: float | None) -> list[Path]:
    """--ref の値（auto / パス複数 / 空）を参照音声のリストにする。"""
    import soundfile as sf

    from .dataset import pick_references

    vals = [r for r in (ref or []) if r]
    if vals == ["auto"]:
        if not dataset:
            raise SystemExit("--ref auto には --dataset が必要です")
        refs = pick_references(p, dataset, style, total_sec=total_sec)
        total = sum(sf.info(str(r)).duration for r in refs)
        print(f"参照音声: {len(refs)} 文・合計 {total:.1f} 秒（{', '.join(r.stem for r in refs)}）")
        return refs
    return [Path(r) for r in vals]


def cmd_train_prepare(a) -> int:
    from . import irodori

    m = irodori.prepare(_p(), a.dataset, device=a.device, dry_run=a.dry_run)
    print(f"manifest: {m}")
    return 0


def cmd_train_run(a) -> int:
    from . import irodori

    out = irodori.train(_p(), a.dataset, a.run or a.dataset, resume=a.resume, extra=a.extra, dry_run=a.dry_run)
    print(f"出力: {out}")
    return 0


def cmd_train_samples(a) -> int:
    """各チェックポイントでテスト文を合成し、聞き比べ用に並べる。"""
    from . import irodori

    p = _p()
    cks = irodori.checkpoints(p, a.run)
    if not cks:
        print(f"チェックポイントがありません: {irodori.run_dir(p, a.run)}")
        return 1
    if a.only:
        cks = [c for c in cks if c.name in a.only.split(",")]
    sentences = a.text or p.section("infer").get("test_sentences", ["こんにちは。"])
    dataset = a.dataset or (_run_dataset(p, a.run) if a.ref == ["auto"] else None)
    ref = _resolve_refs(p, a.ref, dataset, a.style, a.ref_seconds)
    caption = p.caption_for(a.style) if not a.no_caption else None
    t = qc.thresholds(p.section("qc"))
    outdir = irodori.run_dir(p, a.run) / "samples"
    targets = [None] + cks if a.with_base else cks
    # 全チェックポイント × 全文を 1 回のモデル読み込みで合成（LoRA はチェックポイントごとに切り替え）
    jobs = [
        irodori.InferJob(s, outdir / (ck.name if ck else "base") / f"{a.style}_{i:02d}.wav", ck, caption)
        for ck in targets
        for i, s in enumerate(sentences, 1)
    ]
    irodori.infer_batch(p, jobs, ref=ref, device=a.device, precision=a.precision, dry_run=a.dry_run)
    if not a.dry_run:
        for j in jobs:
            if not j.output.is_file():
                print(f"  ✗ {j.output}  生成されていません")
                continue
            rep, _ = qc.check_file(j.output, t, segmented=False)
            print(f"  {'✓' if rep.status != qc.ERROR else '✗'} {j.output}  {'; '.join(map(str, rep.issues))}")
    print(f"→ {outdir}")
    return 0


def _run_dataset(p: Project, run_name: str) -> str:
    from . import irodori

    f = irodori.run_dir(p, run_name) / "myvoice_run.txt"
    if f.is_file():
        for line in f.read_text(encoding="utf-8").splitlines():
            if line.startswith("dataset="):
                return line.split("=", 1)[1]
    raise SystemExit("--ref auto にはデータセット名が必要です（--dataset）")


def cmd_infer(a) -> int:
    from . import irodori

    p = _p()
    # (出力名, 文)。--text-file の行は「文」または「出力名<TAB>文」
    texts: list[tuple[str | None, str]] = [(None, t) for t in a.text or []]
    if a.text_file:
        for line in Path(a.text_file).read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            name, sep, body = line.partition("\t")
            texts.append((name.strip(), body.strip()) if sep else (None, line.strip()))
    if not texts:
        print("--text か --text-file を指定してください", file=sys.stderr)
        return 2
    adapter = irodori.resolve_adapter(p, a.adapter) if a.adapter else None
    if a.ref == ["auto"] and not a.dataset:
        print("--ref auto には --dataset が必要です", file=sys.stderr)
        return 2
    ref = _resolve_refs(p, a.ref, a.dataset, a.style, a.ref_seconds)
    caption = a.caption or (p.caption_for(a.style) if a.style and not a.no_caption else None)
    t = qc.thresholds(p.section("qc"))
    out = Path(a.out)
    rc = 0
    if len(texts) == 1 and out.suffix == ".wav":
        jobs = [irodori.InferJob(texts[0][1], out, adapter, caption)]
    else:
        jobs = [irodori.InferJob(text, out / f"{name or f'{i:03d}'}.wav", adapter, caption) for i, (name, text) in enumerate(texts, 1)]
    if len(jobs) == 1:
        irodori.infer(p, jobs[0].text, jobs[0].output, adapter=adapter, ref=ref, caption=caption, device=a.device, precision=a.precision, extra=a.extra, dry_run=a.dry_run)
    else:  # モデルを 1 回だけ読み込んで全文を合成
        irodori.infer_batch(p, jobs, ref=ref, device=a.device, precision=a.precision, extra=a.extra, dry_run=a.dry_run)
    if not a.dry_run:
        for j in jobs:
            if not j.output.is_file():
                print(f"✗ {j.output}  {j.text[:30]}  生成されていません")
                rc = 1
                continue
            rep, _ = qc.check_file(j.output, t, segmented=False)
            print(f"{'✓' if rep.status != qc.ERROR else '✗'} {j.output}  {j.text[:30]}  {'; '.join(map(str, rep.issues))}")
            rc |= rep.status == qc.ERROR
    return int(rc)


# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="myvoice", description="自分の声で Irodori-TTS を LoRA 学習するためのツール")
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("doctor", help="環境チェック（ffmpeg / uv / Irodori-TTS / GPU）")
    s.set_defaults(func=cmd_doctor)

    s = sub.add_parser("corpus", help="ITA コーパスをダウンロード")
    s.add_argument("--force", action="store_true")
    s.set_defaults(func=cmd_corpus)

    s = sub.add_parser("script", help="録音用の台本をセッションごとに作る")
    s.add_argument("--phase", help="phase1 / phase2 / phase3（省略で全部）")
    s.set_defaults(func=cmd_script)

    s = sub.add_parser("check", help="音声ファイルを検査（テスト録音の雑音・音量・音割れ・形式）")
    s.add_argument("paths", nargs="+")
    s.add_argument("--segmented", action="store_true", help="1 文単位の音声として長さ・前後の無音も検査")
    s.set_defaults(func=cmd_check)

    s = sub.add_parser("split", help="録音を文ごとに切り分けて台本と対応付け")
    s.add_argument("recordings", nargs="+", help="recordings/<セッションID>.m4a")
    s.add_argument("--session", help="ファイル名がセッション ID でない場合に指定")
    s.add_argument("--start", type=int, default=1, help="この録音がセッションの何文目から始まるか（1 始まり）")
    s.add_argument("--count", type=int, help="この録音に含まれる文数")
    s.add_argument("--append", action="store_true", help="既存の結果に追記（1 セッションを複数ファイルで録った場合）")
    s.add_argument("--min-silence", type=float, help="文の区切りとみなす無音の長さ（秒）")
    s.add_argument("--threshold-db", type=float, help="無音判定のしきい値 dBFS（通常は自動）")
    g = s.add_mutually_exclusive_group()
    g.add_argument("--asr", action="store_true", help="常に文字起こしで対応付け・検証する")
    g.add_argument("--no-asr", action="store_true", help="文字起こしを使わない")
    s.set_defaults(func=cmd_split)

    s = sub.add_parser("status", help="録音・処理の進み具合")
    s.set_defaults(func=cmd_status)

    s = sub.add_parser("qc", help="処理済み音声をまとめて検査しレポートを出す")
    s.add_argument("sessions", nargs="*")
    s.add_argument("--asr", action="store_true", help="文字起こしで台本との一致も検査")
    s.add_argument("-v", "--verbose", action="store_true")
    s.set_defaults(func=cmd_qc)

    s = sub.add_parser("build", help="学習用データセットを作る（QC エラーと reject.txt を除外）")
    s.add_argument("name", help="データセット名（例: v1）")
    s.add_argument("--phases", required=True, help="例: phase1,phase2")
    s.add_argument("--styles", help="スタイルを絞る（例: normal,happy）")
    s.add_argument("--strict", action="store_true", help="警告のある音声も除外")
    s.add_argument("--asr", action="store_true")
    s.set_defaults(func=cmd_build)

    s = sub.add_parser("irodori", help="Irodori-TTS の管理")
    ss = s.add_subparsers(dest="sub", required=True)
    x = ss.add_parser("setup", help="third_party/Irodori-TTS を clone して uv sync")
    x.add_argument("--backend", choices=["cu128", "cpu", "rocm", "xpu"], help="既定: Mac=cpu(MPS可), NVIDIA=cu128")
    x.add_argument("--update", action="store_true", help="git pull して更新")
    x.set_defaults(func=cmd_irodori_setup)
    x = ss.add_parser("download", help="ベースモデル（models/）と DACVAE codec（HF キャッシュ）を取得")
    x.set_defaults(func=cmd_irodori_download)

    s = sub.add_parser("train", help="LoRA 学習（GPU サーバー）")
    ss = s.add_subparsers(dest="sub", required=True)
    x = ss.add_parser("prepare", help="データセット → DACVAE 潜在 + manifest")
    x.add_argument("dataset")
    x.add_argument("--device")
    x.add_argument("--dry-run", action="store_true")
    x.set_defaults(func=cmd_train_prepare)

    x = ss.add_parser("run", help="LoRA 学習を実行（-- 以降は train.py にそのまま渡す）")
    x.add_argument("dataset")
    x.add_argument("--run", help="実験名（既定: データセット名）。outputs/runs/<run>/ に保存")
    x.add_argument("--resume", help="再開するチェックポイントのパス")
    x.add_argument("--dry-run", action="store_true")
    x.set_defaults(func=cmd_train_run, accepts_extra=True)

    x = ss.add_parser("samples", help="各チェックポイントでテスト文を合成して聞き比べ")
    x.add_argument("run")
    x.add_argument("--dataset", help="--ref auto 用（既定: run 作成時のデータセット）")
    x.add_argument("--ref", nargs="+", default=["auto"], help="参照音声（auto=データセットから複数自動選択 / パス（複数可）/ 空文字で参照なし）")
    x.add_argument("--ref-seconds", type=float, help="--ref auto で選ぶ合計秒数（既定: [infer] ref_total_sec）")
    x.add_argument("--style", default="normal")
    x.add_argument("--no-caption", action="store_true")
    x.add_argument("--text", action="append", help="テスト文（複数可。既定は project.toml）")
    x.add_argument("--only", help="対象チェックポイント名（カンマ区切り）")
    x.add_argument("--with-base", action="store_true", help="LoRA なしのベースモデルも合成")
    x.add_argument("--device")
    x.add_argument("--precision", default="fp32", choices=["fp32", "bf16"])
    x.add_argument("--dry-run", action="store_true")
    x.set_defaults(func=cmd_train_samples)

    s = sub.add_parser("infer", help="音声合成（Mac は MPS / fp32 が既定）")
    s.add_argument("--text", action="append")
    s.add_argument("--text-file", help="1 行 1 文のテキストファイル（「出力名<TAB>文」なら <出力名>.wav に保存）")
    s.add_argument("--adapter", help="LoRA: パス / run 名（最新チェックポイント）/ run名/checkpoint名")
    s.add_argument("--ref", nargs="+", help="参照音声のパス（複数可）、または auto（--dataset から複数自動選択）。省略で参照なし")
    s.add_argument("--ref-seconds", type=float, help="--ref auto で選ぶ合計秒数（既定: [infer] ref_total_sec）")
    s.add_argument("--dataset")
    s.add_argument("--style", default="normal", help="caption に使うスタイル（project.toml の [styles]）")
    s.add_argument("--caption", help="caption を直接指定")
    s.add_argument("--no-caption", action="store_true")
    s.add_argument("--out", default="outputs/infer", help=".wav パス（1 文時）またはディレクトリ")
    s.add_argument("--device", help="mps / cuda / cpu（既定: 自動）")
    s.add_argument("--precision", default="fp32", choices=["fp32", "bf16"])
    s.add_argument("--dry-run", action="store_true")
    s.set_defaults(func=cmd_infer, accepts_extra=True)  # -- 以降は infer.py にそのまま渡す
    return ap


def main(argv: list[str] | None = None) -> int:
    ap = build_parser()
    # `--` 以降は Irodori-TTS のスクリプトにそのまま渡す。argparse.REMAINDER を位置引数の後ろに置くと
    # `train run v1 --run x --dry-run` の --run 以降まで飲み込んでしまうので、自前で分ける。
    argv = list(sys.argv[1:] if argv is None else argv)
    extra: list[str] = []
    if "--" in argv:
        i = argv.index("--")
        argv, extra = argv[:i], argv[i + 1 :]
    a = ap.parse_args(argv)
    if extra and not getattr(a, "accepts_extra", False):
        ap.error("`--` 以降の引数を受け付けるのは train run と infer だけです")
    a.extra = extra
    try:
        return int(a.func(a) or 0)
    except (FileNotFoundError, KeyError, ValueError, RuntimeError) as e:
        print(f"エラー: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
