from __future__ import annotations

import csv
import subprocess
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
from conftest import SR, recording, silence, voice

from myvoice import qc
from myvoice.align import align, normalize, similarity
from myvoice.corpus import count_morae, get_session, parse_ita
from myvoice.dataset import build, pick_reference, pick_references
from myvoice.project import Project
from myvoice.segment import detect_chunks
from myvoice.sessions import SplitMismatch, load_segments, split_recording


# ---- segment ------------------------------------------------------------------
def test_detect_chunks_counts_sentences():
    x = recording([1.5, 3.0, 2.2, 0.9, 4.0])
    chunks, st = detect_chunks(x, SR)
    assert len(chunks) == 5
    assert st.speech_db - st.noise_db > 30
    durs = [c.duration(SR) for c in chunks]
    assert durs[1] == pytest.approx(3.0 + 0.3, abs=0.15)


def test_short_pause_inside_sentence_is_not_a_split():
    x = np.concatenate([silence(1), voice(1.2), silence(0.4), voice(1.5), silence(1.6), voice(1.0), silence(1)])
    chunks, _ = detect_chunks(x, SR, min_silence_sec=0.9)
    assert len(chunks) == 2


def test_click_is_ignored():
    click = np.zeros(int(0.05 * SR), dtype=np.float32)
    click[::50] = 0.5
    x = np.concatenate([silence(1), voice(1.5), silence(1.5), click, silence(1.5), voice(1.5), silence(1)])
    chunks, _ = detect_chunks(x, SR)
    assert len(chunks) == 2


# ---- align --------------------------------------------------------------------
SENTS = [("えっ嘘でしょ。", "エッウソデショ。"), ("彼が解雇されるとは妙な話だ。", "カレガカイコサレルトワミョーナハナシダ。"), ("時間はあるんだから", "ジカンワアルンダカラ")]


def test_align_skips_retake_and_prefers_later_take():
    hyps = ["えっ嘘でしょ", "彼が解雇", "彼が解雇されるとは妙な話だ", "時間はあるんだから"]
    res = align(hyps, SENTS)
    assert [a.chunk_indices for a in res] == [[0], [2], [3]]


def test_align_merges_split_sentence_and_drops_noise():
    hyps = ["えっ嘘でしょ", "ん", "彼が解雇されると", "は妙な話だ", "時間はあるんだから"]
    res = align(hyps, SENTS)
    assert [a.chunk_indices for a in res] == [[0], [2, 3], [4]]
    assert all(a.score > 0.8 for a in res)


def test_similarity_uses_reading():
    assert similarity("じかんはあるんだから", "時間はあるんだから", "ジカンワアルンダカラ") > 0.8
    assert normalize("カタカナ、。！") == "かたかな"


def test_corpus_parse_and_morae():
    s = parse_ita("EMOTION100_001:えっ嘘でしょ。,エッウソデショ。\n")[0]
    assert s.number == 1 and s.text == "えっ嘘でしょ。"
    assert count_morae("エッウソデショ。") == 6  # エ・ッ・ウ・ソ・デ・ショ
    assert count_morae("キャッキャ") == 3


# ---- qc -----------------------------------------------------------------------
def _write(path, x):
    sf.write(str(path), x, SR, subtype="PCM_24")
    return path


def test_qc_detects_problems(tmp_path):
    t = qc.thresholds(None)
    good = _write(tmp_path / "good.wav", np.concatenate([silence(0.2), voice(2.0), silence(0.2)]))
    rep, _ = qc.check_file(good, t, noise_db=-65)
    assert rep.status == "ok", rep.issues

    clipped = _write(tmp_path / "clip.wav", np.clip(voice(2.0, amp=1.0) * 3, -1, 1))
    rep, _ = qc.check_file(clipped, t)
    assert any(i.code == "clipping" and i.level == "error" for i in rep.issues)

    quiet = _write(tmp_path / "quiet.wav", np.concatenate([silence(0.3, -90), voice(2.0, amp=0.003), silence(0.3, -90)]))
    rep, _ = qc.check_file(quiet, t)
    assert any(i.code == "too_quiet" for i in rep.issues)

    rng = np.random.default_rng(0)
    noisy = _write(tmp_path / "noisy.wav", voice(2.0) + (rng.standard_normal(2 * SR) * 0.05).astype(np.float32))
    rep, _ = qc.check_file(noisy, t, noise_db=-26)
    assert any(i.code == "noisy" for i in rep.issues)

    broken = tmp_path / "broken.wav"
    broken.write_bytes(b"RIFF\x00\x00garbage")
    rep, _ = qc.check_file(broken, t)
    assert rep.status == "error" and rep.issues[0].code == "decode"

    lowsr = tmp_path / "lowsr.wav"
    sf.write(str(lowsr), voice(1.5)[::3], 16000)
    rep, _ = qc.check_file(lowsr, t)
    assert any(i.code == "low_sample_rate" for i in rep.issues)


def test_qc_flags_text_mismatch(tmp_path):
    t = qc.thresholds(None)
    # 6 モーラの文に 6 秒の音声 → 1 モーラ/秒は遅すぎる（ずれの疑い）
    p = _write(tmp_path / "a.wav", voice(6.0))
    rep, _ = qc.check_file(p, t, text="えっ嘘でしょ。", reading="エッウソデショ。")
    assert any(i.code == "mora_rate" for i in rep.issues)


# ---- end-to-end ---------------------------------------------------------------
def _m4a(path, x):
    wav = path.with_suffix(".tmp.wav")
    sf.write(str(wav), x, SR)
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(wav), "-c:a", "aac", "-b:a", "192k", str(path)], check=True)
    wav.unlink()
    return path


def test_split_and_build(project_root):
    p = Project.load()
    rec = p.recordings_dir
    rec.mkdir()
    # 3 文: 約 6, 15, 13 モーラ → 自然な長さ
    _m4a(rec / "emotion-normal-s01.m4a", recording([1.0, 2.2, 2.0]))
    segs = split_recording(p, rec / "emotion-normal-s01.m4a", use_asr="never")
    assert [s.utt_id for s in segs] == [u.utt_id for u in get_session(p, "emotion-normal-s01")]
    assert all((p.processed_dir / "emotion-normal-s01" / s.file).is_file() for s in segs)

    stats = build(p, "v1", phases=["phase1"])
    assert stats["utterances"] == 3
    with open(p.datasets_dir / "v1" / "metadata.csv", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert rows[0]["file_name"] == "EMOTION100_001_normal.wav"
    assert rows[0]["caption"] == p.caption_for("normal")
    assert rows[0]["speaker"] == "me"
    info = sf.info(str(p.datasets_dir / "v1" / rows[0]["file_name"]))
    assert info.samplerate == SR and info.channels == 1

    # 参照音声: S/N の良い順に、合計秒数に達するまで選ぶ
    one = pick_references(p, "v1", total_sec=0.1, min_sec=0.5)
    assert len(one) == 1
    many = pick_references(p, "v1", total_sec=30.0, min_sec=0.5)
    assert len(many) == 3 and len(set(many)) == 3
    cfg = p.root / "configs" / "project.toml"
    cfg.write_text(cfg.read_text(encoding="utf-8").replace('ref_exclude = [', 'ref_exclude = ["EMOTION100_001_normal", '), encoding="utf-8")
    assert all(r.stem != "EMOTION100_001_normal" for r in pick_references(Project.load(), "v1", total_sec=30.0, min_sec=0.5))
    assert pick_reference(p, "v1", min_sec=0.5) == pick_references(p, "v1", total_sec=0.1, min_sec=0.5)[0]

    # reject.txt で除外
    p.reject_file.write_text("EMOTION100_002_normal  # 噛んだ\n", encoding="utf-8")
    assert build(p, "v2", phases=["phase1"])["utterances"] == 2


def test_split_mismatch_writes_review(project_root):
    p = Project.load()
    p.recordings_dir.mkdir()
    _m4a(p.recordings_dir / "emotion-normal-s01.m4a", recording([1.0, 2.2]))  # 1 文足りない
    with pytest.raises(SplitMismatch):
        split_recording(p, p.recordings_dir / "emotion-normal-s01.m4a", use_asr="never")
    assert (p.review_dir / "emotion-normal-s01" / "chunk_001.wav").is_file()
    assert load_segments(p, "emotion-normal-s01") == []


def test_split_partial_with_append(project_root):
    p = Project.load()
    p.recordings_dir.mkdir()
    _m4a(p.recordings_dir / "a.m4a", recording([1.0, 2.2]))
    _m4a(p.recordings_dir / "b.m4a", recording([2.0]))
    split_recording(p, p.recordings_dir / "a.m4a", sid="emotion-normal-s01", count=2, use_asr="never")
    split_recording(p, p.recordings_dir / "b.m4a", sid="emotion-normal-s01", start=3, use_asr="never", append=True)
    assert [s.corpus_id for s in load_segments(p, "emotion-normal-s01")] == ["EMOTION100_001", "EMOTION100_002", "EMOTION100_003"]


# ---- irodori wrapper ------------------------------------------------------------
def test_irodori_commands_dry_run(project_root, monkeypatch, capsys):
    from myvoice import irodori

    fake = project_root / "third_party" / "Irodori-TTS"
    fake.mkdir(parents=True)
    (fake / "train.py").write_text("")
    p = Project.load()
    ds = p.datasets_dir / "v1"
    ds.mkdir(parents=True)
    (ds / "metadata.csv").write_text("file_name,text,caption,speaker\n", encoding="utf-8")

    irodori.prepare(p, "v1", device="cuda", dry_run=True)
    irodori.train(p, "v1", "exp1", dry_run=True)
    out = capsys.readouterr().out
    assert "prepare_manifest.py" in out and "--speaker-id-prefix myvoice" in out
    assert "--max-steps 1500" in out and "--batch-size 16" in out and "--init-checkpoint" in out
    assert "--checkpoint-best-n 0" in out and "--valid-ratio 0.1" in out

    run = irodori.run_dir(p, "exp1")
    for name in ["checkpoint_0001000", "checkpoint_best_val_loss_0000500_1.234567", "checkpoint_final", "checkpoint_0000250"]:
        (run / name).mkdir(parents=True)
    names = [c.name for c in irodori.checkpoints(p, "exp1")]
    assert names == ["checkpoint_0000250", "checkpoint_best_val_loss_0000500_1.234567", "checkpoint_0001000", "checkpoint_final"]
    assert irodori.resolve_adapter(p, "exp1").name == "checkpoint_final"
    assert irodori.resolve_adapter(p, "exp1/checkpoint_0001000").name == "checkpoint_0001000"

    irodori.infer(p, "テスト", run / "o.wav", adapter=run / "checkpoint_final", device="mps", dry_run=True)
    out = capsys.readouterr().out
    assert "--no-ref" in out and "--model-device mps" in out and "--lora-adapter" in out
    assert "--hf-checkpoint" in out  # models/ にベースモデルが無ければ HF から

    # 参照音声: 1 つなら --ref-wav、複数なら --ref-wavs。models/ にあればローカルの重みを使う
    weights = irodori.local_base_model(p)
    weights.parent.mkdir(parents=True)
    weights.write_bytes(b"")
    irodori.infer(p, "テスト", run / "o.wav", ref=[ds / "a.wav", ds / "b.wav"], dry_run=True)
    out = capsys.readouterr().out
    assert "--ref-wavs" in out and "a.wav" in out and "b.wav" in out and "--checkpoint" in out
    assert "--hf-checkpoint" not in out
    irodori.infer(p, "テスト", run / "o.wav", ref=ds / "a.wav", dry_run=True)
    assert "--ref-wav " in capsys.readouterr().out


def test_cli_passes_only_args_after_double_dash(project_root, monkeypatch):
    """`train run v1 --run x --dry-run -- ...` で --run / --dry-run を train.py に流さない。"""
    from myvoice import cli, irodori

    seen = {}
    monkeypatch.setattr(irodori, "train", lambda p, ds, run, resume=None, extra=None, dry_run=False: seen.update(ds=ds, run=run, extra=extra, dry_run=dry_run) or Path("."))
    assert cli.main(["train", "run", "v1", "--run", "x", "--dry-run", "--", "--lora-r", "32"]) == 0
    assert seen == {"ds": "v1", "run": "x", "extra": ["--lora-r", "32"], "dry_run": True}
    assert cli.main(["train", "run", "v1", "--run", "y"]) == 0
    assert seen["extra"] == [] and seen["dry_run"] is False
    with pytest.raises(SystemExit):
        cli.main(["status", "--", "--x"])
