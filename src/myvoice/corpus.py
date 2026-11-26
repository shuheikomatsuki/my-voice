"""ITA コーパス（パブリックドメイン）の取得と、録音セッション台本の生成。

ITA 以外に、自分で書いた台本も使える: data/corpus/<corpus>_transcript_utf8.txt に
ITA と同じ形式（1 行 1 文、`ID:本文,カタカナ読み`、ID は CASUAL_001 のように「_番号」で終わる）で置く。

録音は「セッション」単位で行う。セッション ID は
    <corpus>-<style>-sNN       例: emotion-normal-s01
で、録音ファイル名はこの ID にする（例: recordings/emotion-normal-s01.m4a）。
"""

from __future__ import annotations

import json
import re
import unicodedata
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path

from .project import Project

ITA_BASE = "https://raw.githubusercontent.com/mmorise/ita-corpus/main"
ITA_FILES = {
    "emotion": "emotion_transcript_utf8.txt",
    "recitation": "recitation_transcript_utf8.txt",
}
SESSION_RE = re.compile(r"^(?P<corpus>[a-z0-9]+)-(?P<style>[a-z0-9]+)-s(?P<num>\d{2,})")


@dataclass
class Sentence:
    corpus_id: str  # 例: EMOTION100_001
    text: str
    reading: str  # カタカナ読み

    @property
    def number(self) -> int:
        return int(self.corpus_id.rsplit("_", 1)[1])


@dataclass
class Utterance:
    utt_id: str  # 例: EMOTION100_001_normal
    corpus: str
    style: str
    corpus_id: str
    text: str
    reading: str


def parse_ita(text: str) -> list[Sentence]:
    out = []
    for line in text.splitlines():
        line = line.strip().lstrip("﻿")
        if not line:
            continue
        cid, rest = line.split(":", 1)
        # 本文中に「,」は無く、最後の「,」以降が読み
        body, reading = rest.rsplit(",", 1)
        out.append(Sentence(cid.strip(), body.strip(), reading.strip()))
    return out


def fetch(project: Project, force: bool = False) -> dict[str, Path]:
    project.corpus_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    for name, fname in ITA_FILES.items():
        dst = project.corpus_dir / fname
        if force or not dst.is_file():
            with urllib.request.urlopen(f"{ITA_BASE}/{fname}", timeout=60) as r:
                dst.write_bytes(r.read())
        paths[name] = dst
    return paths


def corpus_path(project: Project, corpus: str) -> Path:
    return project.corpus_dir / ITA_FILES.get(corpus, f"{corpus}_transcript_utf8.txt")


def load_corpus(project: Project, corpus: str) -> list[Sentence]:
    path = corpus_path(project, corpus)
    if not path.is_file():
        if corpus not in ITA_FILES:
            raise FileNotFoundError(f"コーパス '{corpus}' の台本がありません: {path}")
        fetch(project)
    return parse_ita(path.read_text(encoding="utf-8"))


def session_id(corpus: str, style: str, num: int) -> str:
    return f"{corpus}-{style}-s{num:02d}"


def parse_session_id(name: str) -> tuple[str, str, int] | None:
    m = SESSION_RE.match(name)
    if not m:
        return None
    return m["corpus"], m["style"], int(m["num"])


def build_sessions(project: Project, corpus: str, style: str) -> dict[str, list[Utterance]]:
    per = int(project.section("recording").get("sentences_per_session", 25))
    sents = load_corpus(project, corpus)
    sessions: dict[str, list[Utterance]] = {}
    for i in range(0, len(sents), per):
        sid = session_id(corpus, style, i // per + 1)
        sessions[sid] = [
            Utterance(f"{s.corpus_id}_{style}", corpus, style, s.corpus_id, s.text, s.reading) for s in sents[i : i + per]
        ]
    return sessions


def all_sessions(project: Project) -> dict[str, list[Utterance]]:
    out: dict[str, list[Utterance]] = {}
    seen = set()
    for ph in project.phases:
        for style in ph["styles"]:
            key = (ph["corpus"], style)
            if key in seen:
                continue
            seen.add(key)
            out.update(build_sessions(project, ph["corpus"], style))
    return out


def sessions_for_phase(project: Project, phase: str) -> dict[str, list[Utterance]]:
    ph = project.phase(phase)
    out: dict[str, list[Utterance]] = {}
    for style in ph["styles"]:
        out.update(build_sessions(project, ph["corpus"], style))
    return out


def get_session(project: Project, sid: str) -> list[Utterance]:
    parsed = parse_session_id(sid)
    if not parsed:
        raise ValueError(f"セッション ID の形式が不正: {sid}（例: emotion-normal-s01）")
    corpus, style, _ = parsed
    if style not in project.styles:
        raise KeyError(f"style '{style}' が configs/project.toml の [styles] にありません")
    sessions = build_sessions(project, corpus, style)
    if sid not in sessions:
        raise KeyError(f"セッション {sid} は存在しません（{corpus} は {len(sessions)} セッション）")
    return sessions[sid]


def write_scripts(project: Project, phase: str) -> list[Path]:
    """録音用の台本を data/scripts/<phase>/ に書き出す。"""
    sessions = sessions_for_phase(project, phase)
    outdir = project.scripts_dir / phase
    outdir.mkdir(parents=True, exist_ok=True)
    paths = []
    for sid, utts in sessions.items():
        style = utts[0].style
        lines = [
            f"# {sid}  （録音ファイル名: {sid}.m4a）",
            f"# 話し方: {style} — {project.caption_for(style)}",
            f"# {len(utts)} 文。1 文ごとに 1.5〜2 秒あけて読む。間違えたら 3 秒黙ってその文を読み直す。",
        ]
        if utts[0].corpus in ITA_FILES:
            lines.append("# 番号は ITA コーパス朗読者用 PDF の番号と同じ（ふりがなは PDF を参照）。")
        lines.append("")
        for u in utts:
            lines.append(f"{Sentence(u.corpus_id, u.text, u.reading).number:>3}. {u.text}")
            lines.append(f"     （{u.reading}）")
        p = outdir / f"{sid}.txt"
        p.write_text("\n".join(lines) + "\n", encoding="utf-8")
        paths.append(p)
    (outdir / "sessions.json").write_text(
        json.dumps({k: [asdict(u) for u in v] for k, v in sessions.items()}, ensure_ascii=False, indent=1),
        encoding="utf-8",
    )
    return paths


# ---- 読み（カタカナ）からモーラ数を数える ------------------------------------------
_SMALL = set("ァィゥェォャュョヮヵヶぁぃぅぇぉゃゅょゎ")


def count_morae(reading: str) -> int:
    n = 0
    for ch in unicodedata.normalize("NFKC", reading):
        if ch in _SMALL:
            continue  # 拗音の小書きは直前と合わせて 1 モーラ
        if "゠" <= ch <= "ヿ" or "぀" <= ch <= "ゟ":
            if ch in "・゠":
                continue
            n += 1  # ッ・ン・ー も 1 モーラ
    return n
