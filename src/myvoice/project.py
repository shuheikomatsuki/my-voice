"""リポジトリのルート・ディレクトリ構成・設定ファイルの読み込み。"""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Any

CONFIG_REL = Path("configs/project.toml")


def find_root(start: Path | None = None) -> Path:
    env = os.environ.get("MYVOICE_ROOT")
    if env:
        return Path(env).resolve()
    cur = (start or Path.cwd()).resolve()
    for p in [cur, *cur.parents]:
        if (p / CONFIG_REL).is_file():
            return p
    raise FileNotFoundError(
        f"{CONFIG_REL} が見つかりません。my-voice リポジトリ内で実行するか MYVOICE_ROOT を設定してください。"
    )


@dataclass
class Project:
    root: Path

    @classmethod
    def load(cls, start: Path | None = None) -> "Project":
        return cls(find_root(start))

    # ---- 設定 ---------------------------------------------------------------
    @cached_property
    def config(self) -> dict[str, Any]:
        with open(self.root / CONFIG_REL, "rb") as f:
            return tomllib.load(f)

    def section(self, name: str) -> dict[str, Any]:
        return dict(self.config.get(name, {}))

    @property
    def speaker_id(self) -> str:
        return self.config.get("speaker", {}).get("id", "me")

    @property
    def styles(self) -> dict[str, dict[str, Any]]:
        return self.config.get("styles", {})

    def caption_for(self, style: str) -> str:
        try:
            return self.styles[style]["caption"]
        except KeyError:
            raise KeyError(f"style '{style}' が configs/project.toml の [styles] にありません") from None

    @property
    def phases(self) -> list[dict[str, Any]]:
        return list(self.config.get("phases", []))

    def phase(self, name: str) -> dict[str, Any]:
        for p in self.phases:
            if p["name"] == name:
                return p
        raise KeyError(f"phase '{name}' が configs/project.toml にありません")

    # ---- ディレクトリ ---------------------------------------------------------
    @property
    def recordings_dir(self) -> Path:
        return self.root / "recordings"

    @property
    def data_dir(self) -> Path:
        return self.root / "data"

    @property
    def corpus_dir(self) -> Path:
        return self.data_dir / "corpus"

    @property
    def own_corpus_dir(self) -> Path:
        """自分で書いた台本（ITA 以外）。声のデータではないのでリポジトリで管理する。"""
        return self.root / "corpus"

    @property
    def scripts_dir(self) -> Path:
        return self.data_dir / "scripts"

    @property
    def processed_dir(self) -> Path:
        return self.data_dir / "processed"

    @property
    def review_dir(self) -> Path:
        return self.data_dir / "review"

    @property
    def datasets_dir(self) -> Path:
        return self.data_dir / "datasets"

    @property
    def reports_dir(self) -> Path:
        return self.data_dir / "reports"

    @property
    def reject_file(self) -> Path:
        """手動で除外したい utt_id を 1 行 1 つ書くファイル（# 以降はコメント）。"""
        return self.data_dir / "reject.txt"

    @property
    def work_dir(self) -> Path:
        return self.root / "work"

    @property
    def outputs_dir(self) -> Path:
        return self.root / "outputs"

    @property
    def models_dir(self) -> Path:
        return self.root / "models"

    @property
    def irodori_dir(self) -> Path:
        return Path(os.environ.get("IRODORI_DIR", self.root / "third_party" / "Irodori-TTS"))

    def rejected_ids(self) -> set[str]:
        if not self.reject_file.is_file():
            return set()
        ids = set()
        for line in self.reject_file.read_text(encoding="utf-8").splitlines():
            line = line.split("#", 1)[0].strip()
            if line:
                ids.add(line)
        return ids
