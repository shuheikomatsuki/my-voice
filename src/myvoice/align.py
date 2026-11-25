"""切り出した区間と台本の対応付け。

- 区間数と文数が一致すれば順番どおりに対応付ける（ASR 不要）。
- 一致しない場合（言い直し・物音・文中の長い間）は、文字起こし結果と台本を
  動的計画法で単調に対応付ける。言い直しは後ろのテイクを優先して採用し、
  1 文が 2〜3 区間に割れた場合は結合する。
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from difflib import SequenceMatcher

_KEEP = re.compile(r"[\w぀-ヿ一-鿿]")


def normalize(s: str) -> str:
    s = unicodedata.normalize("NFKC", s)
    out = []
    for ch in s:
        if not _KEEP.match(ch) or ch == "_":
            continue
        # カタカナ → ひらがな
        if "ァ" <= ch <= "ヶ":
            ch = chr(ord(ch) - 0x60)
        out.append(ch)
    return "".join(out)


def similarity(hyp: str, text: str, reading: str = "") -> float:
    """文字起こし (hyp) と台本の一致度 0〜1。表記ゆれ対策で本文と読みの高い方を採る。"""
    h = normalize(hyp)
    if not h:
        return 0.0
    best = 0.0
    for ref in (text, reading):
        r = normalize(ref)
        if r:
            best = max(best, SequenceMatcher(None, h, r, autojunk=False).ratio())
    return best


@dataclass
class Assignment:
    sentence_index: int
    chunk_indices: list[int]
    score: float


def align(
    chunk_texts: list[str],
    sentences: list[tuple[str, str]],
    max_merge: int = 3,
    later_take_bonus: float = 1e-3,
) -> list[Assignment]:
    """chunk_texts[j] を sentences[i]=(text, reading) に単調に割り当てる。

    各文はちょうど 1 グループ（連続する 1〜max_merge 区間）に割り当てられ、
    残りの区間は捨てられる（言い直し前のテイク・雑音）。
    """
    n, m = len(sentences), len(chunk_texts)
    if n == 0:
        return []
    if m < n:
        raise ValueError(f"区間数 ({m}) が文数 ({n}) より少ないため対応付けできません")
    NEG = float("-inf")
    # best[i][j]: 先頭 i 文を先頭 j 区間で処理したときの最大スコア
    best = [[NEG] * (m + 1) for _ in range(n + 1)]
    back: list[list[tuple[int, int] | None]] = [[None] * (m + 1) for _ in range(n + 1)]
    best[0][0] = 0.0
    sim_cache: dict[tuple[int, int, int], float] = {}

    def sim(i: int, j: int, k: int) -> float:
        key = (i, j, k)
        if key not in sim_cache:
            hyp = "".join(chunk_texts[j : j + k])
            sim_cache[key] = similarity(hyp, *sentences[i])
        return sim_cache[key]

    for i in range(n + 1):
        for j in range(m + 1):
            cur = best[i][j]
            if cur == NEG:
                continue
            # 区間 j を捨てる
            if j < m and cur > best[i][j + 1]:
                best[i][j + 1] = cur
                back[i][j + 1] = (i, j)
            # 区間 j..j+k-1 を文 i に割り当てる
            if i < n:
                for k in range(1, max_merge + 1):
                    if j + k > m:
                        break
                    sc = cur + sim(i, j, k) + later_take_bonus * (j / max(1, m))
                    if sc > best[i + 1][j + k]:
                        best[i + 1][j + k] = sc
                        back[i + 1][j + k] = (i, j)

    # 復元
    out: list[Assignment] = []
    i, j = n, m
    while (i, j) != (0, 0):
        prev = back[i][j]
        assert prev is not None
        pi, pj = prev
        if pi == i - 1:
            k = j - pj
            out.append(Assignment(pi, list(range(pj, j)), sim(pi, pj, k)))
        i, j = pi, pj
    out.reverse()
    return out
