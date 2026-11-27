"""Irodori-TTS の infer.py を、モデルを 1 回だけ読み込んで複数の文に使い回して実行する。

Irodori-TTS のディレクトリを cwd にして、その uv 環境の python で実行する（myvoice.irodori.infer_batch から呼ぶ）。
    python batch_infer.py --jobs jobs.jsonl <infer.py の共通引数...>
jobs.jsonl は 1 行 1 文で {"args": ["--text", "...", "--output-wav", "...", ...]}。
各行の args を共通引数の後ろに付けて infer.main() を呼ぶので、引数の解釈・合成の処理は infer.py と同じ。
InferenceRuntime.from_key を Irodori 自身のキャッシュ（get_cached_runtime）に差し替えて、
同じ設定ならモデル・コーデックを読み込み直さないようにする。LoRA もパスごとにキャッシュされる。

--peak-dbfs X を付けると、保存する前にピークが X dBFS を超える音声だけ全体の音量を下げる
（Irodori は生成した波形をそのまま 16bit で保存するので、1.0 を超えた部分が音割れになる）。
"""

from __future__ import annotations

import json
import math
import os
import sys
import time
import traceback

# このファイルのディレクトリではなく Irodori-TTS（cwd）から import する
sys.path.insert(0, os.getcwd())

import infer  # noqa: E402  (Irodori-TTS/infer.py)
from irodori_tts import inference_runtime as rt  # noqa: E402


def main() -> int:
    argv = sys.argv[1:]
    if "--jobs" not in argv:
        print("usage: batch_infer.py --jobs jobs.jsonl [infer.py args...]", file=sys.stderr)
        return 2
    i = argv.index("--jobs")
    jobs_path = argv[i + 1]
    common = argv[:i] + argv[i + 2 :]
    peak_dbfs = None
    if "--peak-dbfs" in common:
        k = common.index("--peak-dbfs")
        peak_dbfs = float(common[k + 1])
        del common[k : k + 2]
    with open(jobs_path, encoding="utf-8") as f:
        jobs = [json.loads(line)["args"] for line in f if line.strip()]

    # infer.py が参照する名前だけを差し替える（get_cached_runtime の中では本物の from_key が使われる）
    class _CachedRuntime:
        @staticmethod
        def from_key(key):
            return rt.get_cached_runtime(key)[0]

    infer.InferenceRuntime = _CachedRuntime

    if peak_dbfs is not None:
        limit = 10 ** (peak_dbfs / 20)
        save = infer.save_wav

        def save_limited(path, audio, sample_rate):
            peak = float(audio.abs().max())
            if peak > limit:
                audio = audio * (limit / peak)
                print(f"[batch] peak {20 * math.log10(peak):+.2f} dBFS -> {peak_dbfs:+.1f} dBFS", flush=True)
            return save(path, audio, sample_rate)

        infer.save_wav = save_limited

    failed = 0
    t_all = time.time()
    for n, job in enumerate(jobs, 1):
        t0 = time.time()
        sys.argv = ["infer.py", *common, *job]
        print(f"[batch] {n}/{len(jobs)}", flush=True)
        try:
            infer.main()
        except SystemExit as e:  # argparse のエラーなど
            if e.code not in (0, None):
                failed += 1
                print(f"[batch] FAILED {n}: exit {e.code}", flush=True)
        except Exception:
            failed += 1
            traceback.print_exc()
            print(f"[batch] FAILED {n}", flush=True)
        print(f"[batch] {n}/{len(jobs)} done in {time.time() - t0:.1f} s", flush=True)
    print(f"[batch] all {len(jobs)} jobs in {time.time() - t_all:.1f} s, failed={failed}", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
