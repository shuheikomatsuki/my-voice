"""Irodori-TTS（third_party/Irodori-TTS、独立した uv 環境）の呼び出し。

Irodori-TTS 側のコマンドは公式 README / docs/parameters.md のものに合わせている。
"""

from __future__ import annotations

import json
import os
import platform
import re
import shlex
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

from .project import Project

BACKENDS = ("cu128", "cpu", "rocm", "xpu")


class IrodoriError(RuntimeError):
    pass


def _uv() -> str:
    uv = shutil.which("uv")
    if not uv:
        raise IrodoriError("uv が見つかりません: https://docs.astral.sh/uv/getting-started/installation/")
    return uv


def default_backend() -> str:
    if platform.system() == "Darwin":
        return "cpu"  # PyPI の torch。macOS では MPS も使える
    if shutil.which("nvidia-smi"):
        return "cu128"
    return "cpu"


def default_device() -> str:
    if platform.system() == "Darwin":
        return "mps"
    if shutil.which("nvidia-smi"):
        return "cuda"
    return "cpu"


def ensure_installed(project: Project) -> Path:
    d = project.irodori_dir
    if not (d / "train.py").is_file():
        raise IrodoriError(f"Irodori-TTS がありません: {d}\n  → uv run myvoice irodori setup")
    return d


def run_env_overrides(project: Project) -> dict[str, str]:
    """[irodori.env] の環境変数。シェルで同じ変数を設定していればそちらを優先する。"""
    return {k: str(v) for k, v in project.section("irodori").get("env", {}).items() if k not in os.environ}


def run(project: Project, args: list[str], dry_run: bool = False) -> None:
    """Irodori-TTS の uv 環境で python スクリプトを実行する。"""
    d = ensure_installed(project)
    cmd = [_uv(), "run", "--no-sync", "python", *args]
    extra_env = run_env_overrides(project)
    env_str = "".join(f"{k}={shlex.quote(v)} " for k, v in extra_env.items())
    print("$ (cd", shlex.quote(str(d)), "&&", env_str + " ".join(shlex.quote(c) for c in cmd[1:]) + ")", flush=True)
    if dry_run:
        return
    env = dict(os.environ)
    env.pop("VIRTUAL_ENV", None)  # my-voice 側の venv を引き継がない
    env.update(extra_env)
    proc = subprocess.run(cmd, cwd=d, env=env)
    if proc.returncode != 0:
        raise IrodoriError(f"Irodori-TTS のコマンドが失敗しました (exit {proc.returncode})")


# ---- セットアップ ---------------------------------------------------------------
def setup(project: Project, backend: str | None = None, update: bool = False) -> None:
    cfg = project.section("irodori")
    backend = backend or default_backend()
    if backend not in BACKENDS:
        raise IrodoriError(f"backend は {BACKENDS} のいずれか")
    d = project.irodori_dir
    if not (d / ".git").is_dir():
        d.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", cfg.get("repo", "https://github.com/Aratako/Irodori-TTS.git"), str(d)], check=True)
    elif update:
        subprocess.run(["git", "-C", str(d), "pull", "--ff-only"], check=True)
    ref = cfg.get("ref") or ""
    if ref:
        subprocess.run(["git", "-C", str(d), "fetch", "--depth", "50", "origin", ref], check=False)
        subprocess.run(["git", "-C", str(d), "checkout", ref], check=True)
    env = dict(os.environ)
    env.pop("VIRTUAL_ENV", None)
    print(f"$ uv sync --extra {backend}  (in {d})")
    subprocess.run([_uv(), "sync", "--extra", backend], cwd=d, env=env, check=True)
    commit = subprocess.run(["git", "-C", str(d), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    print(f"Irodori-TTS {commit[:10]} を {backend} でセットアップしました")
    if not ref:
        print(f"  再現性のため configs/project.toml の [irodori] ref = \"{commit}\" に固定するのがおすすめです")


def torch_info(project: Project) -> str:
    d = ensure_installed(project)
    code = (
        "import torch,sys;"
        "print('torch', torch.__version__, '| cuda', torch.cuda.is_available(),"
        " torch.cuda.get_device_name(0) if torch.cuda.is_available() else '',"
        " '| mps', getattr(torch.backends,'mps',None) is not None and torch.backends.mps.is_available())"
    )
    env = dict(os.environ)
    env.pop("VIRTUAL_ENV", None)
    p = subprocess.run([_uv(), "run", "--no-sync", "python", "-c", code], cwd=d, env=env, capture_output=True, text=True)
    return (p.stdout or p.stderr).strip()


def base_model_id(project: Project) -> str:
    return project.section("irodori").get("base_model", "Aratako/Irodori-TTS-v4.1-Small")


def local_base_model(project: Project) -> Path:
    return project.models_dir / base_model_id(project).split("/")[-1] / "model.safetensors"


def download_base_model(project: Project) -> Path:
    repo_id = base_model_id(project)
    weights = local_base_model(project)
    local = weights.parent
    if not weights.is_file():
        code = (
            "import sys; from huggingface_hub import snapshot_download;"
            "snapshot_download(repo_id=sys.argv[1], local_dir=sys.argv[2])"
        )
        run(project, ["-c", code, repo_id, str(local)])
    if not weights.is_file():
        raise IrodoriError(f"model.safetensors が見つかりません: {local}")
    return weights


CODEC_REPO = "Aratako/Semantic-DACVAE-Japanese-32dim"  # prepare_manifest.py / v4 checkpoint の既定 codec


def download_codec(project: Project) -> None:
    """DACVAE codec を HF キャッシュに取得（prepare / infer が hf_hub_download で参照する）。"""
    code = "import sys; from huggingface_hub import hf_hub_download; print(hf_hub_download(repo_id=sys.argv[1], filename='weights.pth'))"
    run(project, ["-c", code, CODEC_REPO])


# ---- 学習 ----------------------------------------------------------------------
def work_dir(project: Project, dataset: str) -> Path:
    return project.work_dir / dataset


def run_dir(project: Project, run_name: str) -> Path:
    return project.outputs_dir / "runs" / run_name


def prepare(project: Project, dataset: str, device: str | None = None, dry_run: bool = False) -> Path:
    ds = project.datasets_dir / dataset
    if not (ds / "metadata.csv").is_file():
        raise IrodoriError(f"データセットがありません: {ds}（uv run myvoice build {dataset} ...）")
    w = work_dir(project, dataset)
    if w.exists() and not dry_run:
        shutil.rmtree(w)
    w.mkdir(parents=True, exist_ok=True)
    manifest = w / "train_manifest.jsonl"
    run(
        project,
        [
            "prepare_manifest.py",
            "--dataset", str(ds.resolve()),
            "--split", "train",
            "--audio-column", "audio",
            "--text-column", "text",
            "--caption-column", "caption",
            "--speaker-column", "speaker",
            "--speaker-id-prefix", "myvoice",
            "--output-manifest", str(manifest.resolve()),
            "--latent-dir", str((w / "latents").resolve()),
            # HF datasets のキャッシュも work/ に置く（~/.cache に声のデータの情報を残さない）
            "--cache-dir", str((w / "hf_cache").resolve()),
            "--device", device or default_device(),
        ],
        dry_run=dry_run,
    )
    return manifest


def train(
    project: Project,
    dataset: str,
    run_name: str,
    resume: str | None = None,
    extra: list[str] | None = None,
    dry_run: bool = False,
) -> Path:
    cfg = project.section("irodori")
    manifest = work_dir(project, dataset) / "train_manifest.jsonl"
    if not manifest.is_file() and not dry_run:
        raise IrodoriError(f"manifest がありません: {manifest}（uv run myvoice train prepare {dataset}）")
    out = run_dir(project, run_name)
    if not dry_run:
        out.mkdir(parents=True, exist_ok=True)
    args = [
        "train.py",
        "--config", cfg.get("lora_config", "configs/train_v4_small_lora.yaml"),
        "--manifest", str(manifest.resolve()),
        "--output-dir", str(out.resolve()),
        "--valid-ratio", str(cfg.get("valid_ratio", 0.05)),
    ]
    # 少量データ向けの上書き（公式 LoRA 設定は大規模データ向け: batch 40×2, 30000 steps）
    for k, v in project.section("irodori").get("train_overrides", {}).items():
        args += [f"--{k.replace('_', '-')}", str(v)]
    if resume:
        args += ["--resume", str(Path(resume).resolve())]
    else:
        weights = download_base_model(project) if not dry_run else local_base_model(project)
        args += ["--init-checkpoint", str(weights.resolve())]
    args += extra or []
    if not dry_run:
        (out / "myvoice_run.txt").write_text(f"dataset={dataset}\nargs={' '.join(args)}\n", encoding="utf-8")
    run(project, args, dry_run=dry_run)
    return out


def _step(p: Path) -> int:
    if p.name.startswith("checkpoint_final"):
        return 1 << 60
    m = re.match(r"checkpoint_(?:best_val_loss_)?(\d+)", p.name)
    return int(m.group(1)) if m else -1


def checkpoints(project: Project, run_name: str) -> list[Path]:
    d = run_dir(project, run_name)
    if not d.is_dir():
        return []
    return sorted((p for p in d.iterdir() if p.is_dir() and p.name.startswith("checkpoint")), key=_step)


def resolve_adapter(project: Project, adapter: str) -> Path:
    """パス、または run 名（→最新/最終チェックポイント）、run名/チェックポイント名 を受け付ける。"""
    p = Path(adapter).expanduser()
    if p.is_dir():
        return p
    if "/" in adapter:
        run_name, ck = adapter.split("/", 1)
        p = run_dir(project, run_name) / ck
        if p.is_dir():
            return p
    cks = checkpoints(project, adapter)
    if cks:
        return cks[-1]
    raise IrodoriError(f"LoRA アダプタが見つかりません: {adapter}")


# ---- 推論 ----------------------------------------------------------------------
def _infer_common_args(
    project: Project, ref: Path | list[Path] | None, device: str, precision: str
) -> list[str]:
    # 学習用に models/ へ落としたベースモデルがあればそれを使う（HF キャッシュへの二重ダウンロードを避ける）
    local = local_base_model(project)
    args = ["--checkpoint", str(local.resolve())] if local.is_file() else ["--hf-checkpoint", base_model_id(project)]
    refs = [ref] if isinstance(ref, Path) else list(ref or [])
    if len(refs) == 1:
        args += ["--ref-wav", str(refs[0].resolve())]
    elif refs:
        # 各クリップを個別にエンコードして順に連結（上限は checkpoint の ref_max_seconds）
        args += ["--ref-wavs", *[str(r.resolve()) for r in refs]]
    else:
        args += ["--no-ref"]
    args += ["--model-device", device, "--codec-device", device]
    args += ["--model-precision", precision, "--codec-precision", precision]
    return args


def _infer_job_args(text: str, output: Path, adapter: Path | None, caption: str | None) -> list[str]:
    args = ["--text", text, "--output-wav", str(output.resolve())]
    if adapter:
        args += ["--lora-adapter", str(adapter.resolve())]
    if caption:
        args += ["--caption", caption]
    return args


def infer(
    project: Project,
    text: str,
    output: Path,
    adapter: Path | None = None,
    ref: Path | list[Path] | None = None,
    caption: str | None = None,
    device: str | None = None,
    precision: str = "fp32",
    extra: list[str] | None = None,
    dry_run: bool = False,
) -> Path:
    return infer_batch(
        project, [InferJob(text, output, adapter, caption)], ref=ref, device=device, precision=precision, extra=extra, dry_run=dry_run
    )[0]


@dataclass
class InferJob:
    text: str
    output: Path
    adapter: Path | None = None  # 文ごとに変えてよい（LoRA はパスごとにキャッシュされる）
    caption: str | None = None


BATCH_SCRIPT = Path(__file__).parent / "irodori_scripts" / "batch_infer.py"


def infer_batch(
    project: Project,
    jobs: list[InferJob],
    ref: Path | list[Path] | None = None,
    device: str | None = None,
    precision: str = "fp32",
    extra: list[str] | None = None,
    dry_run: bool = False,
) -> list[Path]:
    """複数の文を、モデルを 1 回だけ読み込んで合成する（中身は infer.py と同じ処理。infer() もこれを使う）。

    1 文ずつ infer() を呼ぶと毎回 約 20 秒かけてモデルを読み込み直すが、こちらは最初の 1 回だけ。
    参照音声・device・precision・extra は全文で共通。
    """
    device = device or default_device()
    for j in jobs:
        j.output.parent.mkdir(parents=True, exist_ok=True)
    tmp = project.work_dir / "tmp"
    tmp.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", suffix=".jsonl", dir=tmp, delete=False) as f:
        for j in jobs:
            f.write(json.dumps({"args": _infer_job_args(j.text, j.output, j.adapter, j.caption)}, ensure_ascii=False) + "\n")
        jobs_path = Path(f.name)
    try:
        args = [str(BATCH_SCRIPT.resolve()), "--jobs", str(jobs_path), *_infer_common_args(project, ref, device, precision)]
        peak = project.section("infer").get("peak_dbfs")
        if peak is not None:
            args += ["--peak-dbfs", str(peak)]
        if dry_run:
            for j in jobs:
                print("  job:", " ".join(shlex.quote(c) for c in _infer_job_args(j.text, j.output, j.adapter, j.caption)))
        run(project, args + (extra or []), dry_run=dry_run)
    finally:
        jobs_path.unlink(missing_ok=True)
    return [j.output for j in jobs]
