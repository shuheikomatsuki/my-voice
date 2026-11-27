# my-voice

自分の声を録音して、日本語 TTS **[Irodori-TTS](https://github.com/Aratako/Irodori-TTS) v4.1-Small を LoRA 学習**し、
MacBook (MPS) で推論するためのモノレポ。

```
録音 (iPhone .m4a) → 分割・台本対応付け → 品質チェック → データセット → LoRA 学習 (GPU) → 推論 (Mac)
```

すべて **uv** で管理する。

- 本リポジトリのツール: `uv sync` → `uv run myvoice ...`
- Irodori-TTS: `third_party/Irodori-TTS` に clone し、**独立した uv 環境**（`uv run myvoice irodori setup`）。
  myvoice から `uv run --no-sync` で呼び出す。

> ⚠️ 録音・データセット・学習済み LoRA は `.gitignore` で除外している。声のデータはなりすましに悪用されうるので、コミットしないこと。

## ディレクトリ

```
configs/project.toml   設定（caption・録音フェーズ・QC しきい値・学習の上書き設定）
src/myvoice/           ツール本体
tests/                 テスト（uv run pytest）
recordings/            録音ファイル置き場            ※git 管理外
data/
  corpus/              ITA コーパス
  scripts/<phase>/     録音用台本
  processed/<session>/ 文ごとの WAV + segments.jsonl
  review/<session>/    分割に失敗した録音の区間（確認用）
  datasets/<name>/     学習用データセット（WAV + metadata.csv）
  reports/             品質チェックのレポート (TSV)
  reject.txt           手動で除外する utt_id
work/<dataset>/        Irodori の潜在表現・manifest
outputs/runs/<run>/    LoRA チェックポイント・聞き比べ用サンプル
third_party/Irodori-TTS
```

## セットアップ

必要なもの: [uv](https://docs.astral.sh/uv/)、ffmpeg、git。

```bash
# Mac
brew install uv ffmpeg
# Ubuntu (GPU サーバー)
sudo apt install ffmpeg && curl -LsSf https://astral.sh/uv/install.sh | sh
```

```bash
uv sync                  # 本体
uv sync --extra asr      # 任意: 文字起こし（言い直しの自動除去・台本ずれ検出）。強く推奨
uv run myvoice doctor    # 環境チェック
```

## 1. 録音

### 台本を作る

```bash
uv run myvoice script --phase phase1
```

`data/scripts/phase1/emotion-normal-s01.txt` 〜 `s04.txt`（25 文ずつ）ができる。
読むときは ふりがな付きの [ITA コーパス朗読者用 PDF](https://github.com/mmorise/ita-corpus) を併用（番号は同じ）。

| フェーズ | 内容 | 音声 | 録音時間の目安 |
|---|---|---|---|
| phase1 | Emotion 100 文を普通の話し方で | 約 10 分 | 30〜45 分 |
| phase2 | Recitation 324 文を普通の話し方で | 約 45 分 | 2〜3 時間 |
| phase3 | Emotion 100 文を 喜び/怒り/悲しみ で | 各 約 10 分 | 各 30〜45 分 |
| phase4 | 自作の会話調台本を普段しゃべるように | 50 文で 約 2 分 | 50 文で 約 15 分 |

phase4 の台本は ITA ではなく自分で書く: `corpus/casual_transcript_utf8.txt` に
ITA と同じ形式（`CASUAL_001:本文,カタカナ読み`、1 行 1 文）で置く（文字だけなのでリポジトリで管理する）。
朗読口調にならないよう、自分の言い回しで書き、1 文ずつ覚えて画面から目を離して話しかけるように言う。

### iPhone での録り方

- ボイスメモ、録音モード **モノラル**。.m4a のままで OK（自動で 48kHz WAV に変換する）
- iPhone を机（タオルの上）に置き、**下端（USB-C 側のマイク）を口に向けて 15〜25cm**、少し斜めから
- 文と文の間は **1.5〜2 秒** あける。間違えたら 3 秒黙って、その文を最初から読み直す
  （言い直しの除去には `uv sync --extra asr` が必要。無い場合はその録音ファイルを録り直す）
- 1 回 30〜45 分まで。距離・声の大きさ・時間帯をそろえる
- ファイル名をセッション ID にして `recordings/` に置く: `recordings/emotion-normal-s01.m4a`

### まずテスト録音をチェック

```bash
uv run myvoice check recordings/test.m4a
```

雑音レベル・S/N・音量・音割れ・サンプルレート/ビットレートを表示する。目安: **S/N 40dB 以上なら良好、25dB 未満なら部屋を見直す**。

## 2. 分割と品質チェック

```bash
uv run myvoice split recordings/emotion-normal-s01.m4a     # 複数ファイル可: recordings/emotion-normal-*.m4a
uv run myvoice status                                       # 進み具合
uv run myvoice qc                                           # まとめて検査 → data/reports/qc_*.tsv
uv run myvoice qc --asr                                     # 文字起こしで台本との一致も検査
```

- 区間数と文数が一致すれば順番どおりに対応付け。一致しなければ（言い直し・物音）、
  faster-whisper があれば文字起こしで自動対応付け、無ければ `data/review/<session>/` に区間を書き出して止まる。
  - 区間が多い → `--min-silence 1.3`、区間が少ない → `--min-silence 0.6`
  - 1 セッションを途中で分けて録った → `--start 13 --count 13 --append`
- チェック内容:
  - **壊れ**: デコード不可・空・NaN
  - **形式**: サンプルレート 44.1kHz 未満、ビットレートが低すぎる
  - **音質**: 音割れ・声が小さい・雑音 (S/N)・DC オフセット・前後の無音が長い
  - **台本とのずれ**: モーラ/秒が異常（読み飛ばし・分割ずれ）、`--asr` 時は文字起こしとの一致度
- エラーの音声はデータセットから自動で除外。気になる音声は `data/reject.txt` に utt_id を書いて除外
  （例: `EMOTION100_012_normal  # 噛んだ`）

## 3. データセット作成

```bash
uv run myvoice build v1 --phases phase1                # 動作確認用
uv run myvoice build v2 --phases phase1,phase2         # 本番（データが増えたら作り直して最初から学習）
uv run myvoice build v3 --phases phase1,phase2,phase3  # 感情込み
```

`data/datasets/<name>/metadata.csv`（`file_name,text,caption,speaker`）は Irodori-TTS の `prepare_manifest.py` がそのまま読める HF audiofolder 形式。
caption は `configs/project.toml` の `[styles]` から入る。**自分の声の特徴を書き足すと良い。**

GPU サーバーへ転送:

```bash
rsync -av data/datasets/v2 user@gpu-server:~/my-voice/data/datasets/
```

## 4. 学習（GPU サーバー, RTX PRO 4000 24GB）

```bash
git clone https://github.com/shuheikomatsuki/my-voice.git && cd my-voice
uv sync
uv run myvoice irodori setup --backend cu128   # Blackwell は CUDA 12.8 版が必要
uv run myvoice irodori download                 # ベースモデル (models/, 3GB) と codec (HF キャッシュ)
uv run myvoice doctor                           # torch / cuda / GPU 名を確認

uv run myvoice train prepare v2                 # 潜在表現 + manifest（work/v2/）
uv run myvoice train run v2 --run v2-r16        # LoRA 学習 → outputs/runs/v2-r16/
```

- 公式 LoRA 設定（`train_v4_small_lora.yaml`）は大規模データ向けなので、少量データ向けの値を
  `configs/project.toml` の `[irodori.train_overrides]` で上書きしている（v2 の 424 文向けに batch 16、3000 steps ≒ 125 epoch、500 steps ごとに保存）。
  データを増やしたら steps/epoch が変わるので max_steps を見直す
- `checkpoint_best_n = 0` にしている。>0 だと検証 loss 上位以外の途中チェックポイントが消え、聞き比べられなくなる
- 検証 loss（`valid_ratio = 0.1`）は揺れが大きいので発散の確認用。チェックポイントは聞き比べで選ぶ
- `--` 以降は `train.py` にそのまま渡す: `uv run myvoice train run v2 --run v2-r32 -- --lora-r 32 --lora-alpha 64`
- VRAM 不足: `-- --gradient-checkpointing` または `batch_size = 8`
- 再開: `uv run myvoice train run v2 --run v2-r16 --resume outputs/runs/v2-r16/checkpoint_0001000`
- `--dry-run` で実行コマンドだけ表示

### チェックポイントの聞き比べ

```bash
uv run myvoice train samples v2-r16 --with-base
```

各チェックポイントでテスト文（`[infer] test_sentences`）を合成し、`outputs/runs/v2-r16/samples/<checkpoint>/` に保存（生成音声も自動で壊れチェック）。
参照音声はデータセットから S/N の良い 2〜10 秒の文を、合計約 30 秒（`[infer] ref_total_sec`、`--ref-seconds`）になるまで複数自動選択して `--ref-wavs` で渡す（`--ref auto`）。
学習時も同じ話者の別の文をつなげた長い参照を使っているため。`--ref a.wav b.wav` で直接指定もできる。似ていて、かつ発音が崩れていないものを選ぶ。
ステップが進みすぎると過学習で不自然になりやすい。

## 5. 推論（MacBook Pro M2）

```bash
uv sync
uv run myvoice irodori setup          # Mac は cpu extra（PyPI の torch で MPS が使える）
rsync -av user@gpu-server:~/my-voice/outputs/runs/v3-r16/checkpoint_0005000 outputs/runs/v3-r16/
rsync -av user@gpu-server:~/my-voice/data/datasets/v3 data/datasets/   # 参照音声用（任意）
```

```bash
uv run myvoice infer --adapter v3-r16/checkpoint_0005000 --ref auto --dataset v3 --style normal \
  --text "これは自分の声で合成したテストです。" --out outputs/test.wav

uv run myvoice infer --adapter v3-r16/checkpoint_0005000 --ref auto --dataset v3 --style casual --text-file lines.txt --out outputs/batch/
uv run myvoice infer --adapter v3-r16/checkpoint_0005000 --text "参照音声なし、caption のみ" --style casual
```

- 標準のモデルは **v3-r16 の 5000 steps**（朗読 424 文 + 会話調 500 文。2026-11-27 に聞き比べて決定。2500〜5000 steps で耳での差はほぼない）。
  話し方は `--style` の caption で切り替える: `normal`＝落ち着いた普通の調子、`casual`＝くだけた会話調。
  会話調の速さは caption でほぼ決まり、参照音声が朗読でも本物に近い速さになる

- Mac は既定で `--device mps`、**fp32**（MPS で bf16 は遅くなる報告あり）
- 複数の文（`--text-file` や `train samples`）は、モデルを 1 回だけ読み込んで続けて合成する（`irodori.infer_batch`）。
  RTX PRO 4000 で、読み込み 約 20 秒 + 1 文 約 1.5 秒（1 文ずつ読み込み直すと 1 文 約 25 秒）。同じ seed なら 1 文ずつの合成と同じ音声になる
- `--text-file` の行を `名前<TAB>文` にすると `<名前>.wav` で保存（名前なしは 001.wav からの連番）
- 感情を変える: `--style happy`（caption）と、その感情で読んだ参照音声 `--ref` を組み合わせると効きやすい

## 開発

```bash
uv sync
uv run pytest
```

テストは合成音声で 分割・対応付け（言い直し除去・文の結合）・品質チェック・データセット作成・Irodori 呼び出しコマンドを検証する（ffmpeg が必要）。
