# 引き継ぎメモ（クラウド → ローカルの Claude Code セッション）

> 最新の状況（GPU サーバーでの学習結果と次の作業）は [HANDOFF_SERVER.md](HANDOFF_SERVER.md) を参照。このメモは録音前に書いたもの。

ローカルの Claude Code に最初に貼るプロンプト:

```text
このリポジトリ（my-voice）の作業を、クラウドの Claude Code セッションから引き継ぎます。
まず docs/HANDOFF.md と README.md を読み、内容を把握したうえで「現在の状況」と「次のステップ」を要約してください。
コードを変える前に、uv sync --extra asr と uv run pytest が通ることを確認してください。
以降は日本語で、uv でパッケージを管理する前提で進めてください。
```

---

## 目的

自分の声を録音し、日本語 TTS **Irodori-TTS v4.1-Small**（Aratako/Irodori-TTS、MIT）を **LoRA で fine-tuning** して自分専用の音声モデルを作る。

- 学習: 研究室の GPU サーバー（**RTX PRO 4000, VRAM 24GB, Blackwell**）に SSH で接続
- 推論: **MacBook Pro M2**（MPS, fp32）
- 録音: **iPhone 17**（USB マイクは持っていない）

## これまでに決めたこと

| 項目 | 決定内容 | 理由 |
|---|---|---|
| モデル | Irodori-TTS v4.1-Small + LoRA | 日本語専用で表現力が高く、MIT ライセンス。公式に LoRA 学習の設定がある。Mac の MPS で推論できる |
| 比較候補 | Style-Bert-VITS2 JP-Extra（安定していて速い）、GPT-SoVITS、piper-plus（CPU で軽い）、MioTTS | 必要になったら試す |
| 台本 | ITA コーパス（パブリックドメイン）。Emotion 100 文、Recitation 324 文 | 音素のバランスが良く、自分の声を録る用途の定番 |
| 録音フェーズ | phase1: Emotion100 を普通の話し方で（約10分）／phase2: Recitation324 を普通の話し方で（合計約1時間）／phase3: Emotion100 を喜び・怒り・悲しみで | phase2 では、phase1 と phase2 のデータを合わせて**ベースモデルから学習し直す**（追加学習はしない） |
| 感情 | 感情ごとに録音し、`caption` 列に感情の説明を書いて学習する。推論時は `--style`（caption）と、その感情で読んだ参照音声で指定する | 普通の話し方だけで学習すると、感情表現が弱くなる恐れがあるため |
| 管理 | uv のモノレポ。Irodori-TTS は `third_party/` に置き、独立した uv 環境にする | |
| データの扱い | 録音・データセット・LoRA はコミットしない（`.gitignore` 済み） | 声のなりすまし対策 |

## 録音の設定（テスト済み）

- アプリ: iPhone の PCM 録音アプリ（WAV / **48kHz / 24bit / モノラル**）。ボイスメモは AAC 64kb/s で、16kHz 以上の音が削られるため使わない
- 置き方: タオルの上に寝かせ、**下側（USB-C 側のマイク）を口に向けて**15〜25cm、少し斜めから
- 読み方: 普通の会話くらいの声の大きさで、**ゆっくりめ（1秒あたり8〜9モーラが目安）**。文と文の間は **1.5〜2 秒**、録音の最初と最後は 2〜3 秒黙る。言い直すときは 3 秒黙ってから、その文を最初から読む
- 1 ファイル 25 文（`myvoice script` で作る台本 1 つ分）。ファイル名はセッション ID にする（例: `emotion-normal-s01.wav`）

### テスト録音の結果（Emotion の 1〜5 文目、WAV）

- 雑音 −61dBFS、S/N 41〜45dB、声の大きさ −22dBFS 前後、最大 −4dBFS で音割れなし、文の間 1.8〜2.2 秒
- 5 区間を検出して 5 文と一致し、全文 OK。1秒あたりのモーラ数は 8.7〜10.1（後半が少し速い）
- 判定: そのまま本番に進んでよい

## 現在の状況

- ブランチ `claude/voice-pipeline` に実装を push 済み（`main` にはまだコミットがない）。リポジトリが空の状態で最初に push したので、このブランチが GitHub 上の既定ブランチになっている可能性がある。必要なら `main` にマージして、既定ブランチを `main` に変える
- `myvoice` CLI: doctor / corpus / script / check / split / status / qc / build / irodori setup / train prepare・run・samples / infer
- テスト 13 件はすべて通過（合成音声で、分割、言い直しの除去、品質チェック、データセット作成、Irodori の呼び出しコマンドを検証）
- **本番の録音はまだ始めていない**

## 未検証・注意点

1. **Irodori-TTS を実際に動かしたことはまだない。**クラウド環境からは pytorch.org に接続できず、`uv sync` が失敗した。Irodori-TTS のソース（コミット `f01c08cd40c0afa62a862439a5de12f6dcf62054`、2026-11-18）を読んで、各スクリプトの引数名と動作は照合済み
   - `prepare_manifest.py` は内部で `datasets.load_dataset(path=...)` を呼んでいる。ローカルフォルダ（audiofolder 形式: 音声と `metadata.csv`）を渡しても読めるはずだが、実際には試していない。`--normalize-db` を使うには audiotools が必要
   - Blackwell 世代の GPU を使うには `--backend cu128`（CUDA 12.8）が必要
2. **少量データ用の学習設定は私の推測。**公式の LoRA 設定は batch 40×勾配蓄積2、30000 steps と大規模データ向けなので、`configs/project.toml` の `[irodori.train_overrides]` で batch 16、3000 steps、500 steps ごとに保存、に上書きしている。`train samples` で各チェックポイントを聞き比べて選ぶ。VRAM が足りなければ `-- --gradient-checkpointing` を付けるか、batch を 8 に下げる
3. 分割で区間数が文の数と合わないとき（言い直しなど）に自動で対応付けるには `uv sync --extra asr`（faster-whisper large-v3）が必要。無い場合は `data/review/<session>/` に区間を書き出して止まる
4. 1秒あたりのモーラ数が 3.5〜11 の範囲を外れると警告を出す（台本とのずれや読み飛ばしの検出用）

## 次のステップ

1. ローカル（Mac）の準備: `brew install uv ffmpeg`、`uv sync --extra asr`、`uv run myvoice doctor`、`uv run myvoice script --phase phase1`
2. phase1（`emotion-normal-s01`〜`s04`）を録音し、`uv run myvoice split recordings/emotion-normal-s0*.wav`、続けて `uv run myvoice qc`
3. `uv run myvoice build v1 --phases phase1` を実行し、データセットを GPU サーバーへ rsync
4. サーバーで `uv run myvoice irodori setup --backend cu128`、`uv run myvoice doctor`、`train prepare v1`、`train run v1`、`train samples v1 --with-base`。**ここで初めて Irodori-TTS が実際に動くので、エラーが出たらその場で直す**
5. Mac で `uv run myvoice irodori setup`（cpu extra。MPS が使える）を実行し、LoRA を持ってきて `uv run myvoice infer --adapter ... --ref auto --dataset v1 --text ...`
6. 手順全体が問題なく動いたら、phase2 を録音し、v2（phase1+phase2）を作って最初から学習し直す

ローカルのセッションでは、SSH や rsync のコマンド（ホスト名やパス）を自分の環境に合わせて組み立てる。サーバーのホスト名はまだ共有していない。
