# 引き継ぎメモ（GPU サーバー → Mac の Claude Code セッション）

GPU サーバー（ssh 名 `helen-home`、RTX PRO 4000 Blackwell 24GB）で 2026-11-26〜27 に行った作業のまとめ。
`docs/HANDOFF.md` は録音前に書いた古いメモで、ここが最新。全体の使い方は README.md。

## 方針（ユーザーとの約束）
- 日本語でやりとりする。パッケージは uv で管理する。
- 数分以上かかる処理（大きなダウンロード、train prepare / run / samples、まとまった推論）は、
  内容と目安の時間を先に伝え、確認を取ってから実行する。数秒で終わるものはそのまま実行してよい。
- 頼まれたことの範囲で動く。大がかりな比較や分析を勝手に始めない（止められたことがある）。
- 声のデータ（data/、work/、outputs/、録音、LoRA）は絶対にコミットしない。
- コードを直したら、テストを通してからコミットする。push は確認を取ってから。
- GPU を使うときは OOM と温度に気をつける（サーバーでは監視スクリプトを付けて実行した）。

## データ
| データセット | 中身 | 量 |
|---|---|---|
| v1 | phase1（ITA Emotion 100 文、普通の話し方） | 100 文・約 6.9 分 |
| v2 | phase1 + phase2（ITA Recitation 324 文） | 424 文・約 25.6 分（本命） |

検証用に 10% を分けている（v2: 学習 382 文 / 検証 42 文、`split_train_valid_indices(seed=0)`）。

## コードの変更（すべて push 済み、ブランチ claude/voice-pipeline）
- a6648fd: 少量データ向けの学習設定、参照音声の複数自動選択（合計約 30 秒、`--ref-wavs`）、
  `irodori download`、CLI の引数バグ修正（`--` の後ろだけを train.py に渡す）
- e50bc9b: `[irodori.env]` に `IRODORI_SDPA_CUDNN = "0"`。cuDNN の SDPA が入力の形が変わるたびに
  CPU で実行計画を作り直していて、学習が 2.6 s/step → 0.5 s/step（約 5 倍速）になった。推論にも効く
- f28c389: v2 用の学習設定（3000 steps、500 steps ごとに保存、warmup 100 / stable 2300）
- 6c00b6a: 自作台本（ITA 以外）の読み込みと phase4（会話調、style `casual`）を追加
- 539511f: 一括合成。`irodori_scripts/batch_infer.py` が 1 つのプロセスで infer.py の main() を文ごとに呼び、
  モデルの読み込みを Irodori 自身のキャッシュ（get_cached_runtime）に通す。読み込み 約 20 秒 + 1 文 約 1.5 秒
  （以前は 1 文 約 25 秒）。seed を固定すると 1 文ずつの infer.py と完全に同じ音声になることを確認済み
  （LoRA と base の切り替えも）。`myvoice infer` の複数文・`train samples`・1 文の `infer` すべてがこれを通る。
  `--text-file` の行を `名前<TAB>文` にすると `<名前>.wav` で保存
- e139e43: 保存前にピークを `[infer] peak_dbfs`（-1.0）に抑える。Irodori は波形をそのまま 16bit で保存するので、
  1.0 をわずかに超えた部分が音割れになっていた（それまでの生成音声はほぼ全部に数〜24 サンプルの音割れがある）
- テストは 17 件すべて通る

## 学習の結果
LoRA r16（diffusion_attn、学習するパラメータ 26.4M / 792M）、batch 16。

### v1-r16（v1、1500 steps、約 14 分）
- 検証 loss は 250 steps で最小（0.68）、そのあと上がり続けた（1500 で 0.84）。データが少なく過学習
- 聞いた感想: base よりは自分っぽいが、まだ違う

### v2-r16（v2、3000 steps、約 23 分）
- 検証 loss: 500 → 0.703、1000 → 0.693、1500 → 0.685、**1800 → 0.683（最小）**、2000 → 0.689、
  2500 → 0.684、3000 → 0.703。はっきりした過学習はなく、1500〜2500 あたりが良さそう
  （検証 loss はランダムな t・ノイズで揺れるので、最後は耳で選ぶ）
- GPU: 最高 86℃（電力上限 145W で頭打ち）、VRAM 最大 15.5GB
- チェックポイント: サーバーの `outputs/runs/v2-r16/checkpoint_0000500` 〜 `0003000`（各 約 300MB）。
  **Mac にはまだ持ってきていない**

## Mac に rsync した聞き比べ用の音声
どれも `listen.md` を VS Code の Markdown プレビュー（⇧⌘V）か Typora で開くとその場で再生できる。

| Mac 側のフォルダ | 中身 |
|---|---|
| `outputs/listen_v1/` | 検証用 3 文（学習に未使用）: 本物の録音 / base / v1 500 / v1 1500 |
| `outputs/runs/v2-r16/samples/` | テスト文 4 文 × base と v2 の 500〜3000 steps |
| `outputs/casual_v2/` | 口語の 4 文 × caption（普通 / くだけた）× base / v1 1500 / v2 2000 |

参照音声はすべて、データセットから自動選択した朗読の文（約 30 秒）。

## 動画ナレーションの合成（2026-11-27）
- 動画のナレーション用に 22 文を合成した。モデルは v2-r16/checkpoint_0002000、参照音声は v2 から自動選択
  （朗読、約 30 秒）、style は normal
- 出力はサーバーの `outputs/narration/01.wav`〜`22.wav` と `listen.md`（文・長さ・判定・プレイヤーの一覧）。
  Mac へは `rsync -av helen-home:/workH/kmt/prv/my-voice/outputs/narration ~/my-voice/outputs/`
- 22 文の文は `work/narration/texts.tsv`（`id<TAB>文`、英字・数字はかなに置き換え済み）
- チェック結果: ピークは全部 -1.0 dBFS で音割れなし、無音や極端に長いものもなし。16「次に、学習後。」だけ
  0.27 秒/字と少しゆっくり。文字起こしでの読みの確認はしていない（サーバーに faster-whisper が無い）
- 最初は音割れ対策なしで作った（そのテイクは `work/narration/old_clipped/` に残してある）。対策を入れて作り直したので、
  seed が変わり抑揚は別のテイクになっている（長さは同じ）
- ユーザーはまだ作り直した後の音声を聞いていない

## わかったこと（ユーザーの感想）
- v2 で声質は似てきたかもしれない（v1 の時点で似ていた可能性もある）
- 自分っぽくない一番の理由は**話し方**。普段はもっとくだけた会話調で話すのに、合成音声は朗読のように硬い
- caption を「くだけた自然な口調」に変えても、ほとんど変わらなかった。
  学習データ全部が同じ caption と朗読の組み合わせなので、caption では話し方が変わらないと考えている

## 次にやること: 会話調の録音（phase4）
ユーザーは「会話調の台本を、自由に話しているつもりで読む」方法を選んだ。
1. 台本（500 文に増やした）が `corpus/casual_transcript_utf8.txt` にある（リポジトリで管理。git pull で Mac にも来る）
   - 形式は ITA と同じ `CASUAL_001:本文,カタカナ読み`。本文を直したら読みも直す（台本との照合とモーラ数のチェックに使う）
   - 一人称（僕・私・俺など）はユーザーに合うものがわからないので入れていない
2. ユーザーが自分の言い回しに直す。口調に問題がなければ、全部で 250〜300 文（音声で 10〜15 分）まで書き足す
3. `uv run myvoice script --phase phase4` で台本を作る（25 文ずつ casual-casual-s01, s02, …）→ 録音 → split
4. 録れたら、まず学習し直さずに、会話調の録音を参照音声にして v2 で合成し、話し方が変わるかを確かめる
5. `myvoice build v3 --phases phase1,phase2,phase4` → v3 を学習する。会話調の録音には casual の caption が付くので、
   推論時に `--style casual` で会話調に切り替えられるようになるはず。データが増えるので max_steps は見直す

## 会話調（phase4）の録音の処理（2026-11-27）
手順 1〜3 は済んだ。台本は 500 文に増え、20 セッションぶん録音した。学習（v3）はまだしていない。
- 録音 20 本（WAV、各 2 分強、合計 約 47 分）を `recordings/casual_raw/`（録音日時の名前）から日時順に
  `recordings/casual-casual-s01.wav`〜`s20.wav` へコピーした（1 本 = 25 文、s01 = 001〜025 … s20 = 476〜500）。
  casual_raw は残してある
- 文字起こし: `uv sync --extra asr`（faster-whisper 1.2.1 / ctranslate2 4.9.1）、モデルは large-v3
  （`Systran/faster-whisper-large-v3`、約 3GB）。サーバーのシステムには cuDNN 9 が無いので、Irodori の venv のものを
  `LD_LIBRARY_PATH` に足して GPU で動かした（`work/casual/asr.env`）。GPU は最大 82℃・VRAM 3.7GB
- `split --asr` は 20 本で約 11 分（1 本 約 30 秒）。`qc --asr` は split の文字起こしを使い回すので数十秒
- 結果: **499 / 500 文、声の部分の合計 17.7 分**（1 文 0.9〜4.2 秒、中央値 2.1 秒、SNR 中央値 45 dB）。エラー 0
  - s06: CASUAL_148「たしかに。そのほうが迷わないね。」が録音に入っていない（147 と 149 の間が無音）。
    区間が 24 個で split が止まったので、録音を 129 秒で 2 つに切って（`work/casual/s06_part1/2.wav`）
    `--start 1 --count 22` と `--start 24 --count 2 --append` で切り分けた
  - 言い直しは見つからなかった（s06 以外は、無音で区切った区間がちょうど 25 個）
  - 警告 101 件: 99 件は話す速さ（11〜14 モーラ/秒）。上限 11 は朗読向けの値で、会話調では誤検知。
    文字起こしの一致度が 0.55 未満は 2 件（019、468）。0.75 未満は 23 件で、多くは漢字とかなの違いか文字起こしの誤り
- 台本と違う言い回しで読んだ疑いのある文（聞いて確認する）: 010（何にする → 何食べる？）、019（だったと思う → だと思う）、
  020（早起きしなきゃ → 早くしなきゃ）、077（何食べたい気分 → 「何と言ったらいい気分」と文字起こし）、
  468（電話しようか → 「出ましょうか」と文字起こし）
- QC の表: `data/reports/qc_20261127_165520.tsv`

### 撮り直し（6 文、まだ録っていない）
ユーザーが上の 6 文（010、019、020、077、148、468）を撮り直すことにした。台本の文のとおりに読む。
台本はサーバーの `data/scripts/phase4/casual-retake-01.txt`（data/ なのでコミットしていない）。中身は次の順:
1. 今日の昼ごはん、何にする？（010 = s01 の 10 番）
2. たぶん十時くらいだったと思う。（019 = s01 の 19 番）
3. ほんとに？じゃあ早起きしなきゃ。（020 = s01 の 20 番）
4. 何食べたい気分？（077 = s04 の 2 番）
5. たしかに。そのほうが迷わないね。（148 = s06 の 23 番）
6. 電話しようか？（468 = s19 の 18 番）

- 1 本にまとめて `casual-retake-01.wav` として録り、サーバーの `recordings/` に置く。録る環境は元の 20 本とそろえる
- サーバー側の処理: 無音で 6 区間に切り、文ごとに `myvoice split <区間>.wav --session <元のセッション> --start <番号> --count 1 --asr --append`
  で差し替える（`--append` は同じ utt_id を上書きする）。差し替え前に元の wav と segments.jsonl を work/ に退避する。
  そのあと `qc --asr` をやり直す

### 撮り直しの処理（2026-11-27 済み）
- 録音は 2 本（`recordings/casual-retake-01.wav` 32.1 秒、`casual-retake-02.wav` 10.2 秒、どちらも 48kHz・モノラル・24bit）。
  台本の順ではなく、01 は 148 → 468 → 077 → 019 → 010 →（020 の「ほんとに？」抜け。使わない）、02 は 020
- 元の segments.jsonl（s01・s04・s06・s19）と 5 文ぶんの wav を `work/casual/before_retake/` に退避した（148 は元から無い）
- 無音で切った区間と文字起こしがユーザーの表と一致したので、無音の真ん中で切って `work/casual/retake/r_<番号>.wav` にし、
  `split --session ... --start N --count 1 --asr --append` で差し替えた
- 一致度（前 → 後）: 010 0.67 → 0.86、019 0.54 → 0.71、020 0.69 → 0.69（「早くしなきゃ」。ユーザーがこのまま使うと判断）、
  077 0.59 → 1.00、148 なし → 0.77、468 0.50 → 0.50（large-v3 は 2 回とも「出ましょうか」と文字起こし。
  Mac の文字起こしでは「電話しようか」だったので、文字起こしの誤りとみなす）
- `qc --asr` をやり直した結果: **500 / 500 文、合計 17.7 分**、エラー 0、警告 101（99 件は話す速さの誤検知、
  一致度の警告は 468 の 1 件、先頭の無音 1.3 秒が 121 の 1 件）。表は `data/reports/qc_20261127_171449.tsv`

## 参照音声だけを会話調にした v2 の合成（2026-11-27）
- `work/casual_ref/gen.py` → `outputs/casual_ref/`（listen.md）。v2-r16 2000 steps、caption normal、参照音声＝朗読 / 会話調、10 文
- 長さは参照音声を変えてもほぼ同じ（差 0.1 秒以内）。Irodori は文から長さを予測していて、v2 は朗読の長さしか学んでいない。
  本物の会話調の録音より 1.3〜1.4 倍長い。ユーザーはこのあと聞く前に「やっぱり学習させたい」と決めた

## v3 の学習（2026-11-27）
- データ: `data/datasets/v3` = 924 文・43.3 分（朗読 424 文 caption normal + 会話調 500 文 caption casual）。
  サーバーには phase1/2 の processed が無いので、`myvoice build` ではなく v2 と casual のデータセットをつなげて作った
  （wav はハードリンク、stats.json に note あり）。検証用 92 文 / 学習 832 文（約 52 steps/epoch）
- 戦略: base から学習し直し（v2 から続けない）、LoRA r16 diffusion_attn、5000 steps（≒96 epoch）、500 steps ごとに保存。
  Irodori の LoRA 学習は duration predictor も丸ごと学習する（`irodori_tts/lora.py` の modules_to_save）ので、
  caption casual で話す速さも変わることを期待している
- 実行: `work/run_v3.sh`（tmux `v3`）。prepare 35 秒、学習 37 分 22 秒（17:34:14〜18:11:36）、約 2.3 steps/s
- GPU の見張りは `work/gpu_guard.sh` に変えた: 87℃ が 30 秒続いたら SIGSTOP で一時停止 → 75℃ で SIGCONT 再開、
  93℃ か一時停止 30 分で停止。runner は落ちたら最新チェックポイントから最大 2 回再開（OOM なら gradient checkpointing）。
  今回は一度も働かなかった（最高 86℃、VRAM 最大 15.7GB）
- 検証 loss: 100 → 0.760、500 → 0.730、1000 → 0.710、2000 → 0.709、2800 → 0.699、4400 → 0.698、**5000 → 0.692（最小）**。
  2500 steps 以降は 0.70 前後でゆっくり下がり続け、過学習は見えない。dur_mae は 7 前後でほぼ一定
- チェックポイント: `outputs/runs/v3-r16/checkpoint_0000500`〜`0005000` と final
- 聞き比べ（`work/v3_listen/gen.py` → `outputs/v3_listen/listen.md`）: v3 の 2500 / 3500 / 5000 steps × {caption casual・参照 会話調, caption normal・参照 朗読, caption casual・参照 朗読} と v2 2000 steps。本物と並べる会話調 5 文は v3 の検証用（学習に不使用）から選んだ。
  会話調 5 文の長さ（本物＝1.00）: v3 は caption casual で 1.04（参照が朗読でも同じ）、caption normal で 1.10〜1.12、v2 は 1.30。2500〜5000 steps で耳での差はほぼない
- **標準のモデルを v3-r16 checkpoint_0005000 に決定**（2026-11-27、本人が聞いて）。5000 steps より先は伸ばさない（2500 steps 以降は横ばい、最後の下がりは学習率を下げた効果）

## 動画 第2弾のナレーションと聞き比べ用の音声（2026-11-27）
- 文: `work/narration_part2/texts.tsv`（id・text_for_tts・subtitle、1 行目は見出し、31 文、タメ口の会話調）。合成は text_for_tts
- 合成: `work/narration_part2/gen.py`（run.sh、tmux）。v3-r16 checkpoint_0005000、caption casual、参照音声は casual から自動選択
  （11 文・30.6 秒、v3_listen の cas-cas と同じ）。音割れ対策は前回と同じ `[infer] peak_dbfs = -1.0`。合成 69 秒、最高 74℃・VRAM 5GB
- チェック: `work/narration_part2/check.py`（faster-whisper large-v3、cuDNN は `work/casual/asr.env`）→ `check.tsv` と
  `outputs/narration_part2/listen.md`。長さ・秒/字・ピーク・無音（qc）と、文字起こしと台本の一致度（0.8 未満を要確認）
- 結果: 31 文・合計 126.2 秒（2 分 6 秒）。全部ピーク -1.0 dBFS で音割れなし、無音・極端な長さもなし、一致度は最低 0.83
  （表記ゆれ程度）。作り直し: 21（「同じでジーピーユーで」と余計な「で」→ seed 11）、28（「モデルの音」が「モデルノート」
  → seed 33）。23 は秒/字 0.18 と遅めだが、作り直しても長さが同じで「の」が抜けたので元のまま。
  テイクは `work/narration_part2/takes/`（元のテイクは `takes/orig/`）、理由は `replaced.tsv` と listen.md
- 聞き比べ用: `outputs/part2_samples/`（README.md に一覧）。新しい合成はなし、既存からコピー
  - recap_v2.wav / refonly_v2_casualref.wav: v2 2000 steps の「いやー、昨日さ、なんか全然寝れなくてさ。」を
    参照 朗読 / 参照 会話調で（`outputs/casual_ref/{read,casual}/t1.wav`、どちらも 3.68 秒）
  - compare_{A,B,C}_{v2,v3,real}.wav: v3 の検証用の会話調 c022 / c477 / c312（v2 と本物の差が大きい 3 文）。
    本物比 v2 1.40 / 1.53 / 1.29、v3 1.10 / 1.05 / 1.09
- Mac へ: `rsync -av helen-home:/workH/kmt/prv/my-voice/outputs/narration_part2 helen-home:/workH/kmt/prv/my-voice/outputs/part2_samples ~/my-voice/outputs/`
- ユーザーはまだ聞いていない
- **台本の修正（2026-11-28）**: ユーザーが Mac で texts.tsv を直し、05・06・07・15・17・19・20 の 7 文だけ同じ設定で合成し直した
  （`gen.py 05 06 07 15 17 19 20`、run_rev1.sh）。直す前のテイクと check.tsv は `work/narration_part2/takes/before_rev1/`。
  チェックは 31 文とも OK。17 だけ「前回のモデル**が**」と聞こえた（台本は「は」）ので seed 11 / 22 / 33 で作り直し、
  「は」と読めた seed 11 を採用（文字起こしでは「テンポ」が「店舗」だが同じ読み。聞いて確認してもらう）。
  合計は 126.2 → 131.0 秒（2 分 11 秒）。前回の 21・23・28 のテイクは `takes/round1/` に移した

## 注意点
- 生成音声には、ピークが 0 dBFS に当たるものがある（base でも同じ）。今のところ聞いて問題はないが、未対処
- LoRA の出力は base より約 10〜15% 速くなり、本物の録音の長さに近い
- サーバーから GitHub に push するときは、https では認証情報がないので SSH の URL を使った
