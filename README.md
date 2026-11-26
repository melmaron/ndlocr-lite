# NDLOCR-Lite（マルチベンダー GPU 対応フォーク）

[![CI](https://github.com/melmaron/ndlocr-lite/actions/workflows/ci.yml/badge.svg)](https://github.com/melmaron/ndlocr-lite/actions/workflows/ci.yml)
[![Build Flet Cross](https://github.com/melmaron/ndlocr-lite/actions/workflows/build-flet-cross.yml/badge.svg)](https://github.com/melmaron/ndlocr-lite/actions/workflows/build-flet-cross.yml)

[ndl-lab/ndlocr-lite](https://github.com/ndl-lab/ndlocr-lite) **v2.0.0** をベースに、**NVIDIA CUDA / AMD Radeon / Intel Arc** を含むマルチベンダー GPU 推論・PARSeq 動的バッチ化・CPU/GPU 3段パイプライン並列化に対応したフォークリポジトリです。

v2.0.0 の全機能（DEIMv2 レイアウト認識、3段 PARSeq 文字認識カスケード、TableCenterNet v3 表構造解析、縦中横 `--enable-tcy`、Text DAISY、異体字包摂、透明テキスト付き PDF 出力、Flet GUI）を **FP32 既定精度・出力劣化なし** で利用できます。

詳しいインストール手順や技術仕様は **[GPU セットアップガイド](./docs/GPU_SETUP.md)** をご覧ください。

---

## 元プロジェクト（NDLOCR-Lite v2.0.0）からの主な拡張機能

- **マルチベンダー GPU 自動選択 (`--device auto`)**:
  環境に応じて `cuda > migraphx > rocm > openvino (GPU) > webgpu > dml > cpu` の優先順で自動選択し、利用不可または初期化失敗時は診断ヒント付きで `cpu` へ自動フォールバックします（`--strict-device` 指定時はエラーで停止）。
- **PARSeq 動的バッチ化 & ローカルキャッシュ**:
  静的 `batch=1` の公式 PARSeq 3 モデル（`24x256-30`, `24x384-50`, `24x768-100`）を実行時に動的バッチへ自動パッチし、将来の FP16 拡張に備えたキャッシュ名（`*_dynbatch_fp32_v1.onnx`）でローカル保存・再利用します。生成時に `batch=1` との等価性検証を行い、異常時は安全に `batch=1` へフォールバックします。
- **CPU / GPU 3段パイプライン並列化**:
  複数画像・複数ページ PDF の処理時に `[前処理プリフェッチ (CPU)] → [DEIMv2 + 3段 PARSeq バッチ + TableCenterNet (GPU)] → [XML / JSON / TXT / 可視化・PDF 書出し (CPU)]` をオーバーラップ実行します。
- **デスクトップ GUI 統合**:
  出力設定ダイアログに「実行環境・GPU高速化」設定（実行デバイス・GPU ID・バッチサイズ・即時モデル再読み込み）と、メイン画面上部のリアルタイムステータス表示（`実行エンジン: ...`）を追加しました。
- **Plugin EP 対応による ORT 1.31.1 維持**:
  `onnxruntime==1.31.1` 上で `onnxruntime-ep-webgpu==0.6.0`・`onnxruntime-ep-openvino==1.8.1` を動作させているため、ORT >= 1.25 が必要な **TableCenterNet (`DeformConv`)** も汎用 GPU 環境でそのまま動作します（OpenVINO は Intel GPU デバイス検出時のみ有効化し、CPU-only 環境での誤選択を防止しています）。

---

## 対応 GPU・バックエンド一覧

| `--device` | Execution Provider | 対象 GPU / OS | 既定バッチサイズ | TableCenterNet (`DeformConv`) | 導入要件ファイル |
|---|---|---|---|---|---|
| `auto` | 自動判定 | 上記優先順で最適バックエンド選択 | バックエンド準拠 | EP 準拠 | 用途に応じて選択 |
| `cuda` | `CUDAExecutionProvider` | NVIDIA GPU (Windows / Linux) | `32` | 対応 (ORT 1.31.1 / 1.30.0) | [`requirements-cuda.txt`](./requirements-cuda.txt) |
| `migraphx` | `MIGraphXExecutionProvider` | AMD Radeon (Linux + ROCm 6.x+) | `32` | 対応 (ORT 1.27.1) | [`requirements-migraphx.txt`](./requirements-migraphx.txt) |
| `rocm` | `ROCMExecutionProvider` | AMD Radeon (Linux + ORT-ROCm) | `32` | 要 ORT >= 1.25 | 別ビルド ORT-ROCm |
| `openvino` | `OpenVINOExecutionProvider` (`GPU`) | Intel Arc / Intel 内蔵 GPU (Win / Linux x86_64) | `16` | 対応 (ORT 1.31.1 + Plugin) | [`requirements-gpu-generic.txt`](./requirements-gpu-generic.txt) |
| `webgpu` | `WebGpuExecutionProvider` | AMD Radeon / Intel Arc / NVIDIA 他 (Win D3D12 / Linux Vulkan) | `16` | 対応 (ORT 1.31.1 + Plugin) | [`requirements-gpu-generic.txt`](./requirements-gpu-generic.txt) |
| `dml` | `DmlExecutionProvider` | Windows DirectX 12 GPU | `16` | **非対応** (`1.24.4` 制約) | `onnxruntime-directml==1.24.4` |
| `cpu` | `CPUExecutionProvider` | 全環境 (Win / macOS / Linux) | `1` | 対応 | [`requirements.txt`](./requirements.txt) |

---

## クイックスタート

### 1. デスクトップアプリケーション（GUI）

> [!WARNING]
> **デスクトップアプリケーションを利用する際には、日本語（全角文字）を含まないパスにアプリケーションを配置してください。全角文字を含む場合に起動しないことがあります。**

Windows / Linux 向けには **`gpu_generic`**（Radeon / Intel Arc / Vulkan・D3D12 汎用）と **`cuda`**（NVIDIA CUDA）の 2 系統に加え、macOS（Apple Silicon / Intel）向けビルドを GitHub Actions（[build-flet-cross](https://github.com/melmaron/ndlocr-lite/actions/workflows/build-flet-cross.yml) / [Releases](https://github.com/melmaron/ndlocr-lite/releases)）で生成しています。

操作方法の詳細は上流ドキュメント [NDLOCR-Liteの使い方](https://lab.ndl.go.jp/data_set/ndlocrlite-usage/)、ビルド方法は [デスクトップアプリケーションの利用方法](./ndlocr-lite-gui/README.md) を参照してください。

次の GIF アニメーションは、

[国立国会図書館総務部総務課 編『国立国会図書館年報』昭和27年度,国立国会図書館,1954. 国立国会図書館デジタルコレクション https://dl.ndl.go.jp/pid/3048008](https://dl.ndl.go.jp/pid/3048008/1/24)

を NDLOCR-Lite の画面キャプチャ機能によって、画像ファイルを介さずにテキスト化するデモを示しています。

<img src="resource/demo_lite.gif" width="600" alt="キャプチャモードのデモ動画">

### 2. コマンドライン（CLI）導入と実行

> [!IMPORTANT]
> Python 3.11 以上が必要です。`onnxruntime`（CPU/Plugin 用）と `onnxruntime-gpu` は同一パッケージ名空間を共有するため混在させないでください。CUDA へ切り替える際は必ず `pip uninstall -y onnxruntime` を行ってから導入してください。

```bash
git clone https://github.com/melmaron/ndlocr-lite
cd ndlocr-lite
python3 -m venv .venv
source .venv/bin/activate   # Windows PowerShell: .venv\Scripts\Activate.ps1
```

お使いの GPU に合わせていずれかをインストールします（詳細: [GPU セットアップガイド](./docs/GPU_SETUP.md)）:

```bash
# 1) 汎用 GPU（AMD Radeon / Intel Arc: WebGPU + OpenVINO Plugin EP）
pip install -r requirements-gpu-generic.txt

# 2) NVIDIA CUDA 13 系環境（PyPI 既定）
pip uninstall -y onnxruntime && pip install -r requirements-cuda.txt

# 3) NVIDIA CUDA 12 系環境（Google Colab 等を含む）
pip uninstall -y onnxruntime
pip install -r requirements-cuda.txt
pip install --force-reinstall --no-deps \
  --index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/onnxruntime-cuda-12/pypi/simple/ \
  onnxruntime-gpu==1.30.0

# 4) AMD Radeon Linux ROCm 環境（MIGraphX）
pip uninstall -y onnxruntime && pip install -r requirements-migraphx.txt

# 5) 標準 CPU 環境（パッケージインストール例）
uv tool install .
```

まず認識されているバックエンドを診断します:

```bash
python3 src/ocr.py --check-device
# またはパッケージインストール時:
ndlocr-lite --check-device
```

実行例:

```bash
# 1. ディレクトリ一括処理（--device auto で利用可能な GPU を自動使用）
python3 src/ocr.py --sourcedir 9892834_0001 --output tmpdir

# 2. 単一画像を WebGPU 明示・バッチサイズ 16 で処理
python3 src/ocr.py --sourceimg resource/digidepo_2531162_0024.jpg --output tmpdir --device webgpu --batch-size 16

# 3. 縦中横を有効化し、フォールバック禁止で CUDA 実行
python3 src/ocr.py --sourcedir input_images --output tmpdir --device cuda --strict-device --enable-tcy

# 4. PDF を直接 OCR して透明テキスト付き PDF を出力
python3 src/ocr.py --sourcepdf input.pdf --output tmpdir --pdf-output tmpdir/out_text.pdf

# 5. パッケージ導入コマンド（GUI 起動含む）
ndlocr-lite --sourceimg digidepo_1287221_00000002.jpg --output tmpdir
ndlocr-lite gui
```

<details>
<summary>モデルおよび動的バッチパッチのキャッシュ保存先</summary>

- `pip` / `uv` インストール時は初回 OCR 実行時に `https://lab.ndl.go.jp/dataset/ndlocr-lite/onnx/v2/` から未指定モデルを自動取得し（約 157 MB、GUI の表構造モデル追加時は +約 28 MB）、SHA-256 検証後にキャッシュへ保存します（リポジトリ直下 `python3 src/ocr.py` や配布版デスクトップ GUI は同梱 `src/model/` を使用）。
- モデルキャッシュ先:
  - Windows: `%LOCALAPPDATA%\ndlocr-lite\models\v2`
  - macOS: `~/Library/Caches/ndlocr-lite/models/v2`
  - Linux: `${XDG_CACHE_HOME:-~/.cache}/ndlocr-lite/models/v2`
- PARSeq 動的バッチパッチ生成物（`*_dynbatch_fp32_v1.onnx`）は上記キャッシュ配下の `optimized/`（環境変数 `NDLOCR_CACHE_DIR` 指定時は `$NDLOCR_CACHE_DIR/optimized/`）に保存されます。

</details>

---

## 主な CLI オプション一覧

| オプション | 既定値 | 説明 |
|---|---|---|
| `--sourcedir` | — | 入力画像ディレクトリ（jpg / jpeg / png / tiff / tif / jp2 / bmp / webp、PDF を順次処理） |
| `--sourceimg` | — | 単一入力画像ファイルのパス |
| `--sourcepdf` | — | 単一入力 PDF ファイルのパス |
| `--output` | 必須 (`--check-device` 時除く) | OCR 結果（XML / JSON / TXT 等）の出力先ディレクトリ（親ディレクトリ含め自動作成） |
| `--device` | `auto` | 推論デバイス (`auto`, `cuda`, `migraphx`, `rocm`, `openvino`, `webgpu`, `dml`, `cpu`) |
| `--device-id` | `0` | マルチ GPU 環境でのデバイスインデックス |
| `--batch-size` | CUDA/MIGraphX/ROCm=`32`, OpenVINO/WebGPU/DML=`16`, CPU=`1` (`0`=自動) | PARSeq 動的バッチ認識の最大バッチサイズ上限 |
| `--strict-device` | 無効 | 指定 GPU バックエンドの初期化・実行失敗時に CPU へフォールバックせず終了 |
| `--check-device` | 無効 | 利用可能な EP・検出 GPU デバイス一覧・`--device auto` の選択結果を表示して終了 |
| `--viz` | `False` | `--viz True` 指定で文字認識箇所の青枠可視化画像を出力 |
| `--json-only` | 無効 | `.xml` と `.txt` を出力せず JSON のみ出力 |
| `--enable-tcy` | 無効 | 縦書き資料（新聞等）の縦中横読み取りを有効化 |
| `--pdf-output` | — | 透明テキスト埋め込み PDF の出力パス |
| `--pdf-render-dpi` | `150.0` | PDF ページのレンダリング解像度 DPI |
| `--pdf-page-output` | 有効 (`--no-pdf-page-output` で無効) | 複数ページ PDF でページ単位の TXT / JSON / XML を出力 |
| `--pdf-visible-text` | 無効 | デバッグ用に PDF テキスト層を青字で可視描画 |

---

## 検証・ベンチマーク再現手順

現在の検証状況は以下の通りです（詳細・最新条件は [`docs/GPU_SETUP.md`](./docs/GPU_SETUP.md) 第 5 節参照）:

- **自動テストスイート (`pytest tests ndlocr-lite-gui/tests`)**: 全 67 テスト（+ 36 サブテスト）合格。公式 PARSeq 3 モデルの動的バッチ変換における `B=1` 厳密一致（誤差 `<= 1e-5`）・バッチ時 `argmax` 完全一致、および 3 段パイプラインの XML / JSON / TXT 完全一致を検証済み。
- **GitHub Actions CI**: `ubuntu-latest` / `windows-latest` × Python `3.11` / `3.12` の全 4 ジョブ合格、および `build-flet-cross` による 6 ターゲット（Windows/Linux × GPU Generic/CUDA、macOS Apple Silicon/Intel）の Flet ビルド成功を確認済み。
- **Plugin EP / CUDA 環境診断確認**: 隔離仮想環境（ORT `1.31.1` + WebGPU `0.6.0` + OpenVINO `1.8.1`）で WebGPU 実機推論（DEIMv2・PARSeq 動的バッチ・TableCenterNet v3）および OpenVINO CPU-only 誤選択防止を確認済み。Google Colab T4 Standard（`CUDA 13.0`, `onnxruntime-gpu==1.31.1`）で `--check-device`（`CUDAExecutionProvider`・TableCenterNet `DeformConv` 対応）と PARSeq 3 モデル動的バッチ生成を確認済み。

お使いの CUDA 環境または Google Colab で 12 ページ連続ベンチマークおよび 5 画像フル OCR 一致検査（`--enable-tcy` 有無含む）を実行する場合は、以下を実行してください:

```bash
python3 scripts/colab_verify_gpu.py --pages 12 --json-out colab_gpu_report.json
```

---

## OCR結果の例

|資料画像|OCR結果の冒頭（誤認識を含む）|OCR結果のxml|
|---|---|---|
|<img src="./resource/viz_digidepo_2531162_0024.jpg" width="400" alt="レイアウト認識結果"><br>国立国会図書館総務部 編『国立国会図書館スタッフ・マニュアル』E-2,国立国会図書館,1963.8. 国立国会図書館デジタルコレクション <https://dl.ndl.go.jp/pid/2531162/1/23>|(2) 気送子送付管気送手送付には、上記気送管にて送付するものと、空気の圧縮を使用せず.直接落下させる装置の二通りがある。後者の送付管は出納台左側に設置されており.5|[OCR結果(xmlファイル)](./resource/digidepo_2531162_0024.xml)|
|<img src="./resource/viz_digidepo_11048278_po_geppo1803_00021.jpg" width="400" alt="レイアウト認識結果"><br> 館内スコープ　次世代室の謎に迫れ！. 国立国会図書館月報. 2018, (683),　 p.20. <http://dl.ndl.go.jp/info:ndljp/pid/11048278>|はじめまして!私は2017年4月に就職後、次世代システム開発研究室(次世代室)という場所で仕事をしています。でも、「次世代室」って何をするところか想像しにくいですよね。次世代室は、図書館の役割がインターネット等の情報技術で変化する中、より先進的なサービスを検討していくために作られた比較的新しい部署です。|[OCR結果(xmlファイル)](./resource/digidepo_11048278_po_geppo1803_00021.xml)|
|<img src="./resource/viz_digidepo_3048008_0025.jpg" width="400" alt="レイアウト認識結果"><br>国立国会図書館総務部総務課 編『国立国会図書館年報』昭和27年度,国立国会図書館,1954. 国立国会図書館デジタルコレクション <https://dl.ndl.go.jp/pid/3048008/1/25>|第8章職員、庁舍、財政、記念行事等1.職員A.司書職員の研修昭和26年度(第4回)研修に引続き、昭和27年度(第5回)司書職員研修を27年6月2日から28年4月10日まで320時間、研修生50名に実施した。本年度は、東京学芸大学の協力を得て、講師は、本館職員が専門分野の科目を担当した外、東京学芸大学の教授が担当した。本年度も單位科目ごとに試験を実施し、必修科目(11單位)選択科目(4單位)合わせて15單位以上の試験に合格した43名が修了した。|[OCR結果(xmlファイル)](./resource/digidepo_3048008_0025.xml)|
|<img src="./resource/viz_tategaki2026-04-24-094138.png" width="400" alt="レイアウト認識結果">|春はあけぼのやうやう白くなりゆく山ぎはすこしあがりて、紫だちたる雲のほそくたなびきたる夏は夜。月のころはさらなり。やみもなは、蛍の多く飛びちがひたる、|[OCR結果(xmlファイル)](./resource/tategaki2026-04-24-094138.xml)|
|<img src="./resource/viz_yokogaki2026-04-24-103617.png" width="400" alt="レイアウト認識結果">|つれづれなるままに、日くらし、硯におかひて、八つに移りゆくよしなし事をそこはかとなく書きつくれば、あやしうこそものぐるほししけれ|[OCR結果(xmlファイル)](./resource/yokogaki2026-04-24-103617.xml)|

## モデルの認識性能について

### [F値による日本語活字の評価結果](/evaluation_jptype.csv)
日本語活字の図書及び雑誌に対するF値による評価結果を公表しています。

[NDLOCR ver.2.1の評価結果](https://lab.ndl.go.jp/data_set/r4ocr/r4_software#5-3-文字認識性能の評価結果)と同様の評価用データセット・同様の評価手法による値です。

### 手書き文字に対する評価(NDLOCR-Lite ver.2.0.0)

Maeda, K., & Okazaki, N. JaWildText: A Benchmark for Vision-Language Models on Japanese Scene Text Understanding. arXiv preprint arXiv:2603.27942, 2026.(<https://arxiv.org/abs/2603.27942>)

によって作成された[JaWildText（外部サイト）](https://huggingface.co/datasets/llm-jp/jawildtext)のうち、手書き文字データセット1,065画像を用いて評価しました。

- 全体CER（中央値）: `0.1327` / 全体F値（中央値）: `0.8927`（対象: 1065 ファイル）
- 縦書きCER（中央値）: `0.1776` / 縦書きF値（中央値）: `0.8536`（対象: 295 ファイル）
- 横書きCER（中央値）: `0.1156` / 横書きF値（中央値）: `0.9050`（対象: 768 ファイル）

※CER(Character Error Rate、文字誤り率): 0から1までの範囲を取り、0に近いほど高い性能を表すスコア

## モデルの再学習及びカスタマイズについて（開発者向け情報）

[学習及びモデル変換手順](/train/README.md)をご覧ください。

## 技術情報について（開発者向け情報）

NDLOCR-Liteは「レイアウト認識」、「文字列認識」、「読み順整序」の3つの機能（モジュール）を組み合わせて実現しています。

レイアウト認識にはDEIMv2[1]、文字列認識にはPARSeq[2]、表構造認識にはTableCenterNet[3]をそれぞれ用いており、読み順整序については当館が公開している[NDLOCR](https://github.com/ndl-lab/ndlocr_cli)と同様のモジュールを用いています。

- [1] Shihua Huang and Yongjie Hou and Longfei Liu and Xuanlong Yu and Xi Shen. Real-Time Object Detection Meets DINOv3. arXiv:2509.20787, 2025.(<https://arxiv.org/abs/2509.20787>)
- [2] Darwin Bautista, Rowel Atienza. Scene text recognition with permuted autoregressive sequence models. arXiv:2212.06966, 2022. (<https://arxiv.org/abs/2207.06966>)
- [3] Anyi Xiao and Cihui Yang. TableCenterNet: A one-stage network for table structure recognition. arXiv:2504.17522, 2025. (<https://arxiv.org/abs/2504.17522>)

レイアウト認識及び文字列認識の機械学習モデルは、いずれもpytorchをフレームワークとした学習を行った後にONNX形式に変換して利用しています。詳しくは[学習及びモデル変換手順](/train/README.md)をご覧ください。

## ライセンスとクレジット

本プログラムのベースとなった NDLOCR-Lite は、[NDLラボ](https://lab.ndl.go.jp)におけるこれまでの調査研究活動によって得られた知見、特に[NDL古典籍OCR-Lite](https://github.com/ndl-lab/ndlkotenocr-lite)の開発経験を踏まえて国立国会図書館職員が内製で開発し、CC BY 4.0ライセンスで公開しているものです。

詳細については[LICENCE](./LICENCE)をご覧ください。なお、本アプリケーションの実行時に利用するライブラリ等のライセンスについては[LICENCE_DEPENDENCEIES](./LICENCE_DEPENDENCEIES)をご覧ください。

レイアウト認識モデルのバックボーンには DINOv3 を蒸留した ViT-Tiny モデルを利用しています。学習済モデルを使用する場合には、次の DINOv3 ライセンスについても遵守をお願いいたします。
<https://github.com/facebookresearch/dinov3/blob/main/LICENSE.md>
