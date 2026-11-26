# NDLOCR-Lite v2 マルチベンダー GPU 対応ガイド

本リポジトリは [ndl-lab/ndlocr-lite](https://github.com/ndl-lab/ndlocr-lite) v2.0.0 をベースに、NVIDIA CUDA / AMD Radeon / Intel Arc での GPU 推論、PARSeq 動的バッチ化、CPU/GPU 3段パイプライン並列化に対応したフォークです。v2.0.0 の全機能（DEIM レイアウト認識、3段 PARSeq 文字認識カスケード、TableCenterNet v3 表構造解析、縦中横 `--enable-tcy`、Text DAISY、異体字包摂、PDF 透明テキスト、GUI）を **FP32 既定精度・出力劣化なし** で利用できます。

---

## 1. 対応バックエンド一覧

`--device auto`（既定）時は以下の優先順で利用可能なバックエンドを自動選択し、初期化や実行に失敗した場合は警告を表示して自動的に `cpu` へフォールバックします（`--strict-device` 指定時はエラーで停止）。

| `--device` 指定値 | Execution Provider | 対象 GPU / 環境 | 表構造解析 (`TableCenterNet`) | 備考 |
|---|---|---|---|---|
| `auto` | 自動判定 | CUDA > MIGraphX > ROCm > OpenVINO GPU > WebGPU > DML > CPU | EP に準拠 | 既定値 |
| `cuda` | `CUDAExecutionProvider` | NVIDIA GPU (Win / Linux) | 対応 (ORT 1.31.1) | PyPI 既定は CUDA 13 系。CUDA 12 系は下表参照 |
| `migraphx` | `MIGraphXExecutionProvider` | AMD Radeon (Linux ROCm) | 対応 (ORT 1.27.1) | Linux + ROCm 導入環境向け |
| `rocm` | `ROCMExecutionProvider` | AMD Radeon (Linux ROCm) | 要 ORT>=1.25 | 別ビルドの ORT-ROCm 導入環境向け |
| `openvino` | `OpenVINOExecutionProvider` (`GPU`) | Intel Arc / Intel 内蔵 GPU (Win / Linux x86_64) | 対応 (Plugin EP + ORT 1.31.1) | `auto` 時は GPU デバイス検出時のみ選択 |
| `webgpu` | `WebGpuExecutionProvider` | AMD Radeon / Intel Arc / NVIDIA / 他 (Win D3D12 / Linux Vulkan) | 対応 (Plugin EP + ORT 1.31.1) | 汎用 GPU バックエンド |
| `dml` | `DmlExecutionProvider` | Windows DirectX 12 GPU (`onnxruntime-directml==1.24.4`) | 制限あり (ORT < 1.25 のため `DeformConv` 非対応) | 通常の Windows 汎用利用は `webgpu` を推奨 |
| `cpu` | `CPUExecutionProvider` | 全環境 | 対応 | 従来動作 |

---

## 2. インストール方法

> [!IMPORTANT]
> `onnxruntime` と `onnxruntime-gpu` を同一環境に混在インストールするとパッケージファイルが衝突するため、GPU 版へ切り替える際は必ず `onnxruntime` をアンインストールしてから導入してください。

### 2.1 汎用 GPU 構成（AMD Radeon / Intel Arc / Vulkan・D3D12 対応 GPU）

`onnxruntime==1.31.1` に Plugin EP（`onnxruntime-ep-webgpu==0.6.0`・`onnxruntime-ep-openvino==1.8.1`）を組み合わせて導入します。ORT 1.31.1 ベースを維持するため、`TableCenterNet` の `DeformConv` もそのまま利用できます。

```bash
pip install -r requirements-gpu-generic.txt
```

- Linux で `webgpu` を使う場合はホストに Vulkan ドライバ（`libvulkan.so.1`）が必要です。
- Intel Arc / Core Ultra GPU で OpenVINO を使う場合は最新の Intel GPU ドライバ（および Linux では Intel Compute Runtime / Level Zero）を導入してください。

### 2.2 NVIDIA CUDA 構成

`onnxruntime-gpu==1.31.1` を導入します（CUDA Toolkit / cuDNN ランタイム環境に応じたインデックスを選択）。

**CUDA 13 系環境（PyPI 既定）:**
```bash
pip uninstall -y onnxruntime
pip install -r requirements-cuda.txt
```

**CUDA 12 系環境（Google Colab 等を含む、`onnxruntime-cuda-12` Azure インデックスは `onnxruntime-gpu==1.30.0` まで提供、ORT>=1.25 で TableCenterNet 対応）:**
```bash
pip uninstall -y onnxruntime
pip install -r requirements-cuda.txt
pip install --force-reinstall --no-deps \
  --index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/onnxruntime-cuda-12/pypi/simple/ \
  onnxruntime-gpu==1.30.0
```

### 2.3 AMD Radeon Linux ROCm（MIGraphX）構成

ROCm 6.x 以降を導入済みの Linux x86_64 環境向けです。

```bash
pip uninstall -y onnxruntime
pip install -r requirements-migraphx.txt
```

---

## 3. CLI / GUI の使い方

### 3.1 デバイス診断

現在利用可能な EP・検出された GPU デバイス・`--device auto` の選択結果を確認します。

```bash
ndlocr-lite --check-device
# またはソース実行:
python3 src/ocr.py --check-device
```

### 3.2 CLI オプション一覧

| オプション | 既定値 | 説明 |
|---|---|---|
| `--device` | `auto` | 使用する推論デバイス (`auto`, `cuda`, `migraphx`, `rocm`, `openvino`, `webgpu`, `dml`, `cpu`) |
| `--device-id` | `0` | マルチ GPU 環境でのデバイスインデックス |
| `--batch-size` | CUDA/MIGraphX/ROCm=`32`, OpenVINO/WebGPU/DML=`16`, CPU=`1` (`0`=自動) | PARSeq 文字認識のバッチサイズ上限 |
| `--strict-device` | 無効 (`False`) | 指定 GPU バックエンド初期化失敗時に CPU へフォールバックせず終了する |
| `--check-device` | 無効 (`False`) | デバイス診断結果を表示して終了する |

実行例:
```bash
# 自動選択 + 動的バッチ + 3段パイプライン
ndlocr-lite --sourcedir input_dir --output out_dir

# WebGPU を明示指定してバッチサイズ 16 で処理
python3 src/ocr.py --sourcedir input_dir --output out_dir --device webgpu --batch-size 16

# フォールバックを許可せず CUDA で厳密実行
python3 src/ocr.py --sourceimg sample.jpg --output out_dir --device cuda --strict-device
```

### 3.3 デスクトップ GUI

- メイン画面上に現在有効な推論エンジン（例: `実行エンジン: GPU (CUDA), batch=32` や `実行エンジン: GPU (WebGPU), batch=16`）のステータスが表示されます。
- **出力設定ダイアログ**内の「実行環境・GPU高速化」から、実行デバイス・GPU ID・バッチサイズを変更して即座にモデルを再読み込みできます。

---

## 4. 並列化と精度保証の仕組み

1. **PARSeq 動的バッチパッチとローカルキャッシュ**:
   公式 PARSeq 3モデル（`24x256-30`, `24x384-50`, `24x768-100`）は静的 `batch=1` でエクスポートされています。本フォークは実行時に動的バッチ axis (`N`) へパッチしたモデルを自動生成し、将来の FP16 拡張に備えたキャッシュ名（`{stem}_{hash12}_dynbatch_fp32_v1.onnx`）でローカルキャッシュします。生成時に `batch=1` との出力・`argmax` 一致検査を行い、異常があれば安全に `batch=1` 逐次処理へフォールバックします。
   - **DEIMv2 (`deim-s-1024x1024-261124.onnx`)** はデコーダ内の `GatherElements` 構造上 `batch=1` を維持し、**TableCenterNet (`tablecenternet_s_800_v3.onnx`)** も `batch=1` で動作します。
2. **3段パイプライン並列化**:
   複数ページ・PDF・ディレクトリ処理時に `[次ページ画像読み込み + DEIM 前処理 (CPU)]` → `[DEIM + 3段 PARSeq 動的バッチ + 表認識 (GPU)]` → `[XML / JSON / TXT / 可視化・PDF 書出し (CPU)]` をオーバーラップ実行します。
3. **精度の厳密維持**:
   全モデル既定 **FP32** 精度で動作し、CPU `batch=1` と同一の認識結果（テキスト・座標・読み順）を出力します。

---

## 5. 検証済み環境・実測ベンチマーク実行手順

以下の環境および自動テストで動作と出力一致を検証済みです。

- **ローカル自動テスト (`pytest tests ndlocr-lite-gui/tests`)**:
  - 公式 PARSeq 3モデル (`24x256-30` / `24x384-50` / `24x768-100`) の動的バッチパッチ (`_dynbatch_fp32_v1.onnx`) において、`batch=1` との厳密一致 (`ORT_DISABLE_ALL` 誤差 `<= 1e-4` および `argmax` 完全一致) を検証。
  - バッチ推論 + 3段パイプライン並列実行時と CPU `batch=1` 逐次実行時で、サンプル資料の XML / JSON / TXT 出力が完全一致することを検証（縦中横バッチ処理・DEIM 分割前処理含む）。
- **Plugin EP 構成 (`requirements-gpu-generic.txt`: ORT `1.31.1` + `onnxruntime-ep-webgpu==0.6.0` + `onnxruntime-ep-openvino==1.8.1`)**:
  - 両プラグインが正常登録され、OpenVINO CPU-only 環境（Intel GPU 未検出時）で誤選択せず `--device auto` が正しく `webgpu`（または `cpu`）へフォールバック・優先選択することを確認。
  - Linux Vulkan GPU 上の `WebGpuExecutionProvider` で DEIMv2、動的バッチ PARSeq（`batch=4`, `CPU batch=1` と出力一致）、TableCenterNet v3（`DeformConv`）がいずれも CPU フォールバックなしで動作することを確認。
- **Google Colab T4 Standard (`Tesla T4 15360 MiB`, `CUDA 13.0`, `Python 3.13.16`, `onnxruntime-gpu==1.31.1`)**:
  - `src/ocr.py --check-device` にて `CUDAExecutionProvider` の有効化、`TableCenterNet (DeformConv, ORT>=1.25): SUPPORTED`、`--device auto` による `cuda` 自動選択を確認。
  - 公式 PARSeq 3モデルの動的バッチキャッシュ生成と検証 (`_dynbatch_fp32_v1.onnx`) を確認。

お使いの CUDA 環境または Google Colab で 12 ページ連続ベンチマークおよび 5 画像フル OCR 一致検査を実行する場合は、以下を実行してください:

```bash
python3 scripts/colab_verify_gpu.py --pages 12 --json-out colab_gpu_report.json
```
