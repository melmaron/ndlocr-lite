# NDLOCR-Lite v2 マルチベンダー GPU 対応ガイド

本リポジトリは [ndl-lab/ndlocr-lite](https://github.com/ndl-lab/ndlocr-lite) v2.0.0 をベースに、NVIDIA CUDA / AMD Radeon / Intel Arc での GPU 推論、PARSeq 動的バッチ化、CPU/GPU 3段パイプライン並列化に対応したフォークです。v2.0.0 の全機能（DEIM レイアウト認識、3段 PARSeq 文字認識カスケード、TableCenterNet v3 表構造解析、縦中横 `--enable-tcy`、Text DAISY、異体字包摂、PDF 透明テキスト、GUI）を **FP32 既定精度・出力劣化なし** で利用できます。

---

## 1. 対応バックエンド一覧

`--device auto`（既定）時は以下の優先順で利用可能なバックエンドを自動選択し、初期化や実行に失敗した場合は警告を表示して自動的に `cpu` へフォールバックします（`--strict-device` 指定時はエラーで停止）。

| `--device` 指定値 | Execution Provider | 対象 GPU / 環境 | 表構造解析 (`TableCenterNet`) | 備考 |
|---|---|---|---|---|
| `auto` | 自動判定 | CUDA > MIGraphX > ROCm > OpenVINO GPU > WebGPU > DML > CPU | EP に準拠 | 既定値（ソフトウェア/仮想 WebGPU アダプタは自動除外） |
| `cuda` | `CUDAExecutionProvider` | NVIDIA GPU (Win / Linux) | 対応 (ORT 1.31.1 / 1.29.0) | PyPI 既定は CUDA 13 系。CUDA 12 系は下表参照。Ampere+ 一貫性のため既定で `"use_tf32": "0"` |
| `migraphx` | `MIGraphXExecutionProvider` | AMD Radeon (Linux ROCm) | 対応 (ORT 1.27.1) | Linux + ROCm 導入環境向け |
| `rocm` | `ROCMExecutionProvider` | AMD Radeon (Linux ROCm) | 要 ORT>=1.25 | 別ビルドの ORT-ROCm 導入環境向け |
| `openvino` | `OpenVINOExecutionProvider` (`GPU`) | Intel Arc / Intel 内蔵 GPU (Win / Linux x86_64) | 対応 (Plugin EP + ORT 1.31.1) | `auto` 時は GPU デバイス検出時のみ選択 |
| `webgpu` | `WebGpuExecutionProvider` | AMD Radeon / Intel Arc / NVIDIA / 他 (Win D3D12 / Linux Vulkan) | 対応 (Plugin EP + ORT 1.31.1) | 仮想/ソフトウェアアダプタ (`0x1414` / Mesa `llvmpipe` 等) は `auto` で自動除外、`--device webgpu` 強制時は警告付きで実行 |
| `dml` | `DmlExecutionProvider` | Windows DirectX 12 GPU (`onnxruntime-directml==1.24.4`) | 制限あり (ORT < 1.25 のため `DeformConv` 非対応) | バウンディングボックス破損防止のため `DEIMv2` は自動で `cpu` へフォールバックし、`PARSeq` のみ DML GPU で実行（`--strict-device` 時は `DEIMv2` で停止） |
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
- WSL2 上の Microsoft 仮想アダプタ (`vid=0x1414`) や Mesa `llvmpipe` (`vid=0x10005`) などのソフトウェア/仮想アダプタのみが検出された場合、`--device auto` はこれを自動スキップして `cpu`（または他の実ハードウェア EP）を選択します（`--device webgpu` を明示した場合のみ警告出力の上で実行）。
- Windows 環境では `onnxruntime-ep-webgpu==0.6.0` でシェーダコンパイルエラー（`CompileShaderDXC E_INVALIDARG`）が発生する場合があり、非 strict 時は自動で CPU へフォールバックします（Windows + NVIDIA GPU 環境では `cuda` または `dml` を推奨）。
- Intel Arc / Core Ultra GPU で OpenVINO を使う場合は最新の Intel GPU ドライバ（および Linux では Intel Compute Runtime / Level Zero）を導入してください。

### 2.2 NVIDIA CUDA 構成

`onnxruntime-gpu==1.31.1` を導入します（CUDA Toolkit / cuDNN ランタイム環境に応じたインデックスを選択）。

**CUDA 13 系環境（PyPI 既定）:**
```bash
pip uninstall -y onnxruntime
pip install -r requirements-cuda.txt
```
ホストにシステム CUDA 13 ランタイム / cuDNN 9 が未導入（Windows CUDA 12.8 のみや Colab 以外の素の venv 等）でも、以下の pip CUDA 13 ランタイム wheel を同じ仮想環境へ追加すれば、`ort_provider.py` が起動時に `ort.preload_dlls()` を自動呼び出しするため、`PATH` や `LD_LIBRARY_PATH` を手動変更せずに CUDA 13 が動作します。
```bash
pip install nvidia-cuda-runtime nvidia-cublas nvidia-cufft nvidia-curand nvidia-cusparse nvidia-cusolver nvidia-nvjitlink nvidia-cuda-nvrtc nvidia-cudnn-cu13
```

**CUDA 12 系環境（Google Colab 等を含む、`onnxruntime-cuda-12` Azure インデックスは `onnxruntime-gpu==1.29.0` 安定版まで提供、ORT>=1.25 で TableCenterNet 対応）:**
```bash
pip uninstall -y onnxruntime
pip install -r requirements-cuda.txt
pip install --force-reinstall --no-deps \
  --index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/onnxruntime-cuda-12/pypi/simple/ \
  onnxruntime-gpu==1.29.0
```
- **Windows CUDA 12 の補足**: システムに cuDNN 9 をインストールしても `%CUDNN_PATH%\bin\<CUDAバージョン>\x64`（例: `C:\Program Files\NVIDIA\CUDNN\v9.26\bin\12.9\x64`）が `PATH` に登録されていない環境では推論時に `cudnn64_9.dll` 読み込みエラーになるため、`PATH` へ追加するか pip で `nvidia-cudnn-cu12`（Linux/WSL2 でシステム CUDA 12 不足時は `nvidia-*-cu12` 一式）を導入してください。

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
   全モデル既定 **FP32** 精度で動作し、`CUDAExecutionProvider` には `"use_tf32": "0"` を自動指定して Ampere 世代以降（RTX 3070 等）での TF32 演算による微小差（1px 枠ずれ・読み順入替等）を抑止しているため、CPU `batch=1` と同一の認識結果（テキスト・座標・読み順）を出力します。

---

## 5. 検証済み環境・実測ベンチマーク実行手順

以下の環境および自動テストで動作と出力一致を検証済みです。

- **自動テスト (`pytest tests ndlocr-lite-gui/tests`)**:
  - **Linux / WSL2 Ubuntu (`Python 3.12.16`)**: **101 passed（+ 67 subtests）**
  - **Windows 11 (`Python 3.12.16`)**: **100 passed, 1 skipped（+ 63 subtests）**（Windows NTFS 使用不可文字を含む POSIX ファイル名テストをスキップ）
  - 公式 PARSeq 3モデル (`24x256-30` / `24x384-50` / `24x768-100`) の動的バッチパッチ (`_dynbatch_fp32_v1.onnx`) において、`batch=1` との厳密一致（`B=1` で誤差 `<= 1e-5`、動的バッチ時に `ORT_DISABLE_ALL <= 5e-3` / `ORT_ENABLE_ALL <= 2e-2` かつ `argmax` 完全一致）を検証。
  - バッチ推論 + 3段パイプライン並列実行時と CPU `batch=1` 逐次実行時で、サンプル資料の XML / JSON / TXT 出力が完全一致することを検証（縦中横バッチ処理・DEIM 分割前処理含む）。
  - ORT プロバイダの `use_tf32="0"` 設定、WebGPU ソフトウェア/仮想アダプタ除外・明示強制時の警告、DirectML での DEIM CPU フォールバック、および TableCenterNet の実 ORT バージョン表記エラーを単体テストで検証。
- **Plugin EP / DirectML 実機検証 (`requirements-gpu-generic.txt` / `onnxruntime-directml==1.24.4`)**:
  - 両プラグイン（ORT `1.31.1` + `onnxruntime-ep-webgpu==0.6.0` + `onnxruntime-ep-openvino==1.8.1`）が正常登録され、OpenVINO CPU-only 環境（Intel GPU 未検出時）で誤選択せず `--device auto` が正しく `webgpu`（または `cpu`）へフォールバック・優先選択することを確認。
  - Linux Vulkan GPU 上の `WebGpuExecutionProvider` で DEIMv2、動的バッチ PARSeq（`batch=4`, `CPU batch=1` と出力一致）、TableCenterNet v3（`DeformConv`）がいずれも CPU フォールバックなしで動作することを確認。
  - **WSL2 Ubuntu (`RTX 3070`)**: WSL2 の WebGPU Microsoft 仮想アダプタ（`vid=0x1414, id=142`）は `--device auto` で自動除外され `cpu` が選ばれること、また `--device webgpu` 明示時のみ警告表示の上で WebGPU 実行されることを確認済み。
  - **Windows 11 (`RTX 3070`)**: `--check-device` で GPU 検出 (`vid=0x10de`) を確認した一方、現行版 `onnxruntime-ep-webgpu==0.6.0` では DXC シェーダコンパイル時エラー（`CompileShaderDXC E_INVALIDARG`）が発生して非 strict 時は CPU へ自動フォールバックすることを確認（Windows + NVIDIA GPU では `cuda` または `dml` を推奨）。
  - **Windows 11 DirectML (`onnxruntime-directml==1.24.4`)**: `DEIMv2` は `--strict-device` 時に明示エラーで停止、非 strict 時は自動で `cpu` へフォールバックして `PARSeq` のみ `DmlExecutionProvider` で実行。5 画像（通常・`--enable-tcy`）および 3 ページ PDF E2E で CPU `batch=1` と差分 0 の完全一致を確認（`TableCenterNet` は `onnxruntime==1.24.4` 非対応のエラーメッセージを確認済み）。
- **デスクトップ NVIDIA GeForce RTX 3070 8GB (`AMD Ryzen 7 5800X`, Driver `617.42`, `Python 3.12.16`)**:
  - Windows 11・WSL2 Ubuntu 双方で CUDA 13 (`ORT 1.31.1` + pip CUDA 13 runtime wheel、`PATH` / `LD_LIBRARY_PATH` 変更なし) と CUDA 12 (`ORT 1.29.0`) を検証。
  - 同梱 `resource/` 5 画像（通常・`--enable-tcy`）・3 ページ PDF E2E の全構成で TXT / XML / JSON が CPU `batch=1` と **100% 完全一致（差分 0）**、PARSeq 3 モデル (`rec30`/`rec50`/`rec100`) 完全一致、`TableCenterNet v3` も CPU と同一 3 セル・座標差 `<= 6e-05` を確認。
  - 12 ページ連続処理の実測結果:

    | 環境 / バックエンド / モード | ORT 版 | 一括 12p (秒, 初期化込) | 一括 p/s | 定常 12p (秒, 初期化除) | 定常 p/s | 増分 VRAM (定常) |
    |---|---|---|---|---|---|---|
    | Windows 11 / CPU `batch=1` | `1.31.1` | 23.190 | 0.517 | 11.961 | 1.003 | — |
    | Windows 11 / CUDA 13 `batch=1` | `1.31.1` | 28.108 | 0.427 | 16.563 | 0.724 | +988 MiB |
    | Windows 11 / CUDA 13 動的 `batch=8` | `1.31.1` | 13.279 | 0.904 | 3.893 | 3.082 | +1,037 MiB |
    | Windows 11 / CUDA 13 動的 `batch=16` | `1.31.1` | 12.171 | 0.986 | 2.875 | 4.173 | +1,153 MiB |
    | Windows 11 / CUDA 13 動的 `batch=32` | `1.31.1` | **12.055** | **0.995** | **2.537** | **4.730** | +1,547 MiB |
    | Windows 11 / CUDA 12 動的 `batch=16` | `1.29.0` | 11.988 | 1.001 | 3.459 | 3.469 | +1,443 MiB |
    | Windows 11 / CUDA 12 動的 `batch=32` | `1.29.0` | 12.626 | 0.950 | 3.019 | 3.975 | +1,607 MiB |
    | Windows 11 / DML ハイブリッド `batch=16` (`DEIM CPU+PARSeq DML`) | `1.24.4` | 30.829 | 0.389 | 21.316 | 0.563 | +539 MiB |
    | Windows 11 / DML ハイブリッド `batch=32` (`DEIM CPU+PARSeq DML`) | `1.24.4` | 27.901 | 0.430 | 17.114 | 0.701 | +598 MiB |
    | WSL2 Ubuntu / CPU `batch=1` | `1.31.1` | 19.452 | 0.617 | 14.856 | 0.808 | — |
    | WSL2 Ubuntu / CUDA 13 `batch=1` | `1.31.1` | 21.085 | 0.569 | 14.058 | 0.854 | +931 MiB |
    | WSL2 Ubuntu / CUDA 13 動的 `batch=8` | `1.31.1` | 9.835 | 1.220 | 3.735 | 3.213 | +1,062 MiB |
    | WSL2 Ubuntu / CUDA 13 動的 `batch=16` | `1.31.1` | **8.946** | **1.341** | 2.943 | 4.077 | +1,331 MiB |
    | WSL2 Ubuntu / CUDA 13 動的 `batch=32` | `1.31.1` | 9.509 | 1.262 | **2.794** | **4.295** | +1,613 MiB |
    | WSL2 Ubuntu / CUDA 12 動的 `batch=16` | `1.29.0` | 10.345 | 1.160 | 3.084 | 3.891 | +1,158 MiB |
    | WSL2 Ubuntu / CUDA 12 動的 `batch=32` | `1.29.0` | **8.340** | **1.439** | 3.047 | 3.939 | +1,561 MiB |

- **Google Colab T4 Standard (`Tesla T4 15360 MiB`, NVIDIA driver `580.82.07`, `Python 3.13.16`, `onnxruntime-gpu==1.31.1`)**:
  - 動的バッチ版 PARSeq のセッション初期化で `allocation_planner` が展開済みデコーダの動的テンソル再利用計画に約 45 秒以上を費やしていたボトルネックを解消（`parseq.py` / `ort_provider.py` で `dynamic_batch=True` 時のみ `enable_mem_reuse=False`、CUDA に `cudnn_conv_algo_search="HEURISTIC"` を適用）。
  - 同梱 `resource/` の 5 画像（計 225 行）について、`--enable-tcy` の off / on 双方で CUDA 動的バッチ (`batch=32`) と `CPU batch=1` の TXT・XML `LINE STRING`・JSON (`contents`) が完全一致（差分 0）。12 ページ一括でも `CPU batch=1` / `CUDA batch=1` / `CUDA batch=8, 16, 32` 間で出力差分 0 を確認。
  - TableCenterNet v3（`DeformConv`）を CUDA で推論し、`CUDAExecutionProvider` 有効・CPU フォールバックなし・CPU と同一セル数（合成表 3 セル）を確認。
  - 12 ページ（同梱 `resource/` 画像の繰り返し）の一括処理 (`ocr.process`・モデル初期化込み) および独立プロセスでの定常推論（初期化・1 ページウォームアップ除外）・プロセス VRAM 実測:

    | モード | 一括 12p (秒, 初期化込) | 一括 pages/sec | 定常 12p (秒, 初期化除) | 定常 pages/sec | プロセス VRAM |
    |---|---|---|---|---|---|
    | CPU `batch=1` | 91.245 | 0.132 | 79.967 (`*`) | 0.150 | — |
    | CUDA `batch=1` | 23.683 | 0.507 | 12.598 | 0.953 | 882 MiB |
    | CUDA 動的バッチ `8` + 3段パイプライン | 14.671 | 0.818 | 4.294 | 2.795 | 1,008 MiB |
    | CUDA 動的バッチ `16` + 3段パイプライン | 14.477 | 0.829 | 3.966 | 3.026 | 1,142 MiB |
    | CUDA 動的バッチ `32` + 3段パイプライン | 14.595 | 0.822 | 3.876 | 3.096 | 1,524 MiB |

    `*` CPU `batch=1` の初期化除き時間は `ocr.process` 内 12 ページの処理時間合計（モデル初期化時間 11.369 秒、内訳 DET 2.083 秒 + PARSeq 3 モデル 9.287 秒は独立計測）。CUDA 動的バッチの初期化は `batch=8/16/32` で `9.838〜10.242` 秒（`batch=1` の `11.165` 秒より高速）。

    最適化前は T4 上で CUDA 動的バッチ `32` の一括 12 ページが 60.364 秒かかっていましたが、最適化後は **14.595 秒（4.14x 高速化・CPU 比 6.25x・CUDA `batch=1` 比 1.62x）** へ改善しました。初期化を除く定常 12 ページ推論では **`batch=32` が 3.876 秒（`3.096 pages/sec`、CUDA `batch=1` 比 3.25x・CPU 比約 20.6x）** に達します。また `--batch-size 16` は定常速度の約 98%（`3.026 pages/sec`）・12p 合計 `14.477` 秒を保ったまま VRAM を `382 MiB` 削減できるため、VRAM 容量に制約のある GPU（4 GB 級など）では `--batch-size 16` も有効です。

お使いの CUDA 環境または Google Colab で 12 ページ連続ベンチマークおよび 5 画像フル OCR 一致検査を実行する場合は、以下を実行してください:

```bash
python3 scripts/colab_verify_gpu.py --pages 12 --json-out colab_gpu_report.json
```
