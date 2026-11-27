# NDLOCR-Liteのデスクトップアプリケーション

NDLOCR-Liteの使い方を説明します。

## 起動方法
[GitHub Releases ページ](https://github.com/melmaron/ndlocr-lite-gpu/releases) から、お使いのOSに対応する圧縮ファイル（Windows: `ndlocr_lite_gpu_v*_windows_cuda.zip` / `..._windows_gpu_generic.zip`、Linux: `..._linux_cuda.tar.gz` / `..._linux_gpu_generic.tar.gz`）をダウンロードし、日本語（全角文字）を含まないパスへ展開して実行してください（macOS 向け配布バイナリは非提供です）。

### 事前要件（配布バイナリ）
- **`cuda` 版（Windows / Linux）**: **CUDA 13 + cuDNN 9 ランタイムを同梱済み**（CUDA Toolkit の別途導入や `PATH` 設定は不要）です。必要なのは CUDA 13 対応の NVIDIA 公式ドライバ（R580 以降）のみです（同梱により容量は約 1.2〜1.6 GB になります）。
- **`gpu_generic` 版**: WebGPU / OpenVINO / TBB ランタイムを同梱済みです。Windows は DirectX 12 対応ドライバ（NVIDIA 環境では WebGPU のシェーダエラーで CPU へ落ちる場合があるため `cuda` 版を推奨）、Linux は Vulkan（`libvulkan.so.1`）+ GPU ドライバが必要です（OpenVINO は x86_64 のみ）。
- **Windows 版共通**: 不足しやすい Visual C++ ランタイム DLL（`msvcp140_1.dll`, `msvcp140_atomic_wait.dll` 等）も同梱済みです。
- **Linux 版共通**: GTK3 (`libgtk-3-0`) および glibc 2.35 以上（Ubuntu 22.04 以降目安）が必要です。
- **ログ・診断**: GPU が使えず CPU へフォールバックした場合は、画面上部の警告バナーと「環境診断」ダイアログ（対処手順・公式リンク・ログフォルダを開くボタン付き）で確認でき、詳細ログは `ndlocr-lite.log` に保存されます。

なお、Windows 11 ではセキュリティに関する警告が出ることがあります。その場合は「WindowsによってPCが保護されました」→「詳細情報」→「実行」を選んでください。

なお、初回の起動には1分程度時間を要することがあります。お待ちください。

## 操作方法
①OCR処理をかけたいファイルまたはディレクトリを指定します。画像（jpg,png,tiff,bmp,jp2）に加えてPDFファイルに対する処理が可能です。

②出力先となるディレクトリを指定します。

③「OCR」ボタンを押します。

④ ディレクトリを指定した場合、OCR処理が完了した画像から処理結果のプレビューを表示できます。ファイルを指定した場合には当該ファイルの処理結果が表示されます。

## 自分でアプリケーションをビルドする場合の方法（開発者向け情報）
本アプリケーションは[Flet（外部サイト）](https://flet.dev/)を利用します。

いずれのOSの場合にも事前にFlutter-SDK 3.47.9の導入が必要です（CI の [`build-flet-cross.yml`](../.github/workflows/build-flet-cross.yml) と同じバージョン）。依存関係のインストールに関する説明は省略します。

### Windowsの場合
https://flet.dev/docs/publish/windows/
も参照してください。
```
#(コマンドプロンプトを利用、ndlocr-lite-guiと同階層で実行する)
python3 -m venv ocrenv
.\ocrenv\Scripts\activate
pip install flet==1.1.2
xcopy ..\src .\src
flet build windows
```

### Macの場合（参考・配布バイナリ対象外／実機未検証）

```
#(ndlocr-lite-guiと同階層で実行する)
python3 -m venv ocrenv
source ./ocrenv/bin/activate
pip install flet==1.1.2
cp -r ../src .
flet build macos
```

### Linuxの場合
```
#(ndlocr-lite-guiと同階層で実行する)
python3 -m venv ocrenv
source ./ocrenv/bin/activate
pip install flet==1.1.2
cp -r ../src .
flet build linux
```
