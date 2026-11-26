# NDLOCR-Liteのデスクトップアプリケーション

NDLOCR-Liteの使い方を説明します。

## 起動方法
[GitHub Releases ページ](https://github.com/melmaron/ndlocr-lite/releases) から、お使いのOSに対応する圧縮ファイル（Windows: `ndlocr_lite_v*_windows_cuda.zip` / `..._windows_gpu_generic.zip`、Linux: `..._linux_cuda.tar.gz` / `..._linux_gpu_generic.tar.gz`、macOS: `..._macos_apple_silicon.tar.gz` / `..._macos_intel.tar.gz`）をダウンロードし、展開して `ndlocr_lite` をダブルクリック等で実行してください。

なお、セキュリティに関する警告が出ることがあります。

Windows 11の場合は、「WindowsによってPCが保護されました」→「詳細情報」→「実行」を選んでください。

macOSの場合は、

https://zenn.dev/nakamura196/articles/c62a465537ff20

の手順に従ってください。

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

### Macの場合

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
