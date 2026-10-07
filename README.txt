ファイル構成
sotuken
|-input100(入力画像ファイル)
|-requirements.txt
|-zikken1.ipynb(jupyter版)
|-zikken1.py(python版)

jupyter版は内部がどのようになっているかの精査用
python版は一度に多くの画像を変換する用(1度に100枚までは動作確認済です)

推奨実行環境
・jupyter版
プロセッサ	Intel(R) Core(TM) i7-10700F CPU @ 2.90GHz (2.90 GHz)
実装 RAM	32.0 GB
グラフィックス カード	NVIDIA GeForce RTX 3070 (8 GB)
・python版
プロセッサ	インテル Core Ultra 9 プロセッサー 285K (最⼤5.70GHz / 24コア・24スレッド / 36MB)
実装 RAM	Kingston FURY 64GB (16GB×4) DDR5-4800 MT/sXMP対応 RGB (最⼤128G)
グラフィックス カード	NVIDIA GeForce RTX 5090

導入手順
1. 前提条件
Python: python >= 3.10  
PyTorch: torch >= 2.5.1, torchvision >= 0.20.1 （および対応するCUDA環境を推奨）  
OS: Linux, macOS, または Windows（Windowsの場合は WSL (Windows Subsystem for Linux) + Ubuntu の利用）

2. SAM2導入手順
上から順に実行してください
・ファイル作成
・SAM2の導入
以下のコマンドを実行し、Metaの公式リポジトリを sam2 という名前でクローン化します。
これにより、ファイル構成に合わせた <作成ファイル>/sam2/ が作成されます。

git clone https://github.com/facebookresearch/segment-anything-2.git sam2

続いて次のコードで必要なリポジトリのインストールを行う

cd sam2
pip install -e .

事前学習済みモデル（チェックポイント）のダウンロード

cd checkpoints
./download_ckpts.sh
cd ..

・SAM2の動作確認
sam2/notebooks内にあるjupyterコードがそれぞれ動けばSAM2のセットアップ完了です。

3. 実験環境の導入手順
・<作成ファイル>/sam2/にsotukenファイルをコピーする。
・以下のコードを実行して一括で環境をインストールする。

cd sotuken
pip install -r requirements.txt

4. 全体の動作確認
・zikken1.ipynbを上から順に実行し結果を確認する。

