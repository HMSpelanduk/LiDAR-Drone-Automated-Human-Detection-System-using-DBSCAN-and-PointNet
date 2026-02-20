# LiDARデータを用いた自動人物（マネキン）検出システム

本リポジトリは、LiDAR点群データを用いた **マネキン（人物の代替対象）検出システム** のシンプルなパイプラインを実装したものです。
本システムでは以下の手法を使用しています：
- **DBSCANクラスタリング**（物体セグメンテーション）
- **TinyPointNet**（軽量版PointNet風ニューラルネットワーク分類器）
- **Open3D 可視化**（クラスタ表示 + 3Dバウンディングボックス描画）

---

## 必要環境（事前にインストール）

```bash
pip install numpy
pip install open3d
pip install scikit-learn
pip install matplotlib
pip install torch torchvision torchaudio
```

## プロジェクト構成

```bash
LiDAR-Drone-Automated-Human-Detection-System-using-DBSCAN-and-PointNet/
│
├── .venv/                      # Python仮想環境（gitでは無視）
│
├── data/                       # 生の点群データ（gitでは無視）
│   ├── mannequin/              # マネキンのPLYサンプル
│   ├── background/             # 背景物体のPLYサンプル
│   └── test data/              # テスト用フルシーン（※フォルダ名にスペースあり）
│
├── data_npy/                   # .npy形式に変換済みデータセット（gitでは無視）
│   ├── mannequin/              # マネキンサンプル（Nx3のnumpy配列）
│   └── background/             # 背景サンプル（Nx3のnumpy配列）
│
├── clusters_data/              # フルシーンに対するDBSCAN出力クラスタ（gitでは無視）
│   └── cluster_*.ply
│
├── scratch code/               # 旧スクリプト・試験用コード（参考用）
│
├── .gitignore                  # Git除外設定（データ・モデル・IDEファイルなど）
│
├── tinypointnet2_model.py      # TinyPointNetニューラルネットワーク定義
├── prepare_dataset.py          # PLY → NPY変換およびデータセットローダー
├── train_pointnet.py           # TinyPointNet分類器の学習スクリプト
├── check_model.py              # 学習済みモデルの動作確認用スクリプト
├── create_cluster_drone.py     # フルシーン点群に対するDBSCANクラスタリング
├── test_better_pointnet.py     # 検出および可視化を行うメインパイプライン(最も確率の高いマネキンを1体のみ検出)
├── test_multiple_pointnet.py   # データ内の複数のマネキンを検出するために使用
│
└── pointnet_mannequin_classifier.pth
                                # 学習済みモデル重み（gitでは無視）

```
## 学習から検出までの処理フロー

```bash
PLY取得
   ↓
   LiDARセンサから取得したフルシーンの点群データ（.ply形式）を準備する。

DBSCANクラスタリング (create_cluster_drone.ply)
   ↓
   create_cluster_drone.plyを使用し、点群データにDBSCANを適用し、シーン内の物体ごとにクラスタを分割する。
   地面・背景・マネキンなどが個別クラスタとして抽出される。

マネキン / 背景を手動分類
   ↓
   抽出されたクラスタの中からマネキンと背景物体を選択し、
   data/mannequin/ と data/background/ に保存する。
   この作業を繰り返し、十分なデータ数（例：各50件）を収集する。

prepare_dataset.py
   ↓
   .plyデータを正規化済みの .npy 形式へ変換する。
   中心化・スケーリング・点数統一を行い、学習用データセットを作成する。

train_pointnet.py
   ↓
   TinyPointNetモデルを用いて .npy データを学習する。
   背景（label 0）とマネキン（label 1）を分類できるように訓練する。

pointnet_mannequin_classifier.pth 生成
   ↓
   学習完了後、人物検出用の学習済みモデルが生成される。
   これが本システムの検出モデルとなる。

check_model.py（任意：動作確認）
   ↓
   学習済みモデルがデータを正しく分類できているかを簡易的に確認する。
   data_npy/background と data_npy/mannequin を入力し、予測確率と予測クラスを表示する。

test_*.py で新規データ検出
   ↓
   学習済みモデルを用いて新しいLiDARデータを検出する。
   ・test_better_pointnet.py → 最も確率の高いマネキン1体のみ検出
   ・test_multiple_pointnet.py → 複数のマネキンを同時に検出

※ テストを行う場合は .ply ファイルを data/test data/ に配置し、
   Pythonコード内でファイル名を指定する。

```
## 検出結果例
![Image alt](https://github.com/HMSpelanduk/LiDAR-Drone-Automated-Human-Detection-System-using-DBSCAN-and-PointNet/blob/638ffe2beac7cfd3d01342af49a3a561c9a32164/result_screenshot.png)

左：LiDARセンサから取得した元の点群データ。

右：学習済みTinyPointNetモデルによる検出結果（マネキンを赤色で表示）。
