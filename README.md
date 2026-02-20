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

## Project Structure

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
