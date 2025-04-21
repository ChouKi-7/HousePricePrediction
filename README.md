# 🏠 住宅価格予測プロジェクト（House Price Prediction）※学習目的の実装（for practice）
Kaggle の「House Prices - Advanced Regression Techniques」コンペを題材にした、回帰モデルによる住宅価格予測プロジェクトです。
特徴量の前処理から複数モデルの比較、チューニング、予測まで一連の流れを実装しました。
---

## ディレクトリ構成:
firstPython/
├── data/                         # データフォルダ
│   ├── raw/                      # 元データ（train.csv / test.csv）
│   └── p
rocessed/               # 前処理後のデータ（clean_train.csv / clean_test.csv）
│
├── model/                       # モデルや特徴量ファイル保存用
│   ├── final_ridge_model.pkl    # 学習済みモデル（Ridge）
│   └── train_columns.npy        # 学習に使った特徴量列順
│
├── result/                      # 提出用ファイルなど
│   └── result.csv               # テストデータ予測結果（Kaggle提出形式）
│
├── src/                         # ソースコード
│   ├── data_processing.py       # 欠損処理など前処理関数
│   ├── model_training.py        # モデル学習・比較・可視化
│   ├── evaluation.py            # テストデータ評価＆submission出力
│   ├── compare_models.py        # モデル評価・比較関数
│   └── utils.py                 # ハイパーパラメータチューニングなどユーティリティ関数
│
├── README.md                    # プロジェクト説明（このファイル）
└── venv/                        # 仮想環境（Git管理外）

---

## 実装内容

### データ前処理
- 欠損値の列除去（80%以上欠損）
- 数値：中央値で補完
- カテゴリ：`"Missing"` で補完
- One-hot encoding によるカテゴリ変換

### 特徴量変換
- `SalePrice` に `np.log1p()` を適用（外れ値の影響を軽減）

### モデル構築・比較

- `LinearRegression`, `Ridge`, `Lasso` をそれぞれ実装し、基本性能を評価  
- `RandomForest`, `XGBoost` モデルも追加し、非線形モデルとの比較を実施  
- `GridSearchCV` によるハイパーパラメータチューニングを実施  
  - `Ridge`, `Lasso`, `RandomForest`, `XGBoost` に適用  
- `Lasso` による自動特徴選択（非ゼロ特徴数を確認）  
- 目的変数 `SalePrice` に対して `log1p()` 変換を適用し、予測後に `expm1()` により元のスケールに復元して評価  
- 各モデルについて `Train MSE`, `Validation MSE`, `Overfit Gap` を比較し、性能を表形式で可視化  

> **結果として、Ridge モデルが最も安定した性能を示し、最終モデルとして採用しました。**  
> 各モデルの評価指標や比較過程の詳細は `計画とまとめ.txt``計画とまとめ_日本語ver.md` に記録しています。より詳しく知りたい方はそちらをご覧ください。

### モデル評価
- MAE, MSE を評価指標として採用
- `Overfit Gap`（過学習度）を定量評価
- モデルごとの MAE 比較バーグラフを可視化

### モデル保存・再利用
- 最終モデル（Ridge）を `.pkl` で保存
- 特徴量の列順も `.npy` で保存し、再予測時に復元

### 予測・提出ファイル作成
- test データを訓練と同じ形式に整形
- log 予測 → `np.expm1()` で元スケールに戻す
- Kaggle 提出形式の `result.csv` を生成

---

## 結果分析（MAEベース）

最も精度の良かったモデルは **Ridge回帰（alpha=0.1）** で、log変換 + 適度な正則化により過学習を抑えながら安定した予測性能を得られました。

| モデル           | MAE（実価格） |
|------------------|----------------|
| Linear           | 約15030.64円   |
| Ridge            | 約15001.43円   |
| Ridge (Tuned)    | 約16588.35円   |
| Lasso            | 約22735.69円   |
| Lasso (Tuned)    | 精度悪化（過正則化）

---

## 今後の改善アイデア

- `SHAP` によるモデル解釈
- 豪邸・一般住宅に分けたセグメント別モデル構築
- 特徴量重要度のランキング出力
- XGBoost / LightGBM モデル追加

---

## 👤 作成者

ki（GitHub: [ChouKi-7]）  
2025年4月