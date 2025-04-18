import pandas as pd
import joblib
import numpy as np

# 訓練時に使用した特徴量の列順を読み込む
train_columns = np.load("model/train_columns.npy", allow_pickle=True)

test_data = pd.read_csv("data/processed/clean_test.csv")

# one-hot encoding
test_data_encoded = pd.get_dummies(test_data)

#訓練時に存在したがテストデータにない列を補完
missing_cols = set(train_columns) - set(test_data_encoded.columns)
for col in missing_cols:
    test_data_encoded[col] = 0

# 訓練用データと同じ列順に合せること
test_data_encoded = test_data_encoded[train_columns]

# モデルを読み込む
model = joblib.load("model/final_ridge_model.pkl") 

# 予測
pred_price_log = model.predict(test_data_encoded)
pred_price_real = np.expm1(pred_price_log)

# 結果を出力
result = pd.DataFrame({
    "ID": test_data["Id"],
    "SalePrice": pred_price_real
})

result.to_csv("result/result.csv", index=False)
print("resultが出力されました。")