import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso
from xgboost import XGBRegressor
import matplotlib.pyplot as plt


from utils import tune_model

data = pd.read_csv("data/processed/clean_train.csv")

'''
线性回归(Linear Regression)
MAE
'''
# 
y = data['SalePrice']
X = data.drop('SalePrice', axis = 1)

X_encoded = pd.get_dummies(X)

# log変換処理追加
y_log = np.log1p(y)

# log変換抜け
# X_train,X_val,y_train,y_val = train_test_split(X_encoded,y,test_size=0.2,random_state=42)
# log変換あり、y_logを用いてデータを分割
X_train,X_val,y_train,y_val = train_test_split(X_encoded,y_log,test_size=0.2,random_state=42)
print("-----------------------------Linear")
print("Sale Price Mean:", y.mean())

#
model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_val)
MAE = mean_absolute_error(y_val, y_pred)
print("Validation MAE:", MAE)

train_pred = model.predict(X_train)
train_MAE = mean_absolute_error(y_train, train_pred)

print("Train MAE: ",train_MAE)
print("Overfit Gap Rate:",(MAE - train_MAE) / MAE * 100)

y_pred_real = np.expm1(y_pred)
y_val_real = np.expm1(y_val)
MAE_real = mean_absolute_error(y_val_real, y_pred_real)
print("MAE after log inverse (真实房价下):", round(MAE_real, 2))

print("-----------------------------")

'''
Ridgeを試す
2025/4/11
'''
# LinearRegressionをRidgeに切り替え
ridge_model = Ridge(alpha=0.1)
ridge_model.fit(X_train,y_train)

y_pred_ridge = ridge_model.predict(X_val)

# 还原为真实价格
y_pred_real_ridge = np.expm1(y_pred_ridge)
# y_val_real_ridge = np.expm1(y_val)

MAE_ridge = mean_absolute_error(y_val_real, y_pred_real_ridge)
MSE_ridge = mean_squared_error(y_val_real, y_pred_real_ridge)

print("-----------------------------Ridge")
print("✅ Ridge Regression 结果:")
print("MAE_RIDGE (真实价格):", round(MAE_ridge, 2))
print("MSE_RIDGE (真实价格):", round(MSE_ridge, 2))
print("-----------------------------")

'''
param最適化
'''
ridge_param_grid = {"alpha":[0.01,0.1,1.0,10.0,20.0,50.0]}

best_ridge = tune_model(
    Ridge(),
    ridge_param_grid,
    X_train,
    y_train
)
y_pred_ridge_best = best_ridge.predict(X_val)

y_pred_real_ridge_best = np.expm1(y_pred_ridge_best)
# y_val_real_ridge_best = np.expm1(y_val)

MAE_ridge_best = mean_absolute_error(y_val_real,y_pred_real_ridge_best)
MSE_ridge_best = mean_squared_error(y_val_real,y_pred_real_ridge_best)

print("Ridge Regression (Best Alpha)")
print("MAE_RIDGE_BEST:", round(MAE_ridge_best, 2))
print("MSE_RIDGE_BEST:", round(MSE_ridge_best, 2))
print("-----------------------------")


# ---------- Lasso ----------
lasso_model = Lasso(max_iter=10000)
lasso_model.fit(X_train, y_train)

y_pred_lasso = lasso_model.predict(X_val)

# 还原为真实价格
y_pred_real_lasso = np.expm1(y_pred_lasso)
# y_val_real_lasso = np.expm1(y_val)

MAE_lasso = mean_absolute_error(y_val_real,y_pred_real_lasso)
MSE_lasso = mean_squared_error(y_val_real,y_pred_real_lasso)

print("-----------------------------Lasso")
print("✅ Lasso 结果:")
print("MAE_LASSO (真实价格):", round(MAE_lasso, 2))
print("MSE_LASSO (真实价格):", round(MSE_lasso, 2))
print("-----------------------------")

lasso_param_grid = {
    "alpha": [0.001, 0.01, 0.1, 1.0, 10.0]
}
best_lasso = tune_model(
    Lasso(max_iter=10000),  # 防止收敛问题
    lasso_param_grid,
    X_train,
    y_train 
)
y_pred_lasso_best = best_lasso.predict(X_val)

y_pred_real_lasso_best = np.expm1(y_pred_lasso_best)

MAE_lasso_best = mean_absolute_error(y_val_real,y_pred_lasso_best)
MSE_lasso_best = mean_squared_error(y_val_real,y_pred_lasso_best)

print("Lasso (Best Alpha)")
print("MAE_LASSO_BEST:", round(MAE_lasso_best, 2))
print("MSE_LASSO_BEST:", round(MSE_lasso_best, 2))

coef = best_lasso.coef_
nonzero_idx = np.where(coef != 0)[0]
selected_features = X_train.columns[nonzero_idx]

print(f"📌 选中的特征数量: {len(selected_features)} / {len(coef)}")
print("🎯 被保留下来的特征（部分）：")
print(selected_features[:20])  
print("-----------------------------")

# ---------- モデルごとのMAE（平均絶対誤差）を可視化 ----------
# 各モデル（Linear, Ridge, Lassoなど）の予測誤差（MAE）を比較し、
# どのモデルが最も安定しているか、過学習していないかを直感的に確認する。
# 赤い破線は最小のMAEライン（最も良いモデル）を示す。
model_names = ['Linear', 'Ridge', 'Ridge(Tuned)', 'Lasso', 'Lasso(Tuned)']
maes = [MAE_real, MAE_ridge, MAE_ridge_best, MAE_lasso, MAE_lasso_best]

plt.figure(figsize=(10, 6))
plt.bar(model_names, maes, color='skyblue')
plt.ylabel("MAE(real)")
plt.title("MAE Comparison Across Models")
plt.axhline(y=min(maes), color='red', linestyle='--', label='Minimum MAE')
plt.legend()
plt.tight_layout()
plt.show()