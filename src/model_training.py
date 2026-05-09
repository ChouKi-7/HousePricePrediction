import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.linear_model import Lasso
import matplotlib.pyplot as plt
import seaborn as sns
import joblib


from utils import tune_model

data = pd.read_csv("data/processed/clean_train.csv")

# 
y = data['SalePrice']
X = data.drop('SalePrice', axis = 1)

print("encoding前:*********************")
print(X[['MSZoning', 'Street']].head(5))
X_encoded = pd.get_dummies(X)

print("encoding後:*********************")
print(X_encoded.filter(like='MSZoning').head(5))

print("*********************")
print("encoding前列数:", X.shape[1])
print("encoding後列数:", X_encoded.shape[1])

# log変換処理追加
y_log = np.log1p(y)

# log変換抜け
# X_train,X_val,y_train,y_val = train_test_split(X_encoded,y,test_size=0.2,random_state=42)
# log変換あり、y_logを用いてデータを分割
X_train,X_val,y_train,y_val = train_test_split(X_encoded,y_log,test_size=0.2,random_state=42)
print("-----------------------------Linear")
print("Sale Price Mean:", y.mean())

'''
Linear Regression
'''
model = LinearRegression()
model.fit(X_train, y_train)

print("*********************")
coef = pd.Series(model.coef_, index=X_encoded.columns)
print("係数が大きい上位10列:")
print(coef.abs().sort_values(ascending=False).head(10))
print("係数がほぼゼロの列数:", (coef.abs() < 0.001).sum())

y_pred = model.predict(X_val)
MAE_linear = mean_absolute_error(y_val, y_pred)
print("Validation MAE:", MAE_linear)

train_pred = model.predict(X_train)
train_MAE = mean_absolute_error(y_train, train_pred)

print("Train MAE: ",train_MAE)
print("Overfit Gap Rate:",(MAE_linear - train_MAE) / MAE_linear * 100)

y_pred_real = np.expm1(y_pred)
y_val_real = np.expm1(y_val)
MAE_real = mean_absolute_error(y_val_real, y_pred_real)
print("MAE after log inverse (真实房价下):", round(MAE_real, 2))

print("-----------------------------")

'''
Ridgeを試す
'''
# ---------- Ridge(L2) ----------
# LinearRegressionをRidgeに切り替え
ridge_model = Ridge(alpha=0.1)
ridge_model.fit(X_train,y_train)

y_pred_ridge = ridge_model.predict(X_val)
MAE_ridge = mean_absolute_error(y_val, y_pred_ridge)

train_pred_ridge = ridge_model.predict(X_train)
train_MAE_ridge = mean_absolute_error(y_train, train_pred_ridge)

# 还原为真实价格
y_pred_real_ridge = np.expm1(y_pred_ridge)
# y_val_real_ridge = np.expm1(y_val)

MAE_ridge_real = mean_absolute_error(y_val_real, y_pred_real_ridge)
MSE_ridge_real = mean_squared_error(y_val_real, y_pred_real_ridge)


print("-----------------------------Ridge")
print("✅ Ridge Regression Result:")
print("Overfit Gap Rate:",(MAE_ridge - train_MAE_ridge) / MAE_ridge * 100)
print("MAE_RIDGE (log変換あり):", round(MAE_ridge, 2))
print("MAE_RIDGE (real):", round(MAE_ridge_real, 2))
print("MSE_RIDGE (real):", round(MSE_ridge_real, 2))
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


# ---------- Lasso(L1) ----------
lasso_model = Lasso(max_iter=10000)
lasso_model.fit(X_train, y_train)

y_pred_lasso = lasso_model.predict(X_val)
MAE_lasso = mean_absolute_error(y_val, y_pred_lasso)

train_pred_lasso = lasso_model.predict(X_train)
train_MAE_lasso = mean_absolute_error(y_train, train_pred_lasso)

# 还原为真实价格
y_pred_real_lasso = np.expm1(y_pred_lasso)
# y_val_real_lasso = np.expm1(y_val)

MAE_lasso_real = mean_absolute_error(y_val_real,y_pred_real_lasso)
MSE_lasso_real = mean_squared_error(y_val_real,y_pred_real_lasso)

print("-----------------------------Lasso")
print("✅ Lasso Result:")
print("MAE_LASSO (log変換あり):", round(MAE_lasso, 2))
print("Overfit Gap Rate:",(MAE_lasso - train_MAE_lasso) / MAE_lasso * 100)
print("MAE_LASSO (real):", round(MAE_lasso_real, 2))
print("MSE_LASSO (real):", round(MSE_lasso_real, 2))
print("-----------------------------")

lasso_param_grid = {
    "alpha": [0.001, 0.01, 0.1, 1.0, 10.0]
}
best_lasso = tune_model(
    Lasso(max_iter=10000), 
    lasso_param_grid,
    X_train,
    y_train 
)

print("*********************")
coef_lasso = pd.Series(best_lasso.coef_, index=X_encoded.columns)
print("Lassoで係数がゼロになった列数:", (coef_lasso == 0).sum())
print("Lassoで残った列数:", (coef_lasso != 0).sum())
print("\n残った上位10列:")
print(coef_lasso.abs().sort_values(ascending=False).head(10))
print("*********************")

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

'''
Lasso モデルで選ばれた特徴量同士の相関関係をヒートマップ(Heatmap)で可視化
'''
# # 提取 Lasso 选中的特征列
# lasso_selected_data = X_train[selected_features]

# # 计算相关性矩阵
# corr = lasso_selected_data.corr()

# # 可视化热力图
# plt.figure(figsize=(12, 10))
# sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
# plt.title("Heatmap of Lasso-Selected Features", fontsize=16)
# plt.xticks(rotation=45)
# plt.yticks(rotation=0)
# plt.tight_layout()
# plt.show()

# ---------- モデルごとのMAE（平均絶対誤差）を可視化 ----------
# 各モデル（Linear, Ridge, Lassoなど）の予測誤差（MAE）を比較し、
# どのモデルが最も安定しているか、過学習していないかを直感的に確認する。
# 赤い破線は最小のMAEライン（最も良いモデル）を示す。
model_names = ['Linear', 'Ridge', 'Ridge(Tuned)', 'Lasso', 'Lasso(Tuned)']
maes = [MAE_real, MAE_ridge_real, MAE_ridge_best, MAE_lasso_real, MAE_lasso_best]

plt.figure(figsize=(10, 6))
plt.bar(model_names, maes, color='skyblue')
plt.ylabel("MAE(real)")
plt.title("MAE Comparison Across Models")
plt.axhline(y=min(maes), color='red', linestyle='--', label='Minimum MAE')
plt.legend()
plt.tight_layout()
plt.show()

'''
通过对比结果得知ridge_model为最优模型
保存该模型
'''
# # 確定されたモデルを保存する
# joblib.dump(ridge_model, "model/final_ridge_model.pkl")
# # 訓練時に使用した特徴量の列順を保存
# np.save("model/train_columns.npy", X_encoded.columns)
# print("✅ 模型和特征列信息已保存。")
