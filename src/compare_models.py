import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso
from xgboost import XGBRegressor

from utils import evaluate_model
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
print("-----------------------------")
print("Sale Price Mean:", y.mean())

'''
三种模型结果比较
MSE
'''
results = []

# Linear Regression
results.append(evaluate_model(
    "LinearRegression",
    LinearRegression(),
    X_train, X_val, y_train, y_val
))

# Random Forest
results.append(evaluate_model(
    "RandomForest",
    RandomForestRegressor(random_state=42),
    X_train, X_val, y_train, y_val
))

# ------パーラメント最適化追加（4/10）--------
rf_param_grid = {
    "n_estimators": [100, 200],
    "max_depth": [None, 5, 20],
    "min_samples_split": [2, 5, 10]
}
best_rf = tune_model(
    RandomForestRegressor(random_state=42), 
    rf_param_grid, 
    X_train, 
    y_train
)
results.append(evaluate_model(
    "Tuned RandomForest",
    best_rf,
    X_train, X_val, y_train, y_val
))

# XGBoost
results.append(evaluate_model(
    "XGBoost",
    XGBRegressor(random_state=42, verbosity=0),
    X_train, X_val, y_train, y_val
))

# ------パーラメント最適化追加（4/10）--------
xgb_param_grid = {
    "n_estimators": [100, 200],            # 森林中树的数量
    "max_depth": [3, 6, 10],               # 每棵树的最大深度
    "learning_rate": [0.01, 0.1],          # 学习率（越小越稳越慢）
    "subsample": [0.8, 1.0],               # 每棵树的训练样本比例（防止过拟合）
    "colsample_bytree": [0.8, 1.0],        # 每棵树用的特征比例
    "reg_alpha": [0, 1],                   # L1正则化强度
    "reg_lambda": [1, 10]                  # L2正则化强度
}
best_xgb = tune_model(
    XGBRegressor(random_state=42, verbosity=0),
    xgb_param_grid,
    X_train, 
    y_train
)
results.append(evaluate_model(
    "Tuned XGBoost",
    best_xgb,
    X_train, X_val, y_train, y_val
))

# Output result comparison
result_df = pd.DataFrame(results)
result_df = result_df.sort_values(by="Validation MSE")

print("-----------------------------Model Comparison")
print("\n📊 Model Comparison Result:")
print(result_df.to_string(index=False))
