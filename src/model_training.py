import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

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
real_mae = mean_absolute_error(y_val_real, y_pred_real)
print("MAE after log inverse (真实房价下):", round(real_mae, 2))

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
y_val_real_ridge = np.expm1(y_val)

MAE_ridge = mean_absolute_error(y_val_real_ridge, y_pred_real_ridge)
MSE_ridge = mean_squared_error(y_val_real_ridge, y_pred_real_ridge)

print("✅ Ridge Regression 结果:")
print("MAE (真实价格):", round(MAE_ridge, 2))
print("MSE (真实价格):", round(MSE_ridge, 2))

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
y_val_real_ridge_best = np.expm1(y_val)

MAE_ridge_best = mean_absolute_error(y_val_real_ridge_best,y_pred_real_ridge)
MSE_ridge_best = mean_squared_error(y_val_real_ridge_best,y_pred_real_ridge_best)

print("-----------------------------")
print("Ridge Regression (Best Alpha)")
print("MAE_BEST:", round(MAE_ridge_best, 2))
print("MSE_BEST:", round(MSE_ridge_best, 2))