import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

def evaluate_model(name, model, X_train, X_val, y_train, y_val):
    model.fit(X_train, y_train)

    # # log変換なし
    # train_pred = model.predict(X_train)
    # val_pred = model.predict(X_val)

    # train_mse = mean_squared_error(y_train, train_pred)
    # val_mse = mean_squared_error(y_val, val_pred)

    # log変換あり
    # 预测（log空间）
    y_train_pred_log = model.predict(X_train)
    y_val_pred_log = model.predict(X_val)

    # 还原为真实价格
    y_train_pred = np.expm1(y_train_pred_log)
    y_val_pred = np.expm1(y_val_pred_log)
    y_train_true = np.expm1(y_train)
    y_val_true = np.expm1(y_val)

    # MSE计算（原始空间）
    train_mse = mean_squared_error(y_train_true, y_train_pred)
    val_mse = mean_squared_error(y_val_true, y_val_pred)

    return {
        "Model": name,
        "Train MSE": round(train_mse, 2),
        "Validation MSE": round(val_mse, 2),
        "Overfit Gap": round(val_mse - train_mse, 2),
        "Overfit Gap Rate": round((val_mse - train_mse) / val_mse * 100, 2)
    }

# ========================
# GridSearchでのハイパーパラメータチューニング
# ========================
def tune_model(model, param_grid, X_train, y_train):
    grid_search = GridSearchCV(model, param_grid, cv=3, scoring='neg_mean_squared_error', verbose=1, n_jobs=-1)
    grid_search.fit(X_train, y_train)
    print("Best Params:", grid_search.best_params_)
    return grid_search.best_estimator_

