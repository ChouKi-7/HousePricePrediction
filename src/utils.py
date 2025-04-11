import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

def handle_missing_values(df, drop_thresh=0.8, verbose=True):
    """
    Handle missing values:
    1. Drop columns with missing rate higher than drop_thresh (default: 0.8)
    2. Fill missing values in numerical columns with the median
    3. Fill missing values in categorical columns with 'Missing'
    4. Return the cleaned DataFrame
    """
    df_copy = df.copy()  # for protect original DataFrame
    rows = len(df_copy)

    # Step 1: Drop columns with high missing rate
    missing_ratio = df_copy.isnull().sum() / rows
    columns_to_drop = missing_ratio[missing_ratio > drop_thresh].index
    if verbose:
        print(f"🗑️ Dropping columns with missing rate > {drop_thresh*100:.0f}%:")
        print(list(columns_to_drop))
    df_copy.drop(columns=columns_to_drop, axis=1, inplace=True)

    # Step 2: Fill missing values in numerical columns with median
    for col in df_copy.select_dtypes(include=['float64', 'int64']):
        if df_copy[col].isnull().sum() > 0:
            df_copy[col] = df_copy[col].fillna(df_copy[col].median())
    
    # Step 3: Fill missing values in categorical columns with 'Missing'
    for col in df_copy.select_dtypes(include=['object']):
        if df_copy[col].isnull().sum() > 0:
            df_copy[col] = df_copy[col].fillna('Missing')

    # Step 4: Report remaining missing values
    total_missing = df_copy.isnull().sum().sum()
    if verbose:
        print(f"\n✅ Missing value handling completed. Total remaining missing values: {total_missing}")

    return df_copy

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

