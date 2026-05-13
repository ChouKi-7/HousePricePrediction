import numpy as np
import joblib
from model_training import ridge_model,X_encoded

'''
通过对比结果得知ridge_model为最优模型
保存该模型
'''
# 確定されたモデルを保存する
joblib.dump(ridge_model, "model/final_ridge_model.pkl")
# 訓練時に使用した特徴量の列順を保存
np.save("model/train_columns.npy", X_encoded.columns)
print("✅ final_ridge_model.pkl and train_columns.npy saved。")

#cols = np.load("model/train_columns.npy", allow_pickle=True)
#print(cols)
#print(len(cols))