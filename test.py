import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeRegressor

# # 加载数据集
# iris = load_iris()
# X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.3)

# print("依赖库已安装，数据集加载成功！")
# print(X_test)

train_data = pd.read_csv("data/raw/train.csv")
train_data = train_data.dropna(axis=1)

print(train_data.info)

y = train_data.SalePrice

X = train_data

test_model = DecisionTreeRegressor(random_state=1)
test_model.fit(X,y)

print(test_model.predict(X.head()))