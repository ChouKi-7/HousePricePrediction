import pandas as pd

pd.set_option('display.max_rows',None)

train_data = pd.read_csv("data/raw/train.csv")
# print(train_data.head())

test_data = pd.read_csv("data/raw/test.csv")
# print(test_data.head())

# print(train_data.info())
# print(train_data.describe())
print(train_data.isnull().sum())