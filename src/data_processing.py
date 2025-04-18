# import pandas as pd

# pd.set_option('display.max_rows',None)

# # read data
# train_data = pd.read_csv("data/raw/train.csv")
# # print(train_data.head())

# test_data = pd.read_csv("data/raw/test.csv")
# # print(test_data.head())

# # print(train_data.info())
# # print(train_data.describe())
# # print(train_data.isnull().sum())

# # find missing_data
# missdataList = []
# missing_data = train_data.isnull().sum()
# for col, missdata in missing_data.items():
#     if missdata > 0:
#         missdataList.append(col)
#         print(col)
# missing_radio = train_data.isnull().sum()/len(train_data)
# # print(missing_radio)

# # delete column miss data >80%
# columns_to_drop = missing_radio[missing_radio>0.8].index

# reduced_x = train_data.drop(columns=columns_to_drop, axis=1)
# print(reduced_x)
# # filtered = missing_data[missing_data > 1000]
# # print(filtered)

# # 1. 遍历数值型列
# for col in reduced_x.select_dtypes(include=['float64', 'int64']):
#     # 2. 如果这一列有缺失us
#     if reduced_x[col].isnull().sum() > 0:
#         # 3. 用该列的中位数填补所有缺失值
#         reduced_x[col] = reduced_x[col].fillna(reduced_x[col].median())

# # 1. 遍历类别型列
# for col in reduced_x.select_dtypes(include=['object']):
#     # 2. 如果这一列有缺失值
#     if reduced_x[col].isnull().sum() > 0:
#         # 3. 用字符串 'Missing' 填补缺失值
#         reduced_x[col] = reduced_x[col].fillna('Missing')

# reduced_x.to_csv("data/processed/clean_train.csv", index=False)

import pandas as pd
from utils import handle_missing_values 

train_data = pd.read_csv("data/raw/train.csv")
# drop columns with high missing rate + fill other missing values
cleaned_data = handle_missing_values(train_data, drop_thresh=0.8)
# save
cleaned_data.to_csv("data/processed/clean_train.csv", index=False)

test_data = pd.read_csv("data/raw/test.csv")
clean_test = handle_missing_values(test_data, drop_thresh=0.8)
clean_test.to_csv("data/processed/clean_test.csv", index=False)