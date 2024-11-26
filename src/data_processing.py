import pandas as pd

pd.set_option('display.max_rows',None)

train_data = pd.read_csv("data/raw/train.csv")
# print(train_data.head())

test_data = pd.read_csv("data/raw/test.csv")
# print(test_data.head())

# print(train_data.info())
# print(train_data.describe())
# print(train_data.isnull().sum())

missdataList = []
missing_data = train_data.isnull().sum()
for col, missdata in missing_data.items():
    if missdata > 0:
        missdataList.append(col)
        print(col)
missing_radio = train_data.isnull().sum()/len(train_data)
# print(missing_radio)

columns_to_drop = missing_radio[missing_radio>0.8].index

reduced_x = train_data.drop(columns=columns_to_drop, axis=1)
print(reduced_x)
# filtered = missing_data[missing_data > 1000]
# print(filtered)