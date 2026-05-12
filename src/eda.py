import pandas as pd

df = pd.read_csv("data/raw/train.csv")

# ===== 1. Missing rate per column =====
print("=== Missing rate (only columns with missing values) ===")
missing_ratio = df.isnull().sum() / len(df)
missing_ratio = missing_ratio[missing_ratio > 0].sort_values(ascending=False)
print(missing_ratio.to_string())

# ===== 2. Zero ratio in numeric columns =====
print("\n=== Zero ratio in numeric columns (over 50%) ===")
num_cols = df.select_dtypes(include='number').columns
zero_ratio = (df[num_cols] == 0).sum() / len(df)
print(zero_ratio[zero_ratio > 0.5].sort_values(ascending=False).to_string())

# ===== 3. Correlation with SalePrice =====
print("\n=== Correlation with SalePrice (top 15) ===")
corr = df[num_cols].corr()['SalePrice'].drop('SalePrice')
print(corr.abs().sort_values(ascending=False).head(15).to_string())


# ----result-----
'''
=== Missing rate (only columns with missing values) ===
PoolQC          0.995205
MiscFeature     0.963014
Alley           0.937671
Fence           0.807534
MasVnrType      0.597260
FireplaceQu     0.472603
LotFrontage     0.177397
GarageType      0.055479
GarageYrBlt     0.055479
GarageFinish    0.055479
GarageQual      0.055479
GarageCond      0.055479
BsmtFinType2    0.026027
BsmtExposure    0.026027
BsmtFinType1    0.025342
BsmtCond        0.025342
BsmtQual        0.025342
MasVnrArea      0.005479
Electrical      0.000685

=== Zero ratio in numeric columns (over 50%) ===
PoolArea         0.995205
3SsnPorch        0.983562
LowQualFinSF     0.982192
MiscVal          0.964384
BsmtHalfBath     0.943836
ScreenPorch      0.920548
BsmtFinSF2       0.885616
EnclosedPorch    0.857534
HalfBath         0.625342
MasVnrArea       0.589726
BsmtFullBath     0.586301
2ndFlrSF         0.567808
WoodDeckSF       0.521233

=== Correlation with SalePrice (top 15) ===
OverallQual     0.790982
GrLivArea       0.708624
GarageCars      0.640409
GarageArea      0.623431
TotalBsmtSF     0.613581
1stFlrSF        0.605852
FullBath        0.560664
TotRmsAbvGrd    0.533723
YearBuilt       0.522897
YearRemodAdd    0.507101
GarageYrBlt     0.486362
MasVnrArea      0.477493
Fireplaces      0.466929
BsmtFinSF1      0.386420
LotFrontage     0.351799
'''