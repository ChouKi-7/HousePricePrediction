import pandas as pd

def handle_missing_values(df, drop_thresh=0.8, verbose=True):
    """
    处理缺失值：
    1. 删除缺失比例高于 drop_thresh 的列(默认0.8)
    2. 数值型列用中位数填充
    3. 类别型列用 'Missing' 填充
    4. 返回处理后的 DataFrame
    """
    df_copy = df.copy()  # 不直接改原数据
    rows = len(df_copy)

    # 第一步：删除高缺失比例列
    missing_ratio = df_copy.isnull().sum() / rows
    columns_to_drop = missing_ratio[missing_ratio > drop_thresh].index
    if verbose:
        print(f"🗑️ 删除以下缺失率 > {drop_thresh*100:.0f}% 的列：")
        print(list(columns_to_drop))
    df_copy.drop(columns=columns_to_drop, axis=1, inplace=True)

    # 第二步：处理数值型列（中位数填补）
    for col in df_copy.select_dtypes(include=['float64', 'int64']):
        if df_copy[col].isnull().sum() > 0:
            df_copy[col] = df_copy[col].fillna(df_copy[col].median())
    
    # 第三步：处理类别型列（填'Missing'）
    for col in df_copy.select_dtypes(include=['object']):
        if df_copy[col].isnull().sum() > 0:
            df_copy[col] = df_copy[col].fillna('Missing')

    # 第四步：缺失值检查报告
    total_missing = df_copy.isnull().sum().sum()
    if verbose:
        print(f"\n✅ 缺失值处理完成，剩余缺失值数量：{total_missing}")

    return df_copy