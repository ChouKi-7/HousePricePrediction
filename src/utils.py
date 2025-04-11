import pandas as pd

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