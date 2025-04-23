import pandas as pd


def clean_column_names(df):
    df.columns = df.columns.str.strip().str.lower()
    df.columns = df.columns.str.replace(r'[^a-z0-9_]', '_', regex=True)
    return df


def check_mandatory_columns(df, mandatory_columns):
    missing = [col for col in mandatory_columns if col not in df.columns]
    return missing


def convert_data_types(df):
    for col in df.columns:
        try:
            df[col] = pd.to_numeric(df[col], errors='ignore')
            df[col] = pd.to_datetime(df[col], errors='ignore')
        except:
            continue
    return df


def remove_unwanted_columns(df, unwanted_keywords=['id', 'timestamp']):
    unwanted_cols = [col for col in df.columns if any(key in col for key in unwanted_keywords)]
    return df.drop(columns=unwanted_cols, errors='ignore')


def remove_duplicates(df):
    return df.drop_duplicates()


def standardize_categorical_values(df, column_mappings):
    for col, mapping in column_mappings.items():
        if col in df.columns:
            df[col] = df[col].replace(mapping)
    return df


def clean_special_characters(df):
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].astype(str).str.replace('$', '', regex=False)
            df[col] = df[col].str.replace('-', '', regex=False)
            df[col] = df[col].str.strip()
    return df


def enforce_unit_consistency(df, unit_columns):
    # Placeholder logic: Ensure numeric and consistent
    for col in unit_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    return df


def clean_data(df, mandatory_columns, unwanted_keywords, column_mappings, unit_columns):
    df = clean_column_names(df)
    missing_cols = check_mandatory_columns(df, mandatory_columns)
    if missing_cols:
        raise ValueError(f"Missing mandatory columns: {missing_cols}")
    df = convert_data_types(df)
    df = remove_unwanted_columns(df, unwanted_keywords)
    df = remove_duplicates(df)
    # df = standardize_categorical_values(df, column_mappings)
    df = clean_special_characters(df)
    # df = enforce_unit_consistency(df, unit_columns)
    return df
