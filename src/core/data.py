from typing import Dict

import pandas as pd


def format_column_names(df):
    columns = [c.lower().strip().replace(" ", "_") for c in df.columns]
    return columns


def column_percent_null(df):
    print(df.isnull().sum() / len(df) * 100)


def get_categorical_columns(dataframe):
    categorical_columns = dataframe.select_dtypes(
        include=["object", "category"]
    ).columns.tolist()
    return categorical_columns


def csv_to_df(file_paths) -> Dict[str, pd.DataFrame]:
    """
    load csv files into a dictionary of dataframes
    """
    dfs = {}
    for file_path in file_paths:
        file_name = file_path.split("/")[-1].split(".")[0].lower()
        temp_df = pd.read_csv(file_path)
        dfs[file_name] = temp_df
    return dfs
