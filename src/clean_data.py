from pathlib import Path
import pandas as pd
import numpy as np
import typing
from src.overhead_functions import timer, memory_usage


def downsize_df(df: pd.DataFrame) -> pd.DataFrame:
    """This function will be used to downsize the numerical features in the dataframe"""
    # Option 1 Traditional If Else statements
    # def if_else(df):
    #     for col in df.columns:
    #         col_data = df[col]
    #         if col_data.dtype.kind in 'biufc':
    #             c_min = col_data.min()
    #             c_max = col_data.max()
    #             if str(col_data.dtype).startswith("int"):
    #                 if c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
    #                     df[col] = df[col].astype(np.int16)
    #             else:
    #                 if c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
    #                     df[col] = df[col].astype(np.float32)
    #     return df
    # downsized_df = if_else(df)

    # Option 2: Match Case for better readability and logic
    def match_case(df):
        for col in df.columns:
            col_data = df[col]
            match col_data.dtype.kind:
                case 'i' | 'u':
                    c_min = col_data.min()
                    c_max = col_data.max()
                    match (c_min, c_max):
                        case (a, b) if a > np.iinfo(np.uint16).min and b < np.iinfo(np.uint16).max:
                            df[col] = df[col].astype(np.uint16)
                case 'f':
                    c_min = col_data.min()
                    c_max = col_data.max()
                    match (c_min, c_max):
                        case (a, b) if a > np.finfo(np.float32).min and b < np.finfo(np.float32).max:
                            df[col] = df[col].astype(np.float32)
                case _:
                    pass
        return df
    downsized_df = match_case(df)

    return downsized_df


def clean_columns(dfs: typing.Generator) -> pd.DataFrame:
    """This function will be used to clean the columns in the dataframe"""
    merged = pd.DataFrame()
    for df in dfs:
        # Convert all numerical figures to numerical first
        # get date_sold
        df['year_sold'] = df['month'].str[:4]
        df['year_sold'] = df['year_sold'].astype(int)
        df['month'] = df['month'].str[5:]
        df['month_sold'] = df['month'].astype(int)
        df['date_sold'] = df['year_sold'] + df['month_sold']/12

        df['dataset_category'] = df['year_sold'].min()

        df['lease_commence_date'] = df['lease_commence_date'].astype(int)

        # we assume all flats are 99 years lease
        df['remaining_lease'] = 99 - \
            (df['year_sold'] - df['lease_commence_date'])
        # it is possible that there is a release of booking rights for BTO hence the remaining_lease is > 99
        df.loc[df['remaining_lease'] > 99, 'remaining_lease'] = 99

        df['storey'] = df['storey_range'].apply(lambda x: (
            int(x.split(' ')[0]) + int(x.split(' ')[2])) / 2)

        df.drop(columns=['lease_commence_date',
                'year_sold', 'month_sold', 'storey_range'], inplace=True)
        # Downsize the numerical features in the dataframe
        df = downsize_df(df)

        # Perform other non-numerical operations
        df['flat_model'] = df['flat_model'].str.lower()
        df['flat_type'] = df['flat_type'].str.replace('-', ' ')

        merged = pd.concat([merged, df])
    return merged


def merge_data() -> pd.DataFrame:
    """This function will be used to merge the data from the raw data files"""
    hdb_raw_path = Path("data/raw_data")
    files = list(hdb_raw_path.glob("*.csv"))

    # Option 1: Use a generator expression
    @timer
    @memory_usage
    def load_generator():
        return (pd.read_csv(file) for file in files)

    # Option 2: Use a list comprehension
    @timer
    @memory_usage
    def load_list_comp():
        return [pd.read_csv(file) for file in files]

    # Which to Choose?
    # Memory Constraint: Generator, as it only loads one file at a time
    # Speed Constraint: List comprehension, as it loads all files already
    dfs = load_generator()
    # dfs = load_list_comp()

    merged = clean_columns(dfs)

    return merged


def normalize_price(df: pd.DataFrame) -> pd.DataFrame:
    """This function will be used to normalize the resale price based on the property price index"""
    """we want to normalize the resale_price based on the property price index
    """
    hdb_ppi = pd.read_csv(Path("data/processed_data") / "resalepriceindex.csv")
    hdb_ppi['date_sold'] = hdb_ppi['Year'] + hdb_ppi['Quarter'] / 4

    df['date_sold'] = df['date_sold'].apply(lambda x: round(x * 4) / 4)
    df = pd.merge(df, hdb_ppi[['date_sold', 'Index']],
                  on='date_sold', how='left')
    df.dropna(subset=['Index'], inplace=True)
    df['resale_price'] = df['resale_price'] * 100 / df['Index']
    df.drop(columns=['Index'], inplace=True)

    return df


def main() -> pd.DataFrame:
    """This function will be used to clean the data and return a cleaned dataframe"""
    df = merge_data()
    # Reduce memory of the categorical features in the dataframe
    # Keep the columns that required string operations as object to reduce memory
    df['town'] = df['town'].astype('category')
    df['flat_type'] = df['flat_type'].astype('category')
    df['flat_model'] = df['flat_model'].astype('category')

    df = normalize_price(df)
    df = df[['floor_area_sqm', 'remaining_lease',
             'storey', 'flat_type', 'town', 'block',
             'street_name', 'flat_model', 'resale_price']]

    return df
