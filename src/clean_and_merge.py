from pathlib import Path
import pandas as pd
import typing
from src.overhead_functions import timer, memory_usage


def merge_data() -> pd.DataFrame:
    """This function will be used to merge the data from the raw data files"""
    hdb_raw_path = Path("data/raw_data")
    files = list(hdb_raw_path.glob("*.csv"))

    # Option 1: Use a generator expression
    @timer
    @memory_usage
    def load_generator() -> typing.Generator[pd.DataFrame, None, None]:
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


def clean_columns(dfs: typing.Generator) -> pd.DataFrame:
    """This function will be used to clean the columns in the dataframe"""
    merged = pd.DataFrame()
    for df in dfs:
        # Date Sold
        # e.g. 2024 April = 2024.25
        df['year_sold'] = df['month'].str[:4]
        df['year_sold'] = df['year_sold'].astype(int)
        df['month'] = df['month'].str[5:]
        df['month_sold'] = df['month'].astype(int)
        df['date_sold'] = df['year_sold'] + df['month_sold']/12

        df['dataset_category'] = df['year_sold'].min()

        # Remaining Lease
        # we assume all flats are 99 years lease to calculate remaining_lease
        df['lease_commence_date'] = df['lease_commence_date'].astype(int)
        df['remaining_lease'] = 99 - \
            (df['year_sold'] - df['lease_commence_date'])
        # After exploring the dataset there are cases of release of booking rights
        # Hence the remaining_lease is > 99
        # Option 1: Use apply and a custom (lambdas) function

        @timer
        @memory_usage
        def apply_function(df):
            df['remaining_lease'] = df['remaining_lease'].apply(
                lambda x: min(x, 99))
            return df
        # df_apply = apply_function(df)
        # Option 2: Use vectorized operations

        @timer
        @memory_usage
        def vectorized_loc(df):
            df.loc[df['remaining_lease'] > 99, 'remaining_lease'] = 99
            return df
        # df = vectorized_loc(df)
        # Which to Choose?
        # Vectorized functions are always fastest and most memory efficient
        # Only use apply, applymap when you need custom operations

        # Storey Range
        # "8 to 10" = 9
        df['storey'] = df['storey_range'].apply(lambda x: (
            int(x.split(' ')[0]) + int(x.split(' ')[2])) / 2)

        df.drop(columns=['lease_commence_date',
                         #  'year_sold',
                         'month_sold',
                         'storey_range'], inplace=True)

        # Perform other non-numerical operations
        df['flat_model'] = df['flat_model'].str.lower()
        df['flat_type'] = df['flat_type'].str.replace('-', ' ')

        merged = pd.concat([merged, df])
    return merged


def downcast_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """This function will be used to downsize the numerical features in the dataframe"""
    for col in df.columns:
        match df[col].dtype.kind:
            case 'i' | 'u':
                df[col] = pd.to_numeric(df[col], downcast='integer')
            case 'f':
                df[col] = pd.to_numeric(df[col], downcast='float')
            case _:
                pass
    return df


def normalize_price(df: pd.DataFrame) -> pd.DataFrame:
    """This function will be used to normalize the resale price based on the property price index
    we want to normalize the resale_price based on the property price index
    """
    hdb_ppi = pd.read_csv(Path("data/processed_data") / "resalepriceindex.csv")
    hdb_ppi['date_sold'] = hdb_ppi['Year'] + hdb_ppi['Quarter'] / 4

    df['date_sold'] = df['date_sold'].apply(lambda x: round(x * 4) / 4)

    # Vlookup /  mapping operations
    # Option 1 use the merge function
    @timer
    @memory_usage
    def merge_function():
        return pd.merge(df, hdb_ppi[['date_sold', 'Index']],
                        on='date_sold', how='left')
    # Option 2 use the map function

    @timer
    @memory_usage
    def map_function():
        return df['date_sold'].map(hdb_ppi.set_index('date_sold')['Index'])

    # Which to Choose?
    # map_function is for simpler operations, faster and more memory efficient
    # merge_function is used when there are multiple conditions or inner/outer/left/right joins to consider
    # Choose based on the complexity of the mapping required
    # df = merge_function()
    df['Index'] = map_function()

    df.dropna(subset=['Index'], inplace=True)
    df['resale_price'] = df['resale_price'] * 100 / df['Index']
    df.drop(columns=['Index'], inplace=True)

    return df


def main() -> pd.DataFrame:
    """This function will be used to clean the data after merge multiple datasets"""
    df = merge_data()

    # Reduce memory of the dataframe
    df = downcast_numeric(df)
    # category works best if there are limited set of distinct values
    # Keep the columns that required string operations as object to reduce memory
    df['town'] = df['town'].astype('category')
    df['flat_type'] = df['flat_type'].astype('category')
    df['flat_model'] = df['flat_model'].astype('category')

    df = normalize_price(df)
    df = df[['year_sold', 'floor_area_sqm', 'remaining_lease',
             'storey', 'flat_type', 'town', 'block',
             'street_name', 'flat_model', 'resale_price']]

    df.to_csv(Path("data/processed_data") / "merged_data.csv", index=False)

    return df
