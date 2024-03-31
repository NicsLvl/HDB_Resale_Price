from src import clean_and_merge, eda, model
from src.overhead_functions import timer, memory_usage


@timer
def main():
    y_variable = 'resale_price'
    df = clean_and_merge.main()

    # eda.main(df)
    chosen_features = [
        # Traditional metrics used in industry
        'floor_area_sqm',  # 0.81 correlation
        'remaining_lease',  # 0.33 correlation
        'storey',  # 0.21 correlation

        # Other features that might be useful
        # Can drop if model is sufficiently good enough
        'flat_type',
        'town',
        'flat_model',

        # Should be essentially useless
        # 'block',
        # 'street_name',
    ]

    # Remove some minority categories
    # df = df[~df['flat_type'].isin(['MULTI GENERATION', '1 ROOM', '2 ROOM'])]
    # df = df[df['flat_model'].isin(
    #     ['model a', 'improved', 'new generation', 'simplified', 'premium apartment',
    #      'standard', 'apartment', 'maisonette', ' model a2'])]
    model.main(df, y_variable, chosen_features)
    return df


if __name__ == "__main__":
    main()
