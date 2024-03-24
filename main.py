from src import clean_data, eda, model
from src.overhead_functions import timer, memory_usage


@timer
def main():
    y_variable = 'resale_price'
    df = clean_data.main()

    eda.main(df)
    chosen_features = [
        'floor_area_sqm',  # 0.81 correlation
        'remaining_lease',  # 0.33 correlation
        'storey',  # 0.21 correlation
        'flat_type',  # can drop if model is sufficiently good enough
        'town',  # can drop if model is sufficiently good enough
        'flat_model',  # can drop if model is sufficiently good enough
        # 'block',  # should be essentially useless
        # 'street_name',  # should be essentially useless
    ]

    # Remove some minority categories
    # df = df[~df['flat_type'].isin(['MULTI GENERATION', '1 ROOM', '2 ROOM'])]
    # df = df[df['flat_model'].isin(
    #     ['model a', 'improved', 'new generation', 'simplified', 'premium apartment',
    #      'standard', 'apartment', 'maisonette', ' model a2'])]
    model.main(df, y_variable, chosen_features)


if __name__ == "__main__":
    main()
