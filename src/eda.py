import pandas as pd
import pandas as pd
import plotly.express as px
import plotly.io as pio
from pathlib import Path
from src.overhead_functions import timer
import typing

pio.templates.default = "plotly_dark"
color_scale = px.colors.sequential.Viridis
pd.options.display.float_format = '{:.2f}'.format
EDA_PATH = Path('data/eda')


def get_graphs(df: pd.DataFrame):
    """This function will be used to perform exploratory data analysis"""
    print(df.head())
    print('')
    print(df.info())
    print('')
    print(df.describe())
    print('')

    figures = []

    # View distribution of all variables
    for col in df.columns:
        match df[col].dtype.kind:
            case 'i' | 'u' | 'f':
                fig = px.violin(df, y=col, points='all', box=True)
            case _:
                fig = px.bar(df[col].value_counts(), x=df[col].value_counts(
                ).index, y='count', color=df[col].value_counts().index)
        fig.write_html(EDA_PATH / f'{col}_plot.html')
        figures.append(dcc.Graph(figure=fig))

    fig = px.imshow(df.select_dtypes(
        include=['number']).corr(), color_continuous_scale=color_scale)
    fig.write_html(EDA_PATH / f'correlation_matrix.html')
    figures.append(dcc.Graph(figure=fig))

    cat_col = 'flat_type'
    df_numeric_and_cat = df.select_dtypes(include=['number']).copy()
    df_numeric_and_cat[cat_col] = df[cat_col]

    fig = px.parallel_categories(df_numeric_and_cat)
    fig.write_html(EDA_PATH / f'parallel_categories.html')
    figures.append(dcc.Graph(figure=fig))

    return figures


def transform(df: pd.DataFrame):
    """This function is used to showcase different ways to group data
    """
    displays = []

    # Displaying Grouped Data
    # Agg, is the fastest way to group data
    displays.append(df.groupby('storey').agg(
        mean_price=('resale_price', 'mean'),
        std_price=('resale_price', 'std'),
    ))

    displays.append(df.groupby(['town', 'storey']).agg({
        'resale_price': ['mean', 'std'],
        'floor_area_sqm': ['mean', 'std'],
    }))

    # Using Functions: Apply + Lambda, relatively slower but more flexible
    displays.append(pd.DataFrame(df.groupby('storey').apply(
        lambda x: x['resale_price'].max()-x['resale_price'].min()
    )))

    # Creating New Columns that matches the original number of rows in the index
    # Using Transform to create new data
    df['town_storey_avg'] = df.groupby(['town', 'storey'])[
        'resale_price'].transform('mean')
    displays.append(df[['town', 'storey', 'town_storey_avg']])

    for display in displays:
        print(display)
        print('')


def charts(df) -> typing.List:
    """This function is used to create charts for the streamlit app"""
    figs = []

    # First we investigate the skew
    fig = px.histogram(df, x='resale_price', nbins=50)
    fig.update_layout(title='Resale Price Skew')
    figs.append(fig)

    return figs


@timer
def main(df: pd.DataFrame):

    transform(df)
