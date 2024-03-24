import pandas as pd
import pandas as pd
import plotly.express as px
import plotly.io as pio
from pathlib import Path

pio.templates.default = "plotly_dark"
color_scale = px.colors.sequential.Viridis
pd.options.display.float_format = '{:.2f}'.format
EDA_PATH = Path('data/eda')


def main(df: pd.DataFrame):
    """This function will be used to perform exploratory data analysis"""
    print(df.head())
    print('')
    print(df.info())
    print('')
    print(df.describe())
    print('')

    # View distribution of all variables
    for col in df.columns:
        match df[col].dtype.kind:
            case 'i' | 'u' | 'f':
                fig = px.violin(df, y=col)
            case _:
                fig = px.bar(df[col].value_counts(), x=df[col].value_counts(
                ).index, y='count', color=df[col].value_counts().index)
        fig.write_html(EDA_PATH / f'{col}_plot.html')

    fig = px.imshow(df.select_dtypes(
        include=['number']).corr(), color_continuous_scale=color_scale)
    fig.write_html(EDA_PATH / f'correlation_matrix.html')

    cat_col = 'flat_type'
    df_numeric_and_cat = df.select_dtypes(include=['number']).copy()
    df_numeric_and_cat[cat_col] = df[cat_col]

    fig = px.parallel_categories(df_numeric_and_cat)
    fig.write_html(EDA_PATH / f'parallel_categories.html')
