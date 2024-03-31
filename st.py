import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def eda_charts(df: pd.DataFrame) -> list:
    charts = []

    fig = go.Figure()
    for data_line in df['flat_type'].unique():
        fig.add_trace(go.Violin(
            x=df[df['flat_type'] == data_line]['resale_price'],
            name=data_line,
            box_visible=True,
            line_color=px.colors.sequential.Viridis[df['flat_type'].unique(
            ).tolist().index(data_line) % len(px.colors.sequential.Viridis)]
        ))
    fig.update_traces(orientation='h', side='positive', width=3, points=False)
    fig.update_layout(xaxis_showgrid=False, xaxis_zeroline=False)
    charts.append(fig)

    # Investigate distribution of resale prices for all the different flat_types
    rooms_df = (df[df['flat_type'] == x] for x in df['flat_type'].unique())

    for room_df in rooms_df:
        room_df['amplified_remaining_lease'] = room_df['remaining_lease'] ** 3
        fig = px.scatter(room_df, x='floor_area_sqm', y='resale_price',
                         color='flat_model', size='amplified_remaining_lease',
                         title=room_df['flat_type'].iloc[0],
                         size_max=20)
        charts.append(fig)
    return charts


def main():
    df = pd.read_csv("data/processed_data/merged_data.csv")
    df['year_sold'] = df['year_sold'].astype(str)
    for chart in eda_charts(df):
        st.plotly_chart(chart, theme="streamlit")


if __name__ == "__main__":
    main()
