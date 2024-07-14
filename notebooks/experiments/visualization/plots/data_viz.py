# data_viz.py

import plotly.express as px
import pandas as pd

def generate_plot():
    df = pd.DataFrame({
        "Category": ["A", "B", "C"],
        "Values": [10, 20, 30]
    })
    fig = px.bar(df, x="Category", y="Values")
    return fig
