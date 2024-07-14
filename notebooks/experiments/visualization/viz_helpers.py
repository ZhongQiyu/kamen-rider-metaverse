# viz_helpers.py

import plotly.express as px
import pandas as pd

def load_data(filepath: str) -> pd.DataFrame:
    """Load data from a CSV file."""
    return pd.read_csv(filepath)

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Perform basic data cleaning."""
    df = df.dropna()  # Drop rows with missing values
    df = df[df.select_dtypes(include=['number']).ge(0).all(1)]  # Remove negative values for numeric columns
    return df

def create_bar_plot(df: pd.DataFrame, x_col: str, y_col: str, title: str) -> px.bar:
    """Create a bar plot."""
    fig = px.bar(df, x=x_col, y=y_col, title=title)
    return fig

def create_line_plot(df: pd.DataFrame, x_col: str, y_col: str, title: str) -> px.line:
    """Create a line plot."""
    fig = px.line(df, x=x_col, y=y_col, title=title)
    return fig

def create_scatter_plot(df: pd.DataFrame, x_col: str, y_col: str, title: str) -> px.scatter:
    """Create a scatter plot."""
    fig = px.scatter(df, x=x_col, y=y_col, title=title)
    return fig
