# viz_helpers.py

import plotly.express as px
import pandas as pd

class VisualizationHelper:
    def __init__(self, filepath: str = None):
        self.filepath = filepath
        self.data = None

    def load_data(self, filepath: str = None) -> pd.DataFrame:
        """Load data from a CSV file."""
        if filepath:
            self.filepath = filepath
        if not self.filepath:
            raise ValueError("Filepath must be provided to load data.")
        
        self.data = pd.read_csv(self.filepath)
        return self.data

    def clean_data(self) -> pd.DataFrame:
        """Perform basic data cleaning."""
        if self.data is None:
            raise ValueError("No data loaded. Please load data first.")
        
        self.data = self.data.dropna()  # Drop rows with missing values
        self.data = self.data[self.data.select_dtypes(include=['number']).ge(0).all(1)]  # Remove negative values for numeric columns
        return self.data

    def create_bar_plot(self, x_col: str, y_col: str, title: str) -> px.bar:
        """Create a bar plot."""
        if self.data is None:
            raise ValueError("No data loaded. Please load data first.")
        
        fig = px.bar(self.data, x=x_col, y=y_col, title=title)
        return fig

    def create_line_plot(self, x_col: str, y_col: str, title: str) -> px.line:
        """Create a line plot."""
        if self.data is None:
            raise ValueError("No data loaded. Please load data first.")
        
        fig = px.line(self.data, x=x_col, y=y_col, title=title)
        return fig

    def create_scatter_plot(self, x_col: str, y_col: str, title: str) -> px.scatter:
        """Create a scatter plot."""
        if self.data is None:
            raise ValueError("No data loaded. Please load data first.")
        
        fig = px.scatter(self.data, x=x_col, y=y_col, title=title)
        return fig
