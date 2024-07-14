# agent_viz.py

import plotly.express as px
import pandas as pd

def generate_agent_plot():
    df = pd.DataFrame({
        "Agent": ["Agent1", "Agent2", "Agent3"],
        "Performance": [80, 90, 85]
    })
    fig = px.line(df, x="Agent", y="Performance")
    return fig
