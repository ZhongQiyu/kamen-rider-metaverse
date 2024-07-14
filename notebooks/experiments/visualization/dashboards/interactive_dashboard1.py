# interactive_dashboard1.py

import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.express as px
from ..plots.data_viz import generate_plot

app = dash.Dash(__name__)

app.layout = html.Div(children=[
    html.H1(children='Interactive Dashboard 1'),
    dcc.Graph(
        id='example-graph',
        figure=generate_plot()
    )
])

if __name__ == '__main__':
    app.run_server(debug=True)
