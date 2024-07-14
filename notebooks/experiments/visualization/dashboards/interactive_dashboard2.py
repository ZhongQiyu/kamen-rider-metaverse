# interactive_dashboard2.py

import dash
import dash_core_components as dcc
import dash_html_components as html
from ..plots.agent_viz import generate_agent_plot

app = dash.Dash(__name__)

app.layout = html.Div(children=[
    html.H1(children='Interactive Dashboard 2'),
    dcc.Graph(
        id='agent-graph',
        figure=generate_agent_plot()
    )
])

if __name__ == '__main__':
    app.run_server(debug=True)
