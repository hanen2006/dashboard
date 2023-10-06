#!/usr/bin/env python
# coding: utf-8

# In[1]:

import os
import requests

from dash import Dash, dcc, html
from dash.dependencies import Input, Output, State
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd



# In[8]:


# Configuration de l'API 
API_URL = "https://projetcloud-181a7c4bddfe.herokuapp.com/"

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
df = pd.read_csv("X_sample.csv")
feats = df.columns

app = Dash(__name__, external_stylesheets=external_stylesheets)

app.layout = html.Div([
    html.Div([
        dcc.Input(id="client_id", type="number", value='146052'),
        html.Button(id="validation_bt", n_clicks=0, children="Valider")
    ], style={'width': '33%', "float" : "right", 'display': 'inline-block'}),
    
    html.Div(dcc.Graph(id='bar_mean', figure = {"layout" : {"height" : 800}}), 
        style={'width': '33%', 'display': 'inline-block', 'padding': '0 20', "vertical_align" : "middle"}),
    
    html.Div(dcc.Graph(id='boxplot', figure = {"layout" : {"height" : 800}}), 
        style={'width': '33%', 'display': 'inline-block', 'padding': '0 20', "vertical_align" : "middle"}),
    
    html.Div([
        dcc.Graph(id='score')
    ], style={'display': 'inline-block', 'width': '33%', "float":"right"})
])

@app.callback(Output('score', 'figure'), Input('validation_bt', 'n_clicks'), State('client_id', 'value'))
def update_score(n_clicks, client_id):
    try:
        r = requests.get(f"{API_URL}/predict/{client_id}")
        r.raise_for_status()

        val = float(r.json()["predict_proba_0"]) * 100
        score_min = 0
        
        accept, color = ("Accepté", "darkgreen") if val > score_min else ("Refusé", "darkred")
        
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=val,
            domain={"x": [0, 1], "y": [0, 1]},
            title={"text": "Score Client"},
            gauge={"steps": [{"range": [0, score_min], "color": "red"},
                             {"range": [score_min, 100], "color": "green"}],
                   "bar": {"color": "black"}},
            number={"suffix": "%"}
        ))

        fig.add_annotation(text=accept, x=0.5, y=0.4, showarrow=False, font=dict(size=20, color=color), xanchor="center")
        fig.update_layout(height=400, width=400, margin={"t": 50, "b": 50, "l": 50, "r": 50})
        
        return fig
    
    except requests.RequestException as e:
        # Gérer les erreurs d'API ici
        print(e)
        return go.Figure()

@app.callback(Output('boxplot', 'figure'), Input('validation_bt', 'n_clicks'), State('client_id', 'value'))
def plot_boxplot(n_clicks, client_id):
    try:
        results = requests.get(f"{API_URL}/details/{client_id}").json()
        x = list(results['client_data'].keys())
        y = list(results['client_data'].values())

        fig = px.box(x=x, y=y, title="Client Data Box Plot", labels={"x": "Features", "y": "Value"})
        fig.update_layout(yaxis_title="Values", xaxis_title="Features", height=600, width=500)
        
        return fig
    
    except requests.RequestException as e:
        # Gérer les erreurs d'API ici
        print(e)
        return go.Figure()

@app.callback(Output('bar_mean', 'figure'), Input('validation_bt', 'n_clicks'), State('client_id', 'value'))
def plot_bar_mean(n_clicks, client_id):
    try:
        results = requests.get(f"{API_URL}/details/{client_id}").json()
        x = list(results['mean_values'].keys())
        y = list(results['mean_values'].values())

        fig = px.bar(x=x, y=y, title="Mean Values Bar Chart", labels={"x": "Features", "y": "Mean Value"})
        fig.update_layout(yaxis_title="Mean Value", xaxis_title="Features", height=600, width=500)
        
        return fig
    
    except requests.RequestException as e:
        # Gérer les erreurs d'API ici
        print(e)
        return go.Figure()

if __name__ == '__main__':
    app.run_server(debug=True, port=int(os.environ.get('PORT', 8050)))

# In[ ]:




