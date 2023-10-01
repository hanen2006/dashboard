#!/usr/bin/env python
# coding: utf-8

# In[1]:


from dash import Dash 
import dash
import dash_html_components as html
import dash_core_components as dcc
from dash.dependencies import Input, Output, State
import requests

from flask import Flask, jsonify, request, render_template
import joblib
import pickle
import pandas as pd
import shap
import json
import numpy as np
import plotly.graph_objects as go



# In[8]:


import plotly.express as px
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
df = pd.read_pickle("C:/Users/PC/Desktop/projetdatascienc/projet 7/X_data.pkl")
feats = df.columns

app = Dash(__name__, external_stylesheets=external_stylesheets)


app.layout = html.Div([
    html.Div([
        html.Div([
            dcc.Input(id="client_id", type="number", value='100002'),
            html.Button(id="validation_bt", n_clicks=0, children="Valider")
        ],
        style={'width': '33%', "float" : "right", 'display': 'inline-block'})
    ]),
    html.Div(dcc.Graph(id='bar_mean', figure = {"layout" : {"height" : 800}}), 
        style={'width': '33%', 'display': 'inline-block', 'padding': '0 20', "vertical_align" : "middle"}),
    html.Div(dcc.Graph(id='boxplot', figure = {"layout" : {"height" : 800}}), 
        style={'width': '33%', 'display': 'inline-block', 'padding': '0 20', "vertical_align" : "middle"}),
    html.Div([
        dcc.Graph(id='score')
    ], style={'display': 'inline-block', 'width': '33%', "float":"right"})
])


@app.callback(Output('score', 'figure'),
              Input('validation_bt', 'n_clicks'),
              State('client_id', 'value'))
def update_score(n_clicks, client_id):
    score_min = 0
    r = requests.get(f"http://127.0.0.1:5000/predict/{client_id}")
    val = float(r.json()["predict_proba_0"]) * 100
    print('val=', val)
    
    # Determine the acceptance text and color
    accept, color = ("Accepté", "darkgreen") if val > score_min else ("Refusé", "darkred")
    
    # Create the figure and add the indicator trace
    fig1 = go.Figure(go.Indicator(
        mode="gauge+number",
        value=val,
        domain={"x": [0, 1], "y": [0, 1]},
        title={"text": "Score Client", "font": {"size": 24}},
        gauge={
            "axis": {"range": [None, 100], "tickwidth": 1, "tickcolor": "darkblue"},
            "bar": {"color": "black"},
            "bgcolor": "white",
            "borderwidth": 2,
            "bordercolor": "gray",
            "steps": [
                {"range": [0, score_min], "color": "red"},
                {"range": [score_min, 100], "color": "green"},
            ],
        },
        number={"suffix": "%", "font": {"size": 20}},
    ))
    
    # Add annotation for the acceptance text
    fig1.add_annotation(text=accept, 
                        x=0.5, y=0.4, 
                        showarrow=False, 
                        font=dict(size=20, color=color),
                        xanchor="center")
    
    # Adjust the layout
    fig1.update_layout(
        height=400,  
        width=400,   
        margin={"t": 50, "b": 50, "l": 50, "r": 50},
    )
    
    return fig1

@app.callback(Output('boxplot', 'figure'),
              Input('validation_bt', 'n_clicks'),State('client_id', 'value'))
def plot_boxplot(n_clicks, client_id):
    try:
        results = requests.get(f"http://127.0.0.1:5000/details/{client_id}").json()                     
        x = list(results['client_data'].keys())
        y = list(results['client_data'].values())

        if not x or not y:
            raise ValueError("Data is empty or not valid")

        fig2 = px.box(x=x, y=y, title="Client Data Box Plot", labels={"x": "Features", "y": "Value"})
        fig2.update_layout(
            yaxis_title="Values", 
            xaxis_title="Features",
            height=600,  
            width=500    
        )
        return fig2
    except Exception as e:
        print(e)
        # Return an empty figure in case of errors
        return go.Figure()

@app.callback(Output('bar_mean', 'figure'),
              Input('validation_bt', 'n_clicks'),State('client_id', 'value'))
def plot_bar_mean(n_clicks, client_id):
    results = requests.get(f"http://127.0.0.1:5000/details/{client_id}").json()                     
    x = list(results['mean_values'].keys())
    y = list(results['mean_values'].values())
    
    fig3 = px.bar(x=x, y=y, title="Mean Values Bar Chart", labels={"x": "Features", "y": "Mean Value"})
    fig3.update_layout(
        yaxis_title="Mean Value", 
        xaxis_title="Features",
        height=600,  
        width=500    
    )
    return fig3

if __name__ == '__main__':
    app.run_server(debug=True)


# In[ ]:




