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

from dash.exceptions import PreventUpdate


# In[2]:
 
# Configuration de l'API 
API_URL = "https://projetcloud-181a7c4bddfe.herokuapp.com/"

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
df = pd.read_pickle("C:/Users/PC/Desktop/projetdatascienc/projet 7/X_data.pkl")
feats = df.columns
df.drop(columns='TARGET', inplace=True)
server = app.server
# Initialiser l'application
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

def get_all_scores():
    response = requests.get(f"{API_URL}all_scores")
    if response.status_code == 200:
        return response.json()
    else:
        return []  # ou gérer l'erreur comme vous le souhaitez
def create_score_histogram():
    scores = get_all_scores()

    accepted_scores = [score for score in scores if score >= 50]
    rejected_scores = [score for score in scores if score < 50]

    fig = go.Figure()

    fig.add_trace(go.Histogram(
        x=accepted_scores,
        name='Accepté',
        marker_color='darkgreen'
    ))

    fig.add_trace(go.Histogram(
        x=rejected_scores,
        name='Rejeté',
        marker_color='darkred'
    ))

    fig.update_layout(
        title_text='Distribution des scores',
        xaxis_title_text='Score',
        yaxis_title_text='Nombre de clients',
        bargap=0.2,
        barmode='overlay'
    )
    fig.update_traces(opacity=0.75)

    return fig

VARIABLES_TO_PLOT = ['CREDIT_TERM', 'EXT_SOURCE_1', 'DAYS_BIRTH', 'PREV_APPL_MEAN_CNT_PAYMENT', 
                    'ANNUITY_INCOME_PERCENT', 'AMT_CREDIT']
app.layout = html.Div([
    html.Div([
        html.Div([
            dcc.Input(id="client_id", type="number", value='146052'),
            html.Button(id="validation_bt", n_clicks=0, children="Valider")
        ],
        style={'width': '33%', "float" : "right", 'display': 'inline-block'})
    ]),

    html.Div(dcc.Graph(id='DAYS_EMPLOYED_plot', figure={"layout": {"height": 600, "width": 1000}}), 
         style={'width': '100%', 'display': 'inline-block'}),
    html.Div(dcc.Graph(id='shap_plot', figure={"layout": {"height": 800}}), 
         style={'width': '33%', 'display': 'inline-block', 'padding': '0 20'}),
    html.Div(dcc.Graph(id='score_distribution', 
                   figure={"layout": {"height": 600, "width": 1000}}), 
         style={'width': '100%', 'display': 'inline-block'}),
    html.Div(dcc.Graph(id='score_vs_days_employed', figure={"layout": {"height": 600, "width": 1000}}), 
     style={'width': '100%', 'display': 'inline-block'}),
    html.Div([
        dcc.Graph(id='score')
    ], style={'display': 'inline-block', 'width': '33%', "float":"right"}),
    html.Button('Afficher les scores', id='display-button', n_clicks=0),# Ajout du bouton
   
    html.Div([dcc.Graph(id=f'score_vs_{var}', figure={"layout": {"height": 600, "width": 1000}}) 
              for var in VARIABLES_TO_PLOT], style={'width': '100%', 'display': 'inline-block'})
])


@app.callback(
    Output('score-distribution', 'figure'),
    [Input('display-button', 'n_clicks')]
)
def update_histogram(n_clicks):
    if n_clicks > 0:
        return create_score_histogram()
    return go.Figure()

@app.callback(
    [Output(f'score_vs_{var}', 'figure') for var in VARIABLES_TO_PLOT],
    [Input('validation_bt', 'n_clicks')],
    [State('client_id', 'value')]
)
def update_score_vs_variables(n_clicks, client_id):
    if n_clicks == 0:
        raise PreventUpdate

    scores = requests.get(f"{API_URL}all_scores").json()
    client_details = requests.get(f"{API_URL}details/").json()

    figures = []
    for var in VARIABLES_TO_PLOT:
        var_values = [details[var] for sk_id, details in client_details.items()]
        fig = px.scatter(x=var_values, y=scores, title=f"Scores vs {var}")

        specific_client_data = requests.get(f"{API_URL}details/{client_id}").json()["client_data"]
        specific_score = requests.get(f"{API_URL}predict/{client_id}").json()
        specific_score_value = float(specific_score["predict_proba_0"]) * 100
        specific_var_value = specific_client_data[var]

        fig.add_trace(
            go.Scatter(x=[specific_var_value], y=[specific_score_value], mode='markers', marker=dict(size=10, color='coral'), name=f'Client {client_id}')
        )
        
        fig.add_shape(
            type='line',
            y0=50, y1=50,
            x0=min(var_values), x1=max(var_values),
            line=dict(color='red', width=2, dash="dash"),
            name="Limite de score"
        )

        fig.update_layout(
            xaxis_title=var,
            yaxis_title="Score"
        )

        figures.append(fig)

    return tuple(figures)

@app.callback(Output('DAYS_EMPLOYED_plot', 'figure'),
              [Input('validation_bt', 'n_clicks')],
              [State('client_id', 'value')])
def update_DAYS_EMPLOYED(n_clicks, client_id):
    if n_clicks == 0:
        raise PreventUpdate

    # Récupérer la prédiction pour le client spécifique depuis l'API
    prediction_response = requests.get(f"{API_URL}predict/{client_id}")
    
    # Vérification du code de statut de la réponse
    if prediction_response.status_code != 200:
        # Si le code de statut n'est pas 200 (OK), cela signifie que nous n'avons pas pu obtenir les données pour le client spécifique.
        # Retourner l'histogramme sans mettre en évidence le client
        return px.histogram(df, x='DAYS_EMPLOYED', nbins=50, title="Distribution des clients selon DAYS_EMPLOYED")
    
    prediction = int(float(prediction_response.json()['retour_prediction']))


    # Séparer les données en fonction de la prédiction
    if prediction == 0:
        color = 'green'
    else:
        color = 'yellow'

    # Créer un histogramme pour tous les clients
    fig = px.histogram(df, x='DAYS_EMPLOYED', nbins=50, title="Distribution des clients selon DAYS_EMPLOYED", color_discrete_sequence=[color])

    # Récupération des données pour le client spécifique depuis l'API
    specific_data_response = requests.get(f"{API_URL}details/{client_id}")
    specific_data = specific_data_response.json()["client_data"]
    specific_value = specific_data['DAYS_EMPLOYED']
    # Marquage du client spécifique
    fig.add_vline(x=specific_value, line_dash="dash", line_color="coral", line_width=3)

    # Ajout d'une trace pour le client spécifique
    fig.add_trace(
        go.Histogram(x=[specific_value], marker_color='darkgreen', name=f'Client {client_id}')
    )

    return fig
@app.callback(Output('score_vs_days_employed', 'figure'),
              [Input('validation_bt', 'n_clicks')],
              [State('client_id', 'value')])
def update_score_vs_days_employed_plot(n_clicks, client_id):
    if n_clicks == 0:
        raise PreventUpdate

    # Récupérer les scores et les détails de tous les clients
    scores = requests.get(f"{API_URL}all_scores").json()
    client_details = requests.get(f"{API_URL}details/").json()
    
    days_employed_values = [details['DAYS_EMPLOYED'] for sk_id, details in client_details.items()]

    # Créer un scatter plot pour tous les clients
    fig = px.scatter(x=days_employed_values, y=scores, title="Scores vs DAYS_EMPLOYED")

    # Récupérer les données pour le client spécifique
    specific_client_data = requests.get(f"{API_URL}details/{client_id}").json()["client_data"]
    specific_score = requests.get(f"{API_URL}predict/{client_id}").json()
    specific_score_value = float(specific_score["predict_proba_0"]) * 100
    specific_days_employed = specific_client_data['DAYS_EMPLOYED']

    # Marquage du client spécifique avec un point distinct
    fig.add_trace(
        go.Scatter(x=[specific_days_employed], y=[specific_score_value], mode='markers', marker=dict(size=10, color='darkgreen'), name=f'Client {client_id}')
    )

    # Ajout de la barre limite pour les scores de 50
    fig.add_shape(
        type='line',
        y0=50, y1=50,
        x0=min(days_employed_values), x1=max(days_employed_values),
        line=dict(color='red', width=2, dash="dash"),
        name="Limite de score"
    )

    # Ajout des titres pour les axes
    fig.update_layout(
        xaxis_title="DAYS_EMPLOYED",
        yaxis_title="Score"
    )

    return fig

@app.callback(Output('score', 'figure'),
              Input('validation_bt', 'n_clicks'),
              State('client_id', 'value'))
def update_score(n_clicks, client_id):
    score_min = 0
    r = requests.get(f"http://127.0.0.1:5001/predict/{client_id}")
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

@app.callback(Output('score_distribution', 'figure'),   # Notez l'underscore ici
              [Input('validation_bt', 'n_clicks')])

def update_all_scores(n_clicks):
    if n_clicks == 0:
        raise PreventUpdate
    return create_score_histogram()

    # Récupérer la liste des clients depuis l'API
    client_response = requests.get(f"{API_URL}predict")
    client_list = client_response.json()["list_client_id"]

    scores = []
    predictions = []
    
    # Pour chaque client, récupérez le score et la prédiction
    for client_id in client_list:
        r = requests.get(f"{API_URL}predict/{client_id}")
        scores.append(float(r.json()["predict_proba_0"]) * 100)
        predictions.append(int(float(r.json()['retour_prediction'])))

    # Convertir les scores et les prédictions en DataFrame
    df_scores = pd.DataFrame({
        'Client': client_list,
        'Score': scores,
        'Prediction': predictions
    })

    # Créez le diagramme à barres
    fig = px.bar(df_scores, x='Client', y='Score', color='Prediction',
                 color_discrete_sequence=['green', 'red'],
                 labels={'Prediction': 'Résultat de la prédiction', 'Score': 'Score Client', 'Client': 'ID du client'})
    
    fig.update_layout(title="Distribution des scores par client")

    return fig


@app.callback(Output('shap_plot', 'figure'),
              Input('validation_bt', 'n_clicks'),
              State('client_id', 'value'))
def update_shap_plot(n_clicks, client_id):
    try:
        # Obtenez les valeurs SHAP à partir de votre API Flask
        results = requests.get(f"http://127.0.0.1:5001/shap_values/{client_id}").json()
        shap_values = results['shap_values']

        # Définir la couleur des barres en fonction de leurs valeurs
        colors = ['darkred' if val > 0 else 'darkblue' for val in shap_values]

        # Créez un bar plot avec les valeurs SHAP
        fig = go.Figure(data=[
            go.Bar(x=df.columns, y=shap_values, marker=dict(color=colors), name='SHAP Values'),
        ])
        
        # Mettre à jour la disposition pour une meilleure taille
        fig.update_layout(
            title=f"Principales caractéristiques pour le client{client_id}", 
            xaxis_title="Principales caractéristiques", 
            yaxis_title="SHAP Valeur",
            width=800,   # largeur du tracé en pixels
            height=1600   # hauteur du tracé en pixels
        )

        return fig

    except Exception as e:
        print(e)
        return go.Figure()



if __name__ == '__main__':
    #app.run_server(port=8052)
    
  
    


# In[ ]:
