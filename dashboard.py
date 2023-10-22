
# ====================================================================
# Chargement des librairies
# ====================================================================
import sys
sys.setrecursionlimit(3000)

import time
import streamlit as st
import numpy as np
import requests

import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import pickle
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import seaborn as sns
import shap
from urllib.request import urlopen
import json
import plotly.express as px
import plotly.figure_factory as ff
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import streamlit.components.v1 as components

# ====================================================================
# CHARGEMENT DES DONNEES
# ====================================================================
data = pd.read_csv("X_sample.csv")
data.set_index('SK_ID_CURR', inplace=True)
df_clients = pd.read_csv("X_sample_intiale.csv")
df_clients.set_index('SK_ID_CURR', inplace=True)
# Chargement des features importance de ligthgbm
with open("shapvalue.pkl", "rb") as file:
    shap_values= pickle.load(file)
# Chargement  du modèle
best_model = pickle.load(open("LGBMClassifier.pkl", "rb"))


### Data
def show_data ():
    st.write(data.head(10))


### Solvency
def pie_chart(thres):
    #st.write(100* (data['TARGET']>thres).sum()/data.shape[0])
    percent_sup_seuil =100* (data['TARGET']>thres).sum()/data.shape[0]
    percent_inf_seuil = 100-percent_sup_seuil
    d = {'col1': [percent_sup_seuil,percent_inf_seuil], 'col2': ['% Non remboursé','% remboursé',]}
    df = pd.DataFrame(data=d)
    fig = px.pie(df,values='col1', names='col2', title=' Pourcentage de remboursements des clients')
    st.plotly_chart(fig)
def show_overview():
    st.title("Risque")
    risque_threshold = st.slider(label = 'Seuil de risque', min_value = 0.0,
                    max_value = 1.0 ,
                     value = 0.5,
                     step = 0.1)
    #st.write(risque_threshold)
    pie_chart(risque_threshold) 

### Graphs
def filter_graphs():
    st.subheader("Filtre des Graphes")
    col1, col2,col3 = st.columns(3)
    is_educ_selected = col1.radio("Graph Education",('non','oui'))
    is_statut_selected = col2.radio('Graph Statut',('non','oui'))
    is_income_selected = col3.radio('Graph Revenu',('non','oui'))

    return is_educ_selected,is_statut_selected,is_income_selected

def hist_graph ():
    st.bar_chart(df_clients['DAYS_BIRTH'])
    df = pd.DataFrame(df_clients[:200],columns = ['DAYS_BIRTH','AMT_CREDIT'])
    df.hist()
    st.pyplot()

def education_type(client_id):
    ed = df_clients.groupby('NAME_EDUCATION_TYPE').NAME_EDUCATION_TYPE.count()
    u_ed = df_clients.NAME_EDUCATION_TYPE.unique()

    fig = go.Figure(data=[go.Bar(
        x=u_ed,
        y=ed
    )])
    fig.update_layout(title_text='Data education')

    st.plotly_chart(fig)

    ed_solvable = df_clients[df_clients['TARGET'] == 0].groupby('NAME_EDUCATION_TYPE').NAME_EDUCATION_TYPE.count()
    ed_non_solvable = df_clients[df_clients['TARGET'] == 1].groupby('NAME_EDUCATION_TYPE').NAME_EDUCATION_TYPE.count()
    u_ed = df_clients.NAME_EDUCATION_TYPE.unique()

    fig = go.Figure(data=[
        go.Bar(name='Solvable', x=u_ed, y=ed_solvable),
        go.Bar(name='Non Solvable', x=u_ed, y=ed_non_solvable)
    ])
    fig.update_layout(title_text='Solvabilité Vs education')

    # Récupérer les données du client en fonction de son identifiant
    client_data = df_clients.loc[df_clients.index== client_id]

    # Si le client existe dans les données
    if not client_data.empty:
        client_education = client_data['NAME_EDUCATION_TYPE'].values[0]

        # Ajouter une marque pour le client
        fig.add_trace(go.Scatter(
            x=[client_education],
            y=[ed_solvable[client_education] if client_data['TARGET'].values[0] == 0 else ed_non_solvable[client_education]],
            mode='markers',
            marker=dict(color='red', size=10),
            name='Client'
        ))

    st.plotly_chart(fig)


def statut_plot (client_id):
    ed = df_clients.groupby('NAME_FAMILY_STATUS').NAME_FAMILY_STATUS.count()
    u_ed = df_clients.NAME_FAMILY_STATUS.unique() 
    #fig = plt.bar(u_ed, ed, bottom=None, color='blue', label='Education')
    #st.plotly_chart(fig)

    fig = go.Figure(data=[go.Bar(
            x=u_ed,
            y=ed
        )])
    fig.update_layout(title_text='Data situation familiale')
    st.plotly_chart(fig)

    ed_solvable = df_clients[df_clients['TARGET']==0].groupby('NAME_FAMILY_STATUS').NAME_FAMILY_STATUS.count()
    ed_non_solvable = df_clients[df_clients['TARGET']==1].groupby('NAME_FAMILY_STATUS').NAME_FAMILY_STATUS.count()
    u_ed = df_clients.NAME_FAMILY_STATUS.unique() 
    #fig = plt.bar(u_ed, ed, bottom=None, color='blue', label='Education')
    #st.plotly_chart(fig)

    fig = go.Figure(data=[
        go.Bar(name='Solvable',x=u_ed,y=ed_solvable),
        go.Bar(name='Non Solvable',x=u_ed,y=ed_non_solvable) 
        ])
    fig.update_layout(title_text='Solvabilité Vs situation familiale')
    # Récupérer les données du client en fonction de son identifiant
    client_data = df_clients.loc[df_clients.index == client_id]

    # Si le client existe dans les données
    if not client_data.empty:
        client_statut = client_data['NAME_FAMILY_STATUS'].values[0]

        # Ajouter une marque pour le client
        fig.add_trace(go.Scatter(
            x=[client_statut],
            y=[ed_solvable[client_statut] if client_data['TARGET'].values[0] == 0 else ed_non_solvable[client_statut]],
            mode='markers',
            marker=dict(color='red', size=10),
            name='Client'
        ))

    st.plotly_chart(fig)
   

def income_type (client_id):
    ed = df_clients.groupby('NAME_INCOME_TYPE').NAME_INCOME_TYPE.count()
    u_ed = df_clients.NAME_INCOME_TYPE.unique() 
    #fig = plt.bar(u_ed, ed, bottom=None, color='blue', label='Education')
    #st.plotly_chart(fig)

    fig = go.Figure(data=[go.Bar(
            x=u_ed,
            y=ed
        )])
    fig.update_layout(title_text='Data Type de Revenu')

    st.plotly_chart(fig)

    ed_solvable = df_clients[df_clients['TARGET']==0].groupby('NAME_INCOME_TYPE').NAME_INCOME_TYPE.count()
    ed_non_solvable = df_clients[df_clients['TARGET']==1].groupby('NAME_INCOME_TYPE').NAME_INCOME_TYPE.count()
    u_ed = df_clients.NAME_INCOME_TYPE.unique() 

    fig = go.Figure(data=[
        go.Bar(name='Solvable',x=u_ed,y=ed_solvable),
        go.Bar(name='Non Solvable',x=u_ed,y=ed_non_solvable) 
        ])
    fig.update_layout(title_text='Solvabilité Vs Type de Revenu')
    # Récupérer les données du client en fonction de son identifiant
    client_data = df_clients.loc[df_clients.index== client_id]

    # Si le client existe dans les données
    if not client_data.empty:
        client_income = client_data['NAME_INCOME_TYPE'].values[0]

        # Ajouter une marque pour le client
        fig.add_trace(go.Scatter(
            x=[client_income],
            y=[ed_solvable[client_income] if client_data['TARGET'].values[0] == 0 else ed_non_solvable[client_income]],
            mode='markers',
            marker=dict(color='red', size=10),
            name='Client'
        ))

    st.plotly_chart(fig)
   
 

###------------------------ Distribution ------------------------
def filter_distribution():
    st.subheader("Filtre des Distribution")
    col1, col2 = st.columns(2)
    is_age_selected = col1.radio("Distribution Age ",('non','oui'))
    is_incomdis_selected = col2.radio('Distribution Revenus ',('non','oui'))

    return is_age_selected,is_incomdis_selected 

def age_distribution(client_id):
    df = pd.DataFrame({'Age': df_clients['DAYS_BIRTH'],
                       'Solvabilite': df_clients['TARGET']})

    dic = {0: "solvable", 1: "non solvable"}
    df = df.replace({"Solvabilite": dic})

    # Récupérer l'âge du client en fonction de son identifiant
    client_age = df_clients.loc[df_clients.index == client_id, 'DAYS_BIRTH'].values[0]

    # Créer le graphique
    fig = px.histogram(df, x="Age", color="Solvabilite", nbins=40)

    # Ajouter un point spécifique pour le client
    fig.add_trace(go.Scatter(x=[client_age], y=[0], mode='markers', name='Client', marker=dict(color='red', size=10)))

    st.subheader("Distribution des ages selon la solvabilité")
    st.plotly_chart(fig)



def revenu_distribution(client_id):
    df = pd.DataFrame({'Revenus':df_clients['AMT_INCOME_TOTAL'],
                'Solvabilite':df_clients['TARGET']})

    dic = {0: "solvable", 1: "non solvable"}        
    df=df.replace({"Solvabilite": dic})    
     # Récupérer revenu du client en fonction de son identifiant
    client_revenu = df_clients.loc[df_clients.index== client_id, 'AMT_INCOME_TOTAL'].values[0]

    # Créer le graphique
    fig = px.histogram(df,x="Revenus", color="Solvabilite", nbins=40)
    # Ajouter un point spécifique pour le client
    fig.add_trace(go.Scatter(x=[client_revenu], y=[0], mode='markers', name='Client', marker=dict(color='red', size=10)))
    st.subheader("Distribution des revenus selon la sovabilité")
    st.plotly_chart(fig)



#--------------------------- Client Predection --------------------------

def show_client_predection():
    
    
    client_id =st.number_input("Donnez Id du Client",step=1)
    if st.button('Voir Client'):
        # Configuration de l'API 
        API_URL = "https://projetcloud-181a7c4bddfe.herokuapp.com/predict/"+ str(client_id)
        with st.spinner('Chargement du score du client...'):
            json_url = urlopen( API_URL)
            API_data = json.loads(json_url.read())
            y_pred = API_data['retour_prediction']
            y_proba_0 = API_data['predict_proba_0']
            y_proba_1 = API_data['predict_proba_1']
            
        st.info('Prediction du client : ' + str(y_proba_1) + ' %')
        
       
        seuil_risque = 0.5  # le seuil de risque
        
        client_prediction = st.progress(0)
        for percent_complete in range(int(float(y_proba_1) * 100)):
            time.sleep(0.01)

        client_prediction.progress(percent_complete + 1)
        
        if float(y_proba_1) < seuil_risque:
            st.success('Client solvable')
        else:
            st.error('Client non solvable')
        
        # Extraire les détails du client à partir du dataframe
        
        st.subheader("Tous les détails du client :")
        st.write(df_clients.loc[df_clients.index== client_id])
  
        

### ----------------------- Prédiction d'un client ----------------


def file_selector(folder_path='.'):
    filenames = os.listdir(folder_path)
    selected_filename = st.selectbox('Sélectionner un fichier client', filenames)
    return os.path.join(folder_path, selected_filename)

def show_client_prediction():
    st.subheader("Sélectionner source des données du client")
    selected_choice = st.radio("",('Client existant dans le dataset','Nouveau client'))

    if selected_choice == 'Client existant dans le dataset':
        client_id =st.number_input("Donnez Id du Client",step=1)
        if st.button('Prédire Client'):
            # Configuration de l'API 
            API_URL = "https://projetcloud-181a7c4bddfe.herokuapp.com/predict/"+ str(client_id)
            with st.spinner('Chargement des clients...'):
                response = requests.get(API_URL)
                API_data = response.json()
                y_pred = API_data['retour_prediction']
                y_proba_0 = API_data['predict_proba_0']
                y_proba_1 = API_data['predict_proba_1']
                val = float(y_proba_0) * 100
                score_min = 50
        
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
        
                st.plotly_chart(fig)
    
    elif selected_choice == 'Nouveau client':   
        filename = file_selector()
        st.write('Fichier du nouveau client sélectionné `%s`' % filename)
        
        if st.button('Prédire Client'):
            nouveau_client = pd.read_csv(filename)
            with st.spinner('Chargement du score du client...'):
                    response = requests.get(API_url)
                    API_data = response.json()
                    y_pred = API_data['retour_prediction']
                    y_proba_0 = API_data['predict_proba_0']
                    y_proba_1 = API_data['predict_proba_1']
                    val = float(response.json()["predict_proba_0"]) * 100
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
        
                    st.plotly_chart(fig)


# --------------------------------------------------------------------
# FACTEURS D'INFLUENCE : SHAP VALUE
# --------------------------------------------------------------------
    
def affiche_facteurs_influence():
    ''' Affiche les facteurs d'influence du client courant
    '''
    html_facteurs_influence="""
        <div class="card">
            <div class="card-body" style="border-radius: 10px 10px 0px 0px;
                  background: #DEC7CB; padding-top: 5px; width: auto;
                  height: 40px;">
                  <h3 class="card-title" style="background-color:#DEC7CB; color:Crimson;
                      font-family:Georgia; text-align: center; padding: 0px 0;">
                      Variables importantes
                  </h3>
            </div>
        </div>
        """
    
    # ====================== GRAPHIQUES COMPARANT CLIENT COURANT / CLIENTS SIMILAIRES ===========================
    
    if st.sidebar.checkbox("Voir facteurs d\'influence"):     
        
        st.markdown(html_facteurs_influence, unsafe_allow_html=True)

        with st.spinner('**Affiche les facteurs d\'influence du client courant...**'):                 
                        
            data.drop(columns='TARGET', inplace=True)
                
            explainer = shap.TreeExplainer(best_model)
            id_input = st.number_input("Donnez Id du Client")

            if id_input in data.index:
                client_index = data.index.get_loc(id_input)
              
                X_test_courant =data.iloc[client_index]
                X_test_courant_array = X_test_courant.values.reshape(1, -1)
                
                shap_values_courant = explainer.shap_values(X_test_courant_array)
                
                col1, col2 = st.columns([1, 1])
                # BarPlot du client courant
                with col1:
                    plt.clf()
                    # BarPlot du client courant
                    shap.summary_plot(shap_values_courant[1], X_test_courant.to_frame().T, plot_type="bar")
                    fig = plt.gcf()
                    fig.set_size_inches((10, 20))
                    # Plot the graph on the dashboard
                    st.pyplot(fig)
     
                # Décision plot du client courant
                with col2:
                    plt.clf()
                    # Décision Plot
                    shap.decision_plot(explainer.expected_value[1], shap_values_courant[1], X_test_courant.to_frame().T)
                    fig2 = plt.gcf()
                    fig2.set_size_inches((10, 15))
                    # Plot the graph on the dashboard
                    st.pyplot(fig2)
                          
            else:
                st.error("L'identifiant fourni n'existe pas dans les données.")

# ====================================================================
# HEADER - TITRE
# ====================================================================
html_header = """
    <head>
        <title>Implémentez un modèle de scoring</title>
        <meta charset="utf-8">
        <meta name="keywords" content="Home Crédit Group, Dashboard, prêt, crédit score">
        <meta name="description" content="Application de Crédit Score - dashboard">
        <meta name="author" content="Hassine Hanen">
        <meta name="viewport" content="width=device-width, initial-scale=1">
    </head>             
    <h1 style="font-size:350%; color:#4B0082; font-family:'Segoe UI'; text-align:center;"> 
        Prêt à dépenser <br>
        <h2 style="color:#8A2BE2; font-family:'Segoe UI'; text-align:center;"> 
            DASHBOARD
        </h2>
        <hr style="display: block; margin-top: 0.5em; margin-bottom: 0.5em; margin-left: auto; margin-right: auto; border-style: inset; border-width: 2px;"/>
    </h1>
"""
# STYLE PERSONNALISÉ POUR LE SIDEBAR
st.markdown("""
    <style>
        .sidebar .sidebar-content {
            background-color: #4B0082;  /* Couleur de fond du menu */
            color: white;              /* Couleur du texte */
        }
        
        /* Bordures arrondies pour le radio-button */
        .stRadio > div[role="listitem"] > div {
            border-radius: 15px;
            background-color: #8A2BE2;  /* Couleur de fond des options non sélectionnées */
        }

        .stRadio > div[role="listitem"] > div[aria-checked="true"] {
            background-color: #FF4500;  /* Couleur de fond de l'option sélectionnée */
        }

        /* Style du slider */
        .stSlider > div > div > div:nth-child(2) {
            background-color: #FF4500;  /* Couleur du slider */
        }
    </style>
""", unsafe_allow_html=True)

#st.set_page_config(page_title="Prêt à dépenser - Dashboard", page_icon="", layout="wide")
st.markdown('<style>body{background-color: #fbfff0}</style>',unsafe_allow_html=True)
st.markdown(html_header, unsafe_allow_html=True)

# Cacher le bouton en haut à droite
st.markdown(""" <style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
</style> """, unsafe_allow_html=True)

# Suppression des marges par défaut
padding = 1
st.markdown(f""" <style>
    .reportview-container .main .block-container{{
        padding-top: {padding}rem;
        padding-right: {padding}rem;
        padding-left: {padding}rem;
        padding-bottom: {padding}rem;
    }} </style> """, unsafe_allow_html=True)


# ====================================================================
# MENUS
# ====================================================================


st.sidebar.title("Menus")
sidebar_selection = st.sidebar.radio(
    'Select Menu:',
    ['Data Analysis', 'Model & Prediction','Prédire solvabilité client','Intéprétabilité'],
)


if sidebar_selection == 'Data Analysis':
    selected_item = st.sidebar.selectbox('Select Menu:', 
                                ('Graphs', 'Distributions'))

if sidebar_selection == 'Model & Prediction':
    selected_item = "Prediction"

if sidebar_selection == 'Prédire solvabilité client':
    selected_item="predire_client"

if sidebar_selection == 'Intéprétabilité':
    selected_item =""
    affiche_facteurs_influence()
    

seuil_risque = st.sidebar.slider("Seuil de Solvabilité", min_value=0.0, max_value=1.0, value=0.5, step=0.01)

if selected_item == 'Data':
    show_data ()  

if selected_item == 'Solvency':
    show_overview ()  

if selected_item == 'Graphs':
    # Demander à l'utilisateur de saisir l'identifiant du client
    client_id =st.number_input("Entrez l'identifiant du client",step=1)
    
    # hist_graph()
    is_educ_selected, is_statut_selected, is_income_selected = filter_graphs()
    
    if (is_educ_selected == "oui"):
        education_type(client_id)
    
    if (is_statut_selected == "oui"):
        statut_plot(client_id)
    
    if (is_income_selected == "oui"):
        income_type(client_id)


if selected_item == 'Distributions':
    # Demander à l'utilisateur de saisir l'identifiant du client
    client_id = st.number_input("Entrez l'identifiant du client",step=1)
    
    is_age_selected,is_incomdis_selected = filter_distribution()
    if(is_age_selected=="oui"):
        age_distribution(client_id)
    if(is_incomdis_selected=="oui"):
        revenu_distribution(client_id)

    
    

if selected_item == 'Prediction':
    show_client_predection()

if selected_item == 'Model':
    show_model_analysis()

if selected_item == 'predire_client':
    show_client_prediction()

if selected_item == 'Fluance':
    affiche_facteurs_influence()
        
    

# ====================================================================
# FOOTER
# ====================================================================
html_line="""
<br>
<br>
<br>
<br>
<hr style= "  display: block;
  margin-top: 0.5em;
  margin-bottom: 0.5em;
  margin-left: auto;
  margin-right: auto;
  border-style: inset;
  border-width: 1.5px;">
<p style="color:Gray; text-align: right; font-size:12px;">Auteur : hhcine@yahoo.fr - 19/10/2023</p>
"""
st.markdown(html_line, unsafe_allow_html=True)
    
    
    
    
    
