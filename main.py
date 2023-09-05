import streamlit as st
import numpy as np
from scipy.stats import boxcox
import joblib
import plotly.graph_objs as go
import pandas as pd
import sklearn
from PIL import Image
from pathlib import Path
import openpyxl

st.set_page_config(
        page_title="eldery_age_app",
)


st.header('**APPLICATION du travail : "ESTIMATION DE L’AGE AU DECES CHEZ LES SUJETS DE PLUS DE 50 ANS : UNE APPROCHE ANTHROPOLOGIQUE NOVATRICE"**')
st.markdown('---')

st.sidebar.markdown('''
- <h2><a href="#application" style="scroll-behavior: smooth;">Application</a></h2>
- <h2><a href="#mesures" style="scroll-behavior: smooth;">Mesures</a></h2>
- <h2><a href="#resultats" style="scroll-behavior: smooth;">Résultats et metriques d'erreur</a></h2>
- <h2><a href="#visualisation-du-modele" style="scroll-behavior: smooth;">Visualisation du modèle</a></h2>
- <h2><a href="#auteurs" style="scroll-behavior: smooth;">Auteurs</a></h2>
- <h2><a href="#contact" style="scroll-behavior: smooth;">Contact</a></h2>
''', unsafe_allow_html=True)


st.subheader('Application')
container1= st.container()
# Charger les modèles
modele_folder = Path(__file__).parent / "modeles"

femur_model = joblib.load(modele_folder / "modele_femur.joblib")
colonne_model = joblib.load(modele_folder / "modele_colonne.joblib")
age_model = joblib.load(modele_folder / "modele_age.joblib")

container1.write("Entrez les valeurs pour la prédiction")
left_col, right_col=container1.columns(2)

taille_femur = left_col.number_input("Taille du fémur en cm")
taille_colonne = left_col.number_input("Taille de la colonne vertébrale en cm")
densite_osseuse = left_col.number_input("Densité osseuse en UH")
right_col.empty()
pred = container1.empty()


if container1.button("Prédire"):
    taille_par_femur = femur_model.predict([[taille_femur]])
    taille_par_colonne = colonne_model.predict([[taille_colonne]])

    diff = taille_par_femur - taille_par_colonne
    diff = diff + 100  

    fitted_lambda=1.6619518970309919
    diff = (diff**fitted_lambda - 1) / fitted_lambda
    densite_osseuse = np.sqrt(densite_osseuse) 

    age_data = [[diff, densite_osseuse]] 

    # Prédiction de l'âge
    age_predit = age_model.predict(age_data)
    container1.write(f"La prédiction de l'âge est : {age_predit[0].round(2)}")

st.markdown('---')
st.subheader('Mesures')
container2= st.container()
# Charger les images en utilisant des chemins relatifs
image_folder = Path(__file__).parent / "images"

image1 = Image.open(image_folder/"mesure_femur.png")
image2 = Image.open(image_folder/"mesure_colonne.png")
image3 = Image.open(image_folder/"mesure_densite.png")

container2.write("Voici comment effectuer les mesures: ")
container2.image(image1, caption='mesure du fémur')
container2.write("- La taille du fémur est mesurée sur le fémur droit, en mesurant ligne droite entre le plus haut point et le plus bas")
container2.image(image2, caption='mesure de la colonne vertébrale')
container2.write("- La taille de la colonne vertébrale sera mesurée selon la ligne droite entre la première vertèbre cervicale C1 et la dernière vertèbre lombaire L5. Les images scannographiques étant compartimentées, la mesure est réalisée en 2 étapes, en additionnant la taille des vertèbres cervicales à la taille des vertèbres dorsales et lombaires. Mais vous pouvez effectuer la mesure en une seule fois, si vous le pouvez.")
container2.image(image3, caption='mesure de la densité osseuse')
container2.write("- La mesure de la densité est effectuée sur les quatre premières vertèbres lombaires et nous utilisons la densité la moins importante enregistrée de L1 à L4.")


st.markdown('---')

st.subheader('Resultats')
container3= st.container()
container3.write("Résultats et metriques d''erreur")

# Affichage des résultats sous forme de tableau
results_table = """
|              | R²   | MAE  | MSE   | Erreur maximale | MAPE    | Validation croisée | Coef1 | Coef2 | Ordonnée |
|--------------|------|------|-------|-----------------|---------|--------------------|-------|-------|----------|
|    Modèle    | 0.73 | 3.94 | 19.54 | 8.69            | 5.73%   | 0.68 ± 0.07        | 1.14  | -7.41 | 98.11    |
"""

container3.markdown(results_table, unsafe_allow_html=True)


st.markdown('---')

### modele

st.subheader('Visualisation du modele')
container4= st.container()

excel_path = Path(__file__).parent / "data/df_for_graph.xlsx"
df_for_graph = pd.read_excel(excel_path)

fig = go.Figure(data=[go.Scatter3d(
    x=df_for_graph['diff'],
    y=df_for_graph['densite_min'],
    z=df_for_graph['age'],
    mode='markers',
    marker=dict(
        size=5,
        color=df_for_graph['age'],
        colorscale='Viridis',
        opacity=0.8
    ),
     hovertext=df_for_graph.index
)])

# Créer une grille de coordonnées dans l'espace 3D
x_vals = np.linspace(df_for_graph['diff'].min(), df_for_graph['diff'].max(), 50)
y_vals = np.linspace(df_for_graph['densite_min'].min(), df_for_graph['densite_min'].max(), 50)
X, Y = np.meshgrid(x_vals, y_vals)
Z = np.zeros_like(X)

# Calculer les prédictions pour chaque point de la grille
for i in range(len(x_vals)):
    for j in range(len(y_vals)):
        Z[j,i] = age_model.predict([[X[j,i], Y[j,i]]])

# Tracer la surface correspondante
fig.add_trace(go.Surface(x=X, y=Y, z=Z, opacity=0.5))


fig.update_layout(scene=dict(
        xaxis_title='Diff',
        yaxis_title='Densite_min',
        zaxis_title='Age'
    ),)

container4.plotly_chart(fig)

st.markdown('---')

st.subheader('Auteurs')
container5= st.container()
container5.write("Resident Mohamed Kenani sous la direction du Docteur Marwa Boussaid, AHU au service de médecine légale du CHU Tahar Sfar à Mahdia, en Tunisie.")

with open("style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

st.markdown('---')

st.subheader('Contact')
container6= st.container()

contact_form = """
<form action="https://formsubmit.co/kenanimohamed19@gmail.com" method="POST">
    <input type="hidden" name="_captcha" value="false">
     <input type="text" name="name" required placeholder="Your name" required>
     <input type="email" name="email" required placeholder="Your email" required>
     <textarea name="message" placeholder="Your message here" required></textarea>
     <button type="submit">Send</button>
</form>"""
left_column, right_column=container6.columns(2)
left_column.markdown(contact_form, unsafe_allow_html=True)
right_column.empty()