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


st.header('**Application of the Work: "ESTIMATION OF AGE AT DEATH IN SUBJECTS OVER 50 YEARS OLD: AN INNOVATIVE ANTHROPOLOGICAL APPROACH"**')
st.markdown('---')

st.sidebar.markdown('''
- <h2><a href="#application" style="scroll-behavior: smooth;">Application</a></h2>
- <h2><a href="#measurements" style="scroll-behavior: smooth;">Measurements</a></h2>
- <h2><a href="#results" style="scroll-behavior: smooth;">Results and Error Metrics</a></h2>
- <h2><a href="#model-visualization" style="scroll-behavior: smooth;">Model Visualization</a></h2>
- <h2><a href="#authors" style="scroll-behavior: smooth;">Authors</a></h2>
- <h2><a href="#contact" style="scroll-behavior: smooth;">Contact</a></h2>
''', unsafe_allow_html=True)


st.subheader('Application')
container1= st.container()
# Charger les modèles
modele_folder = Path(__file__).parent / "modeles"

femur_model = joblib.load(modele_folder / "modele_femur.joblib")
colonne_model = joblib.load(modele_folder / "modele_colonne.joblib")
age_model = joblib.load(modele_folder / "modele_age.joblib")

container1.write("Enter values for prediction")
left_col, right_col = container1.columns(2)

taille_femur  = left_col.number_input("Femur length in cm")
taille_colonne  = left_col.number_input("Vertebral column length in cm")
densite_osseuse  = left_col.number_input("Bone density in HU")
right_col.empty()
pred = container1.empty()


if container1.button("Predict"):
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
    container1.write(f"The age prediction is : {age_predit[0].round(2)}")

st.markdown('---')
st.subheader('Measurements')
container2= st.container()
# Charger les images en utilisant des chemins relatifs
image_folder = Path(__file__).parent / "images"

image1 = Image.open(image_folder/"mesure_femur.png")
image2 = Image.open(image_folder/"mesure_colonne.png")
image3 = Image.open(image_folder/"mesure_densite.png")

container2.write("Here's how to take the measurements:")
container2.image(image1, caption='Femur measurement')
container2.write("- Measure the length of the femur on the right femur by measuring a straight line from the highest point to the lowest point.")
container2.image(image2, caption='Vertebral column measurement')
container2.write("- The length of the vertebral column will be measured along a straight line from the first cervical vertebra C1 to the last lumbar vertebra L5. Since the scan images are compartmentalized, the measurement is done in two steps by adding the length of the cervical vertebrae to the thoracic and lumbar vertebrae. However, you can perform the measurement in one go if you are able to do so.")
container2.image(image3, caption='Bone density measurement')
container2.write("- Bone density measurement is taken from the first four lumbar vertebrae, using the lowest recorded density from L1 to L4.")


st.markdown('---')

st.subheader('Results')
container3= st.container()
container3.write("Results and Error Metrics")

# Affichage des résultats sous forme de tableau
results_table = """
|              | R²   | MAE  | MSE   | maximum error   | MAPE    | Cross Validation   | Coef1 | Coef2 | Intercept|
|--------------|------|------|-------|-----------------|---------|--------------------|-------|-------|----------|
|    Modèle    | 0.73 | 3.94 | 19.54 | 8.69            | 5.73%   | 0.68 ± 0.07        | 1.14  | -7.41 | 98.11    |
"""

container3.markdown(results_table, unsafe_allow_html=True)


st.markdown('---')

### modele

st.subheader('Model Visualization')
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

st.subheader('Authors')
container5= st.container()
container5.write("Resident Mohamed Kenani under the guidance of Dr. Marwa Boussaid, Assistant Hospitalier Universitaire (AHU) at the Forensic Medicine Department of CHU Tahar Sfar in Mahdia, Tunisia.")

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