import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier

st.write("""
# Penguin Prediction App

This app predicts the **Palmer Penguin** species!

Data obtained from the [Palmer Penguins](https://allisonhorst.github.io/palmerpenguins/) project.
""")

st.sidebar.header('User Input Features')

test={
    'island': ['Biscoe'],
    'bill_length_mm': [43.9],
    'bill_depth_mm': [17.2],
    'flipper_length_mm': [201.0],
    'body_mass_g': [4207.0],
    'sex': ['male']
}
example = pd.DataFrame(test)

#Obtiene los inputs
uploaded_file = st.sidebar.file_uploader("Upload your CSV File", type=['csv'])

if uploaded_file is not None:
    input_df = pd.read_csv(uploaded_file)

else:
    def user_input():
        island = st.sidebar.selectbox('Island', ('Biscoe','Dream','Torgersen'))
        sex = st.sidebar.selectbox('Sex',('male', 'female'))
        bill_length_mm = st.sidebar.slider('Bill lenght (mm)',32.1,59.6,43.9)
        bill_depth_mm = st.sidebar.slider('bill depth (mm)',13.1,21.5,17.2)
        flipper_length_mm = st.sidebar.slider('Flipper lenght (mm)',172.0,231.0,201.0)
        body_mass_g = st.sidebar.slider('Body mass (g)',2700.0,6300.0,4207.0)

        data = {
            'island': island,
            'bill_length_mm': bill_length_mm,
            'bill_depth_mm': bill_depth_mm,
            'flipper_length_mm': flipper_length_mm,
            'body_mass_g': body_mass_g,
            'sex': sex
        }

        features = pd.DataFrame(data,index=[0])
        return features
    
    input_df = user_input()

#combina los inputs con el dataset
penguins_raw = pd.read_csv('Penguins.csv')
penguins = penguins_raw.drop(columns=['species'])
df = pd.concat([input_df,penguins],axis=0)


encode = ['sex','island']
for col in encode:
    dummy = pd.get_dummies(df[col],prefix=col)
    df = pd.concat([df,dummy],axis=1)
    del df[col]

#selecciona solo la primer linea del input del usuario
df = df[:1]

#muestra los input
st.subheader('User Input Features')

if uploaded_file is not None:
    st.write(df)
else:
    st.write('Awaiting CSV file to be uploaded. Currently using example input parameters (shown below)')
    st.write(df)
    st.table(example)

# lee el modelo
load_clf = pickle.load(open('penguins_clf.pkl','rb'))

# realiza predicciones
prediction = load_clf.predict(df)
prediction_proba = load_clf.predict_proba(df)

st.subheader('Prediction')
penguins_species = np.array(['Adeline','Chinstrap','Gentoo'])
st.write(penguins_species[prediction])

st.subheader('Prediction Probability')
df = pd.DataFrame(prediction_proba, columns=(penguins_species))
st.write(df)
