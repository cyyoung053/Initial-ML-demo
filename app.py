import streamlit as st
import pandas as pd
import joblib
from sklearn.datasets import load_iris
iris = load_iris()


st.title("🌸 Iris Classifier")

# Load model
model = joblib.load("iris_model.joblib")

# Sidebar inputs
st.sidebar.header("Input Features")
def user_input():
    sepal_length = st.sidebar.slider("Sepal length", 4.0, 8.0, 5.4)
    sepal_width  = st.sidebar.slider("Sepal width", 2.0, 4.5, 3.4)
    petal_length = st.sidebar.slider("Petal length", 1.0, 7.0, 1.3)
    petal_width  = st.sidebar.slider("Petal width", 0.1, 2.5, 0.2)
    return pd.DataFrame({
        "sepal length (cm)": [sepal_length],
        "sepal width (cm)":  [sepal_width],
        "petal length (cm)": [petal_length],
        "petal width (cm)":  [petal_width]
    })

df = user_input()
st.subheader("Features")
st.write(df)

# Predict
pred = model.predict(df)[0]
st.subheader("Prediction")
st.write(iris.target_names[pred])
