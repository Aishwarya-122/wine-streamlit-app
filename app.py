import streamlit as st
import pandas as pd
from sklearn.datasets import load_wine
from sklearn.linear_model import LogisticRegression

# Load dataset
wine = load_wine()

# Convert to DataFrame
df = pd.DataFrame(wine.data, columns=wine.feature_names)
df['target'] = wine.target

# Train model
X = df.drop('target', axis=1)
y = df['target']

model = LogisticRegression(max_iter=200)
model.fit(X, y)

# Streamlit UI
st.title("🍷 Wine Classification App")

st.write("Enter the values to predict the wine class")

# User inputs
inputs = []
for feature in wine.feature_names:
    value = st.number_input(f"{feature}", float(df[feature].min()), float(df[feature].max()))
    inputs.append(value)

# Predict button
if st.button("Predict"):
    prediction = model.predict([inputs])
    st.success(f"Predicted Wine Class: {prediction[0]}")
