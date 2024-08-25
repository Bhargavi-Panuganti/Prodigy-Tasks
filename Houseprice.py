import streamlit as st
st.set_page_config(page_title="House Price Predictor", page_icon="🏠", layout="wide")
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Load your dataset
# df = pd.read_csv('your_dataset.csv')

# For example purposes, let's create a dummy dataset
# Replace this with your actual dataset loading code
data = {
    'Area': [1000, 1500, 2000, 2500, 3000],
    'Price': [200000, 300000, 400000, 500000, 600000]
}
df = pd.DataFrame(data)

# Train your model
X = df[['Area']]
y = df['Price']
model = LinearRegression()
model.fit(X, y)

# Streamlit App
st.title("House Price Prediction")
st.set_page_config(page_title="House Price Predictor", layout="wide")

# User input for area
area = st.number_input("Enter the area in square feet:", min_value=0, max_value=10000, value=1500)

# Predict the price
if st.button("Predict"):
    prediction = model.predict(np.array([[area]]))
    st.write(f"The predicted price for a {area} sqft house is Rs{prediction[0]:,.2f}")

