import streamlit as st
import pandas as pd
import pickle

# Load the trained model pipeline
try:
    with open('best_food_price_model.pkl', 'rb') as f:
        pipe = pickle.load(f)
except Exception as e:
    st.error(f"❌ Failed to load model: {e}")

# App title
st.title("Food Price Prediction (2017)")
st.write("Enter the country and its global rank to estimate the 2017 food price.")

# Input fields
country = st.text_input("Enter Country Name")
global_rank = st.number_input("Enter Global Rank", min_value=1, step=1)

# Predict button
if st.button("Predict Price in 2017"):
    try:
        # Create a DataFrame with the user input
        df = pd.DataFrame([[country, global_rank]], columns=['Country', 'Global_Rank'])

        # Make prediction
        result = pipe.predict(df)

        st.success(f"📊 Estimated Price in 2017: ₹{round(result[0], 2)}")
    except Exception as e:
        st.error(f"Prediction failed: {e}")