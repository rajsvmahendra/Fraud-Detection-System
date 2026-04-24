import streamlit as st
import pandas as pd
import joblib

@st.cache_data
def load_data():
    """
    Load the credit card dataset and cache it.
    This prevents reloading the 150MB file on every page reload or interaction.
    """
    return pd.read_csv("creditcard.zip", compression='zip')

@st.cache_resource
def load_model():
    """
    Load the trained machine learning model and cache it globally.
    This prevents reloading the 3.5MB pickle file on every interaction.
    """
    return joblib.load("fraud_model.pkl")
