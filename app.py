# app_dashboard.py
import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix
import altair as alt
# -------------------------------
# Load Model
# -------------------------------
best_model = joblib.load("best_air_quality_model.pkl")

# لو استخدمت scaler
scaler = joblib.load("scaler.pkl")

# -------------------------------
# Dashboard Layout
# -------------------------------
st.set_page_config(page_title="Air Quality Dashboard", layout="wide")
st.title("🌿 Air Quality Prediction Dashboard")

# -------------------------------
# Sidebar - User Inputs
# -------------------------------
st.sidebar.header("Enter Feature Values")

temperature = st.sidebar.number_input("Temperature (°C)", value=25.0)
humidity = st.sidebar.number_input("Humidity (%)", value=60.0)
pm25 = st.sidebar.number_input("PM2.5", value=10.0)
pm10 = st.sidebar.number_input("PM10", value=20.0)
no2 = st.sidebar.number_input("NO2", value=20.0)
so2 = st.sidebar.number_input("SO2", value=5.0)
co = st.sidebar.number_input("CO", value=1.0)
proximity = st.sidebar.number_input("Proximity to Industrial Areas", value=5.0)
population = st.sidebar.number_input("Population Density", value=300)

input_df = pd.DataFrame({
    'Temperature':[temperature],
    'Humidity':[humidity],
    'PM2.5':[pm25],
    'PM10':[pm10],
    'NO2':[no2],
    'SO2':[so2],
    'CO':[co],
    'Proximity_to_Industrial_Areas':[proximity],
    'Population_Density':[population]
})

# -------------------------------
# Predict Button
# -------------------------------
if st.sidebar.button("Predict Air Quality"):
    prediction = best_model.predict(input_df)[0]
    st.subheader("Predicted Air Quality 🌟")
    st.success(f"{prediction}")

    # Probabilities
    if hasattr(best_model, "predict_proba"):
        probs = best_model.predict_proba(input_df)[0]
        classes = best_model.classes_
        prob_df = pd.DataFrame({'Air Quality': classes, 'Probability': probs})
        st.subheader("Prediction Probabilities")
        st.bar_chart(prob_df.set_index('Air Quality'))

# -------------------------------
# Feature Importance
# -------------------------------
if hasattr(best_model, "feature_importances_"):
    st.subheader("Feature Importance")
    feat_imp = pd.Series(best_model.feature_importances_, index=input_df.columns)
    feat_imp.sort_values().plot(kind='barh', figsize=(8,4), color='purple')
    st.pyplot(plt.gcf())
    plt.clf()

# -------------------------------
# Average Pollutants by Class
# -------------------------------
# Load dataset if available
try:
    df = pd.read_csv("updated_pollution_dataset.csv")  # Change filename to match your dataset
    # st.subheader("Average Pollutants by Air Quality Level")
    #avg_pollutants = df.groupby("Air Quality")[["PM2.5","PM10","NO2","SO2","CO"]].mean()
    # st.bar_chart(avg_pollutants)
    if 'df' in globals():
        st.subheader("Average Pollutants by Air Quality Level")

        avg_pollutants = df.groupby("Air Quality")[["PM2.5","PM10","NO2","SO2","CO"]].mean()
        avg_pollutants = avg_pollutants.reset_index().melt(id_vars="Air Quality",
                                                        var_name="Pollutant",
                                                        value_name="Average Value")

        chart = alt.Chart(avg_pollutants).mark_bar().encode(
            x='Pollutant:N',
            y='Average Value:Q',
            color='Air Quality:N',
            column='Air Quality:N'  
    ).properties(width=100, height=300)

    st.altair_chart(chart, use_container_width=True)
except FileNotFoundError:
    st.info("Dataset file not found. Please ensure 'updated_pollution_dataset.csv' is in the same directory.")
except Exception as e:
    st.warning(f"Could not load dataset: {e}")

# -------------------------------
# Confusion Matrix
# -------------------------------
try:
    if 'X_test' in globals() and 'y_test' in globals():
        st.subheader("Confusion Matrix on Test Set")
        y_pred_test = best_model.predict(X_test)
        cm = pd.DataFrame(confusion_matrix(y_test, y_pred_test), 
                          index=best_model.classes_, 
                          columns=best_model.classes_)
        
        plt.figure(figsize=(6,5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Purples', square=True)
        st.pyplot(plt.gcf())
        plt.clf()
except NameError:
    st.warning("Test data (X_test, y_test) not available. Please load your test dataset to display confusion matrix.")
