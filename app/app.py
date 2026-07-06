import streamlit as st
st.set_page_config(page_title="Customer Insights Platform",layout="wide")

st.sidebar.title("Navigation")
page=st.sidebar.radio("Go to",["Insights Dashboard", "Churn Risk Checker", "Sales Forecast", "Review Analysis"])

if page == "Insights Dashboard":
    st.title("Business Insights Dashboard")

    import pandas as pd
    import matplotlib.pyplot as plt

    daily_revenue = pd.read_csv('data/daily_revenue.csv', index_col=0, parse_dates=True)

    st.subheader("Daily Revenue Trend")
    fig, ax = plt.subplots(figsize=(12,4))
    ax.plot(daily_revenue.index, daily_revenue.iloc[:,0])
    st.pyplot(fig)

elif page == "Churn Risk Checker":
    st.title("Customer Churn Risk Checker")

    import joblib
    model = joblib.load('models/churn_model.pkl')

    st.write("Enter customer details to estimate churn risk:")

    frequency = st.number_input("Number of orders placed", min_value=1, value=1)
    monetary = st.number_input("Total amount spent (R$)", min_value=0.0, value=100.0)
    avg_order_value = st.number_input("Average order value (R$)", min_value=0.0, value=100.0)
    avg_installments = st.number_input("Average payment installments", min_value=1.0, value=1.0)

    if st.button("Predict Churn Risk"):
        input_data = [[frequency, monetary, avg_order_value, avg_installments]]
        probability = model.predict_proba(input_data)[0][1]
        st.metric("Churn Probability", f"{probability*100:.1f}%")

        if probability > 0.5:
            st.error("High churn risk — consider a retention offer.")
        else:
            st.success("Low churn risk.")

elif page == "Sales Forecast":
    st.title("Sales Forecast")

    import joblib
    import numpy as np
    from tensorflow.keras.models import load_model
    import pandas as pd

    forecast_model = load_model('models/forecast_model.keras', compile=False)
    scaler = joblib.load('models/forecast_scaler.pkl')

    daily_revenue = pd.read_csv('data/daily_revenue.csv', index_col=0, parse_dates=True)
    last_30_days = daily_revenue.iloc[-30:].values.reshape(-1, 1)

    scaled_input = scaler.transform(last_30_days)
    scaled_input = scaled_input.reshape((1, 30, 1))

    prediction_scaled = forecast_model.predict(scaled_input)
    prediction = scaler.inverse_transform(prediction_scaled)

    st.metric("Predicted Next-Day Revenue", f"R$ {prediction[0][0]:,.2f}")

    st.line_chart(daily_revenue.iloc[-60:])

elif page == "Review Analysis":
    st.title("Review Sentiment & Ticket Classifier")

    import joblib
    import re

    sentiment_model = joblib.load('models/sentiment_model.pkl')
    vectorizer = joblib.load('models/tfidf_vectorizer.pkl')
    ticket_model = joblib.load('models/ticket_model.pkl')

    def clean_text(text):
        text = text.lower()
        text = re.sub(r'[^a-zA-ZáéíóúâêôãõçÁÉÍÓÚÂÊÔÃÕÇ\s]', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    user_text = st.text_area("Paste a customer review or complaint (Portuguese):")

    if st.button("Analyze"):
        cleaned = clean_text(user_text)
        vectorized = vectorizer.transform([cleaned])

        sentiment_pred = sentiment_model.predict(vectorized)[0]
        st.write(f"Sentiment: {sentiment_pred}")

        ticket_pred = ticket_model.predict(vectorized)[0]
        st.write(f"Likely Category: {ticket_pred}")