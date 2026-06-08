import streamlit as st
import pandas as pd
import joblib

model = joblib.load("best_model_random_forest.pkl")
model_columns = joblib.load("model_columns.pkl")

st.title("MSDE" , "Prédiction du Status de prêt- Modul Machine Learning ")

st.write("Application simple basée sur les variables les plus importantes.")

loan_amount = st.number_input("Loan Amount", value=116500.0)
rate_of_interest = st.number_input("Rate of Interest", value=3.99)
interest_rate_spread = st.number_input("Interest Rate Spread", value=0.3904)
upfront_charges = st.number_input("Upfront Charges", value=2596.45)
property_value = st.number_input("Property Value", value=118000.0)
income = st.number_input("Income", value=1740.0)
credit_score = st.number_input("Credit Score", value=758.0)
ltv = st.number_input("LTV", value=98.73)
dtir1 = st.number_input("DTIR1", value=45.0)

credit_type = st.selectbox("Credit Type", ["CRIF", "EXP"])

input_data = pd.DataFrame(columns=model_columns)
input_data.loc[0] = 0

# Valeurs numériques
values = {
    "loan_amount": loan_amount,
    "rate_of_interest": rate_of_interest,
    "Interest_rate_spread": interest_rate_spread,
    "Upfront_charges": upfront_charges,
    "property_value": property_value,
    "income": income,
    "Credit_Score": credit_score,
    "LTV": ltv,
    "dtir1": dtir1
}

for col, val in values.items():
    if col in input_data.columns:
        input_data[col] = val

# Valeurs fixes importantes pour éviter que tout reste à 0
fixed_values = {
    "term": 360,
    "credit_type_EXP": 1 if credit_type == "EXP" else 0,
    "credit_type_CRIF": 1 if credit_type == "CRIF" else 0,
    "business_or_commercial_nob/c": 1,
    "Neg_ammortization_not_neg": 1,
    "interest_only_not_int": 1,
    "lump_sum_payment_not_lpsm": 1,
    "construction_type_sb": 1,
    "occupancy_type_pr": 1,
    "submission_of_application_to_inst": 1,
    "Region_south": 1,
    "Security_Type_direct": 1
}

for col, val in fixed_values.items():
    if col in input_data.columns:
        input_data[col] = val

if st.button("Prédire"):
    prediction = model.predict(input_data)[0]
    proba = model.predict_proba(input_data)[0]

    st.subheader("Résultat")

    if prediction == 1:
        st.success("Résultat : Status = 1")
    else:
        st.info("Résultat : Status = 0")

    st.write(f"Probabilité Status 0 : {proba[0]*100:.2f}%")
    st.write(f"Probabilité Status 1 : {proba[1]*100:.2f}%")

    with st.expander("Données envoyées au modèle"):
        st.dataframe(input_data)