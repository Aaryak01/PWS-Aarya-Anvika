import streamlit as st
import joblib
import numpy as np
import pandas as pd

# ==================================================
# LOAD SAVED MODELS
# (these .pkl files must be in the same folder as this app.py)
# ==================================================

lr_standard  = joblib.load('lr_standard.pkl')
lr_balanced  = joblib.load('lr_balanced.pkl')
rf_model     = joblib.load('rf_model.pkl')
nn_model     = joblib.load('nn_model.pkl')
label_encoder = joblib.load('label_encoder.pkl')

# ==================================================
# PAGE SETUP
# ==================================================

st.set_page_config(page_title="Pregnancy Risk Prediction Tool", layout="centered")
st.title("Pregnancy Risk Prediction — Model Comparison Tool")
st.write(
    "Enter a patient's clinical values below to compare predictions "
    "across four machine learning models trained for this research."
)

# ==================================================
# INPUT FORM
# ==================================================

st.header("Patient Information")

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", min_value=10, max_value=60, value=28)
    systolicbp = st.number_input("Systolic Blood Pressure", min_value=60, max_value=220, value=120)
    diastolicbp = st.number_input("Diastolic Blood Pressure", min_value=40, max_value=140, value=80)
    bs = st.number_input("Blood Sugar (bs)", min_value=2.0, max_value=25.0, value=6.5, step=0.1)
    bodytemp = st.number_input("Body Temperature (°F)", min_value=95.0, max_value=105.0, value=98.6, step=0.1)
    heartrate = st.number_input("Heart Rate", min_value=40, max_value=160, value=75)
    bmi = st.number_input("BMI", min_value=12.0, max_value=50.0, value=24.0, step=0.1)

with col2:
    previous_complications = st.selectbox("Previous Complications", ["No", "Yes"])
    preexisting_diabetes = st.selectbox("Preexisting Diabetes", ["No", "Yes"])
    gestational_diabetes = st.selectbox("Gestational Diabetes", ["No", "Yes"])
    mental_health = st.selectbox("Mental Health Concerns", ["No", "Yes"])

# ==================================================
# PREDICT
# ==================================================

if st.button("Predict Risk", type="primary"):

    input_data = pd.DataFrame([{
        "age": age,
        "systolicbp": systolicbp,
        "diastolicbp": diastolicbp,
        "bs": bs,
        "bodytemp": bodytemp,
        "heartrate": heartrate,
        "bmi": bmi,
        "previous_complications": 1.0 if previous_complications == "Yes" else 0.0,
        "preexisting_diabetes": 1.0 if preexisting_diabetes == "Yes" else 0.0,
        "gestational_diabetes": 1.0 if gestational_diabetes == "Yes" else 0.0,
        "mental_health": 1.0 if mental_health == "Yes" else 0.0,
    }])

    # Each saved model is a full pipeline (preprocessing + classifier),
    # so we can pass the raw input_data directly into .predict()

    pred_lr_standard = lr_standard.predict(input_data)[0]
    pred_lr_balanced = lr_balanced.predict(input_data)[0]
    pred_rf          = rf_model.predict(input_data)[0]

    # Neural network was trained on label-encoded targets (0/1/2),
    # so we decode its prediction back to "high"/"low"/"mid"
    pred_nn_encoded = nn_model.predict(input_data)[0]
    pred_nn = label_encoder.inverse_transform([pred_nn_encoded])[0]

    proba_lr_standard = lr_standard.predict_proba(input_data)[0]
    proba_lr_balanced = lr_balanced.predict_proba(input_data)[0]
    proba_rf          = rf_model.predict_proba(input_data)[0]
    proba_nn          = nn_model.predict_proba(input_data)[0]

    # Get class order for each model (sklearn stores this on .classes_)
    classes_lr_standard = lr_standard.classes_
    classes_lr_balanced = lr_balanced.classes_
    classes_rf          = rf_model.classes_
    classes_nn          = label_encoder.inverse_transform(nn_model.classes_)

    def confidence_for(pred, classes, proba):
        idx = list(classes).index(pred)
        return proba[idx]

    conf_lr_standard = confidence_for(pred_lr_standard, classes_lr_standard, proba_lr_standard)
    conf_lr_balanced = confidence_for(pred_lr_balanced, classes_lr_balanced, proba_lr_balanced)
    conf_rf          = confidence_for(pred_rf, classes_rf, proba_rf)
    conf_nn          = confidence_for(pred_nn, classes_nn, proba_nn)

    # ==================================================
    # DISPLAY RESULTS
    # ==================================================

    st.header("Results")

    results_df = pd.DataFrame({
        "Model": [
            "Standard Logistic Regression",
            "Balanced Logistic Regression",
            "Random Forest",
            "Neural Network",
        ],
        "Predicted Risk": [
            pred_lr_standard, pred_lr_balanced, pred_rf, pred_nn
        ],
        "Confidence": [
            f"{conf_lr_standard*100:.1f}%",
            f"{conf_lr_balanced*100:.1f}%",
            f"{conf_rf*100:.1f}%",
            f"{conf_nn*100:.1f}%",
        ],
    })

    st.table(results_df)

    predictions = [pred_lr_standard, pred_lr_balanced, pred_rf, pred_nn]
    if len(set(predictions)) > 1:
        st.warning(
            "The models do not agree on this patient's risk level. "
            "This kind of disagreement tends to happen most often on "
            "borderline / mid-risk cases, which reflects the class-imbalance "
            "challenge discussed in this research."
        )
    else:
        st.success("All four models agree on this patient's risk level.")

    st.subheader("Confidence Comparison")
    chart_df = pd.DataFrame({
        "Model": results_df["Model"],
        "Confidence (%)": [
            conf_lr_standard*100, conf_lr_balanced*100, conf_rf*100, conf_nn*100
        ],
    }).set_index("Model")
    st.bar_chart(chart_df)

st.caption(
    "This tool is a proof-of-concept demonstration built for this research project. "
    "It is not validated for clinical use and should not be used to make real medical decisions."
)
