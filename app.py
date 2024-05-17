import gradio as gr
import pandas as pd
import xgboost as xgb
import numpy as np
import pickle

# Load the model from a .pkl file
with open('xgboost-model.pkl', 'rb') as f:
    model = pickle.load(f)

def handle_outliers(df, colm):
    q1 = df[colm].quantile(0.25)
    q3 = df[colm].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - (1.5 * iqr)
    upper_bound = q3 + (1.5 * iqr)
    df[colm] = np.where(df[colm] > upper_bound, upper_bound, df[colm])
    df[colm] = np.where(df[colm] < lower_bound, lower_bound, df[colm])
    return df

def predict_death_event(age, anaemia, creatinine_phosphokinase, diabetes, ejection_fraction, high_blood_pressure,
                        platelets, serum_creatinine, serum_sodium, sex, smoking, time):
    data = pd.DataFrame({
        'age': [age],
        'anaemia': [anaemia],
        'creatinine_phosphokinase': [creatinine_phosphokinase],
        'diabetes': [diabetes],
        'ejection_fraction': [ejection_fraction],
        'high_blood_pressure': [high_blood_pressure],
        'platelets': [platelets],
        'serum_creatinine': [serum_creatinine],
        'serum_sodium': [serum_sodium],
        'sex': [sex],
        'smoking': [smoking],
        'time': [time]
    })
    # Convert 'sex' to numeric (ensure this aligns with how the model was trained)
    data['sex'] = data['sex'].map({'Male': 1, 'Female': 0})

    # Handle outliers
    for col in ['creatinine_phosphokinase', 'ejection_fraction', 'platelets', 'serum_creatinine', 'serum_sodium']:
        handle_outliers(data, col)

    # Prediction
    prediction = model.predict(data)
    return 'Survival' if prediction[0] == 0 else 'Death'

iface = gr.Interface(
    fn=predict_death_event,
    inputs=[
        gr.Number(label="Age"), gr.Checkbox(label="Anaemia"), gr.Number(label="Creatinine Phosphokinase"),
        gr.Checkbox(label="Diabetes"), gr.Number(label="Ejection Fraction"), gr.Checkbox(label="High Blood Pressure"),
        gr.Number(label="Platelets"), gr.Number(label="Serum Creatinine"), gr.Number(label="Serum Sodium"),
        gr.Radio(choices=["Male", "Female"], label="Sex"), gr.Checkbox(label="Smoking"),
        gr.Number(label="Time since Last Event")
    ],
    outputs=gr.Textbox(label="Prediction"),
    title="Patient Survival Prediction",
    description="Predict survival of patient with heart failure, given their clinical record.",
    allow_flagging = 'never'
)

if __name__ == "__main__":
    iface.launch(server_name="0.0.0.0", server_port = 8001)