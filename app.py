import streamlit as st
import pandas as pd
import joblib

# Load the trained machine learning model
try:
    model = joblib.load('heart_disease_predictor.pkl')
except FileNotFoundError:
    st.error("Model file not found. Please run model_training.py to generate the model.")
    st.stop()

# Set the page configuration
st.set_page_config(page_title="Heart Disease Predictor", page_icon="❤️", layout="wide")

# App title and description
st.title('❤️ Heart Disease Prediction App')
st.markdown("""
This application uses a Logistic Regression model to predict the likelihood of a patient having heart disease.
Please enter the patient's details in the sidebar to get a prediction.
*This is a tool for educational purposes and not a substitute for professional medical advice.*
""")

# Sidebar for user input
st.sidebar.header('Patient Input Features')

def user_input_features():
    """Collects user input features into a dataframe."""
    age = st.sidebar.slider('Age', 29, 77, 55)
    sex = st.sidebar.selectbox('Sex', ('Male', 'Female'))
    cp = st.sidebar.selectbox('Chest Pain Type (CP)', ('Typical Angina', 'Atypical Angina', 'Non-anginal Pain', 'Asymptomatic'))
    trestbps = st.sidebar.slider('Resting Blood Pressure (trestbps)', 94, 200, 132)
    chol = st.sidebar.slider('Serum Cholestoral in mg/dl (chol)', 126, 564, 246)
    fbs = st.sidebar.selectbox('Fasting Blood Sugar > 120 mg/dl (fbs)', ('False', 'True'))
    restecg = st.sidebar.selectbox('Resting Electrocardiographic Results (restecg)', ('Normal', 'ST-T wave abnormality', 'Left ventricular hypertrophy'))
    thalach = st.sidebar.slider('Maximum Heart Rate Achieved (thalach)', 71, 202, 149)
    exang = st.sidebar.selectbox('Exercise Induced Angina (exang)', ('No', 'Yes'))
    oldpeak = st.sidebar.slider('ST depression induced by exercise relative to rest (oldpeak)', 0.0, 6.2, 1.0)
    slope = st.sidebar.selectbox('Slope of the peak exercise ST segment', ('Upsloping', 'Flat', 'Downsloping'))
    ca = st.sidebar.selectbox('Number of major vessels colored by flourosopy (ca)', (0, 1, 2, 3, 4))
    thal = st.sidebar.selectbox('Thalassemia (thal)', ('Normal', 'Fixed defect', 'Reversable defect'))

    # Map categorical inputs to numerical values for the model
    sex_map = {'Male': 1, 'Female': 0}
    cp_map = {'Typical Angina': 0, 'Atypical Angina': 1, 'Non-anginal Pain': 2, 'Asymptomatic': 3}
    fbs_map = {'False': 0, 'True': 1}
    restecg_map = {'Normal': 0, 'ST-T wave abnormality': 1, 'Left ventricular hypertrophy': 2}
    exang_map = {'No': 0, 'Yes': 1}
    slope_map = {'Upsloping': 0, 'Flat': 1, 'Downsloping': 2}
    thal_map = {'Normal': 1, 'Fixed defect': 2, 'Reversable defect': 3}

    # Create a dictionary of the data
    data = {
        'age': age,
        'sex': sex_map[sex],
        'cp': cp_map[cp],
        'trestbps': trestbps,
        'chol': chol,
        'fbs': fbs_map[fbs],
        'restecg': restecg_map[restecg],
        'thalach': thalach,
        'exang': exang_map[exang],
        'oldpeak': oldpeak,
        'slope': slope_map[slope],
        'ca': ca,
        'thal': thal_map[thal]
    }
    features = pd.DataFrame(data, index=[0])
    return features

input_df = user_input_features()

# Display user input in the main area
st.subheader('Patient Input Summary')
st.write(input_df)

# Prediction button
if st.button('**Predict**'):
    # Ensure column order matches the model's training data
    feature_order = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']
    input_df = input_df[feature_order]

    # Make prediction
    prediction = model.predict(input_df)
    prediction_proba = model.predict_proba(input_df)

    st.subheader('Prediction Result')
    if prediction[0] == 1:
        st.error(f'**The model predicts a HIGH RISK of heart disease.**')
        st.write(f"Confidence: {prediction_proba[0][1]*100:.2f}%")
    else:
        st.success(f'**The model predicts a LOW RISK of heart disease.**')
        st.write(f"Confidence: {prediction_proba[0][0]*100:.2f}%")

    st.info("Please consult a medical professional for an accurate diagnosis.")
