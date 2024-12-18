import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import streamlit as st
from PIL import Image

df = pd.read_csv('assets/data/heart_attack_prediction_dataset.csv')

df.drop(columns='Patient ID', inplace=True)

numerical_attributes = df.select_dtypes(include=[np.number]).columns
categorical_attributes = df.select_dtypes(include=['object']).columns

numerical_binaries = []
numerical_nonbinaries = []

for i in numerical_attributes:
    if df[i].nunique() == 2:
        numerical_binaries.append(i)
    else:
        numerical_nonbinaries.append(i)

df[['Systolic_BP', 'Diastolic_BP']] = df['Blood Pressure'].str.split('/', expand=True)
df['Systolic_BP'] = pd.to_numeric(df['Systolic_BP'])
df['Diastolic_BP'] = pd.to_numeric(df['Diastolic_BP'])
df = df.drop('Blood Pressure', axis=1)

X = df.drop(columns=['Heart Attack Risk'])
y = df['Heart Attack Risk']

categorical_attributes = X.select_dtypes(include=['object']).columns
for column in categorical_attributes:
    le = LabelEncoder()
    X[column] = le.fit_transform(X[column])

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

st.set_page_config(page_title="Heart Attack Prediction", page_icon=":heart:", layout="wide", initial_sidebar_state="expanded")

logo = Image.open('assets/images/logo.png')

st.sidebar.image(logo, use_column_width=True)
st.sidebar.title("Navigation")
nav_selection = st.sidebar.radio("Go to", ["Home", "Predict Heart Attack Risk", "Field Descriptions"])

if nav_selection == "Home":
    st.title("Welcome to the Heart Attack Prediction App")
    st.write("This app uses machine learning to predict the risk of a heart attack based on various health and lifestyle factors.")
    st.write("To get started, select 'Predict Heart Attack Risk' from the navigation menu.")

elif nav_selection == "Predict Heart Attack Risk":
    st.title("Heart Attack Risk Prediction")

    st.subheader("Enter Patient Information")

    with st.form(key="patient_info_form"):
        age = st.number_input("Age", min_value=18, max_value=120, step=1, help="The patient's age in years.")
        sex = st.selectbox("Sex", ["Male", "Female"], help="The patient's biological sex.")
        cholesterol = st.number_input("Cholesterol (mg/dL)", min_value=0.0, max_value=500.0, step=1.0, help="The patient's total cholesterol level in milligrams per deciliter.")
        systolic_bp = st.number_input("Systolic Blood Pressure (mmHg)", min_value=80, max_value=200, step=1, help="The patient's systolic blood pressure in millimeters of mercury.")
        diastolic_bp = st.number_input("Diastolic Blood Pressure (mmHg)", min_value=50, max_value=120, step=1, help="The patient's diastolic blood pressure in millimeters of mercury.")
        heart_rate = st.number_input("Heart Rate (bpm)", min_value=40, max_value=200, step=1, help="The patient's resting heart rate in beats per minute.")
        diabetes = st.selectbox("Diabetes", ["No", "Yes"], help="Whether the patient has been diagnosed with diabetes.")
        family_history = st.selectbox("Family History of Heart Disease", ["No", "Yes"], help="Whether the patient has a family history of heart disease.")
        smoking = st.selectbox("Smoking", ["No", "Yes"], help="Whether the patient currently smokes or has a history of smoking.")
        obesity = st.selectbox("Obesity", ["No", "Yes"], help="Whether the patient is considered obese based on their body mass index (BMI).")
        alcohol = st.selectbox("Alcohol Consumption", ["No", "Yes"], help="Whether the patient regularly consumes alcoholic beverages.")
        exercise = st.number_input("Exercise Hours per Week", min_value=0.0, max_value=168.0, step=0.5, help="The number of hours per week the patient engages in physical exercise.")
        diet = st.selectbox("Healthy Diet", ["No", "Yes"], help="Whether the patient maintains a healthy, balanced diet.")
        previous_heart_problems = st.selectbox("Previous Heart Problems", ["No", "Yes"], help="Whether the patient has a history of previous heart problems or cardiovascular issues.")
        medication_use = st.selectbox("Medication Use", ["No", "Yes"], help="Whether the patient is currently taking any medications, including prescription drugs.")
        stress = st.selectbox("Stress Level", ["Low", "Moderate", "High"], help="The patient's perceived level of stress.")
        sedentary_hours = st.number_input("Sedentary Hours per Day", min_value=0.0, max_value=24.0, step=0.5, help="The number of hours per day the patient spends in sedentary activities.")
        income = st.number_input("Income (USD)", min_value=0, max_value=1000000, step=1000, help="The patient's annual household income in US dollars.")
        bmi = st.number_input("BMI", min_value=10.0, max_value=50.0, step=0.1, help="The patient's body mass index, calculated as weight in kilograms divided by height in meters squared.")
        triglycerides = st.number_input("Triglycerides (mg/dL)", min_value=0.0, max_value=1000.0, step=1.0, help="The patient's triglyceride level in milligrams per deciliter.")
        physical_activity = st.number_input("Physical Activity Days per Week", min_value=0, max_value=7, step=1, help="The number of days per week the patient engages in physical activity.")
        sleep_hours = st.number_input("Sleep Hours per Day", min_value=4, max_value=12, step=1, help="The number of hours per day the patient sleeps.")
        country = st.text_input("Country", help="The country where the patient resides.")
        continent = st.text_input("Continent", help="The continent where the patient's country is located.")
        hemisphere = st.text_input("Hemisphere", help="The hemisphere (Northern or Southern) where the patient's country is located.")

        predict_button = st.form_submit_button("Predict Heart Attack Risk")

    if predict_button:
        user_input = np.array([age, 1 if sex == "Male" else 0, cholesterol, systolic_bp, diastolic_bp, heart_rate, 1 if diabetes == "Yes" else 0, 1 if family_history == "Yes" else 0, 1 if smoking == "Yes" else 0, 1 if obesity == "Yes" else 0, 1 if alcohol == "Yes" else 0, exercise, 1 if diet == "Yes" else 0, 1 if previous_heart_problems == "Yes" else 0, 1 if medication_use == "Yes" else 0, 1 if stress == "High" else (0 if stress == "Low" else 0.5), sedentary_hours, income, bmi, triglycerides, physical_activity, sleep_hours, 1 if country == "USA" else (0 if country == "Canada" else 0.5), 1 if continent == "North America" else (0 if continent == "Europe" else 0.5), 1 if hemisphere == "Northern" else 0]).reshape(1, -1)
        prediction = model.predict(user_input)[0]

        if prediction == 1:
            st.session_state.prediction_result = "The patient is at high risk of a heart attack."
            st.session_state.prediction_color = "red"
        else:
            st.session_state.prediction_result = "The patient is at low risk of a heart attack."
            st.session_state.prediction_color = "green"

        st.markdown(
            f"""
            <div class="fixed top-0 left-0 w-screen h-screen bg-black bg-opacity-50 flex items-center justify-center z-50">
                <div class="bg-white rounded-lg shadow-lg p-8" style="max-width: 500px;">
                    <h2 class="text-2xl font-bold mb-4" style="color: {st.session_state.prediction_color};">{st.session_state.prediction_result}</h2>
                    <p class="mb-4">Based on the provided information, the model has predicted the patient's heart attack risk.</p>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

elif nav_selection == "Field Descriptions":
    st.title("Field Descriptions")
    st.write("This section provides information about the purpose of each input field in the 'Predict Heart Attack Risk' section.")

    st.subheader("Age")
    st.write("The patient's age in years.")

    st.subheader("Sex")
    st.write("The patient's biological sex.")

    st.subheader("Cholesterol (mg/dL)")
    st.write("The patient's total cholesterol level in milligrams per deciliter.")

    st.subheader("Systolic Blood Pressure (mmHg)")
    st.write("The patient's systolic blood pressure in millimeters of mercury.")

    st.subheader("Diastolic Blood Pressure (mmHg)")
    st.write("The patient's diastolic blood pressure in millimeters of mercury.")

    st.subheader("Heart Rate (bpm)")
    st.write("The patient's resting heart rate in beats per minute.")

    st.subheader("Diabetes")
    st.write("Whether the patient has been diagnosed with diabetes.")

    st.subheader("Family History of Heart Disease")
    st.write("Whether the patient has a family history of heart disease.")

    st.subheader("Smoking")
    st.write("Whether the patient currently smokes or has a history of smoking.")

    st.subheader("Obesity")
    st.write("Whether the patient is considered obese based on their body mass index (BMI).")

    st.subheader("Alcohol Consumption")
    st.write("Whether the patient regularly consumes alcoholic beverages.")

    st.subheader("Exercise Hours per Week")
    st.write("The number of hours per week the patient engages in physical exercise.")

    st.subheader("Healthy Diet")
    st.write("Whether the patient maintains a healthy, balanced diet.")

    st.subheader("Previous Heart Problems")
    st.write("Whether the patient has a history of previous heart problems or cardiovascular issues.")

    st.subheader("Medication Use")
    st.write("Whether the patient is currently taking any medications, including prescription drugs.")

    st.subheader("Stress Level")
    st.write("The patient's perceived level of stress.")

    st.subheader("Sedentary Hours per Day")
    st.write("The number of hours per day the patient spends in sedentary activities.")

    st.subheader("Income (USD)")
    st.write("The patient's annual household income in US dollars.")

    st.subheader("BMI")
    st.write("The patient's body mass index, calculated as weight in kilograms divided by height in meters squared.")

    st.subheader("Triglycerides (mg/dL)")
    st.write("The patient's triglyceride level in milligrams per deciliter.")

    st.subheader("Physical Activity Days per Week")
    st.write("The number of days per week the patient engages in physical activity.")

    st.subheader("Sleep Hours per Day")
    st.write("The number of hours per day the patient sleeps.")

    st.subheader("Country")
    st.write("The country where the patient resides.")

    st.subheader("Continent")
    st.write("The continent where the patient's country is located.")

    st.subheader("Hemisphere")
    st.write("The hemisphere (Northern or Southern) where the patient's country is located.")