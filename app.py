import streamlit as st
import pandas as pd
import joblib

# Load the trained model and label encoders
model = joblib.load('gradient_boosting_model.pkl')
label_encoders = joblib.load('label_encoders.pkl')

# Load your dataset (used only to extract dropdown options)
df = pd.read_csv("dataset.csv")  # ← replace with your actual dataset file

# Strip column names in case of extra spaces
df.columns = df.columns.str.strip()

# Define the 8 required columns
feature_columns = [
    'Gender', 'Age', 'Department', 'Attendance (%)', 'Study_Hours_per_Week',
    'Family_Income_Level', 'Stress_Level (1-10)', 'Sleep_Hours_per_Night'
]

# Split categorical and numerical columns
categorical_cols = df[feature_columns].select_dtypes(include='object').columns.tolist()
numerical_cols = [col for col in feature_columns if col not in categorical_cols]

# Streamlit UI
st.title("🎓 Student Exam Pass/Fail Predictor")
st.write("Enter the student details to predict whether the student will **Pass** or **Fail** the exam.")

with st.form("student_form"):
    user_input = {}

    # Categorical input fields (dropdowns)
    for col in categorical_cols:
        options = sorted(df[col].dropna().unique().tolist())
        user_input[col] = st.selectbox(f"{col}:", options)

    # Numerical input fields
    for col in numerical_cols:
        user_input[col] = st.number_input(f"{col}:", min_value=0.0, step=1.0)

    submit = st.form_submit_button("Predict")

# Perform prediction
if submit:
    input_df = pd.DataFrame([user_input])
    input_df.columns = input_df.columns.str.strip()  # Strip any accidental whitespace

    # Apply label encoding to categorical features
    for col in categorical_cols:
        le = label_encoders.get(col)
        if le:
            try:
                input_df[col] = le.transform(input_df[col])
            except ValueError:
                st.error(f"🚫 Unseen value in '{col}': {input_df[col].values[0]}")
                st.stop()
        else:
            st.error(f"No encoder found for column: {col}")
            st.stop()

    # Reorder columns to match training order
    input_df = input_df[feature_columns]

    # Predict
    prediction = model.predict(input_df)[0]
    result = "✅ Pass" if prediction == 1 else "❌ Fail"

    st.subheader("📊 Prediction Result")
    st.success(f"The student is predicted to: **{result}**")
