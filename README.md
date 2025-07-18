# 🎓 Student Exam Performance Prediction

This project is designed to predict student exam results (Pass/Fail) using machine learning models based on academic behavior data such as attendance, study hours, sleep patterns, and stress levels. The project includes data preprocessing, model training, evaluation, visualization, and deployment using Streamlit.

---

## 🔍 Features

- Dataset cleaning and preprocessing
- Categorical encoding and vectorization
- ML model training and evaluation (Accuracy, F1-score)
- Data visualization (bar graphs, heatmaps)
- Streamlit-based interactive web application
- Project deployed and available online
- All code available via Google Colab and GitHub

---

## 🚀 Live Demo

👉 [Click here to use the Streamlit App](https://student-pass-fail-prediction-system.streamlit.app/)

---

## 📓 Google Colab Notebook

👉 [Open in Google Colab](https://colab.research.google.com/drive/1IMEkFvDbMmq5UL2QggcVxysg7v2Xp0Ie?usp=sharing)

---

## 🛠️ Technologies Used

- Python
- Pandas, NumPy
- Scikit-learn
- Matplotlib, Seaborn
- Streamlit
- Google Colab
- Git & GitHub

---

## 📁 Project Structure
├── data/ # Dataset files (if applicable)
├── app.py # Streamlit frontend app
├── model.pkl # Trained model file
├── preprocessing.py # Data preprocessing steps
├── requirements.txt # Python dependencies
├── README.md # Project overview and instructions

📊 Results:-
## 📷 Prediction Results (Screenshots)

Below are example predictions from the deployed Streamlit app:

### ✅ Case 1: Likely to Pass
- Good attendance, study time, and sleep.
![Pass Prediction](images/prediction_pass.png)

---

### ❌ Case 2: Likely to Fail (Very low study and attendance)
- No study time, poor attendance, low sleep.
![Fail Prediction](images/prediction_fail_low_study.png)

---

### ❌ Case 3: Likely to Fail (All inputs zero)
- Extremely poor values for all input features.
![Fail Prediction](images/prediction_fail_all_zero.png)


📬 Contact
If you have any questions or suggestions, feel free to reach out!

Prathmesh Dudhale
📧 prathmeshdudhale96@email.com (optional)

