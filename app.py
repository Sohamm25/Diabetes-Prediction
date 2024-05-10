import pandas as pd
import numpy as np
import streamlit as st
from PIL import Image
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
df = pd.read_csv("diabetes.csv")
zero_features = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
data_mean = df[zero_features].mean()
df[zero_features] = df[zero_features].replace(0, data_mean)
X = df.drop('Outcome', axis=1)
y = df['Outcome']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5)
param = {'depth': [4], 'l2_leaf_reg': [5], 'learning_rate': [0.1], 'n_estimators': [100], 'random_seed': [5]}
model = GridSearchCV(CatBoostClassifier(verbose=False), param_grid=param, n_jobs=-1, cv=3)
model.fit(X_train, y_train)
accuracy = model.score(X_test, y_test)
def diabetes_prediction(input_data):
    input_data = np.array(input_data)
    prediction = model.predict(input_data.reshape(1, -1))
    if prediction == 0:
        return "The Person is not Diabetic."
    else:
        return "The Person is Diabetic."
def main():
    # Setting page title and favicon
    st.set_page_config(page_title="Diabetes Prediction Web App", page_icon=":hospital:")
    st.title("Diabetes Prediction Web App")
    st.markdown("**By Soham Pawar**")
    image = Image.open("diabetes_icon.webp")
    st.image(image, use_column_width=True)
    st.markdown("Please enter the following information:")
    Pregnancies = st.text_input("Number of Pregnancies")
    Glucose = st.text_input("Glucose Level")
    BloodPressure = st.text_input("Blood Pressure Value")
    SkinThickness = st.text_input("Skin Thickness Value")
    Insulin = st.text_input("Insulin Value")
    BMI = st.text_input("BMI Value in decimal")
    DiabetesPedigreeFunction = st.text_input("Diabetes Pedigree Function Value in decimal")
    Age = st.text_input("Age of the Person")
    diagnosis = ''
    if st.button('Diabetes Test Result'):
        if Pregnancies == '' or Glucose == '' or BloodPressure == '' or SkinThickness == '' or Insulin == '' or BMI == '' or DiabetesPedigreeFunction == '' or Age == '':
            st.error("Please fill all the fields.")
        else:
            input_data = [Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]
            diagnosis = diabetes_prediction(input_data)
            st.success(diagnosis)
if __name__ == "__main__":
    main()
