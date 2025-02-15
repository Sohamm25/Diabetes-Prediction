# Diabetes Prediction Project

## Introduction

Welcome to the Diabetes Prediction Project! This project aims to help predict if a person might have diabetes by looking at their health information. The project is managed by **Soham Pawar**, who is learning about machine learning and how it can be used to help people.

## Dataset
 
The dataset used in this project contains diagnostic measurements from female individuals of Pima Indian heritage, aged 21 years or older. It comprises essential features such as pregnancies, glucose concentration, blood pressure, skin thickness, insulin level, BMI, diabetes pedigree function, and age.

## Tip:
For running the project type streamlit run app.py and then after entering values press enter after each entry and then after all entries entered wait for 5 secs only and then click on Diabetes Test Result.

## Model Selection and Evaluation

In our quest to find the ultimate predictive model, we embarked on a journey through the realm of machine learning, exploring the capabilities of seven powerful algorithms:

- **Logistic Regression**: A classic yet effective method for binary classification tasks.
- **KNeighbors Classifier**: Harnessing the wisdom of the crowd, this model finds its path through the data by consulting its nearest neighbors.
- **Random Forest Classifier**: An ensemble of decision trees, each with its own unique perspective, coming together to form a forest of knowledge.
- **Support Vector Classifier (SVC)**: Drawing lines in the sand, this model boldly separates the data into distinct territories.
- **Gaussian Naive Bayes**: Embracing simplicity, this model relies on the fundamental principles of probability to make predictions.
- **XGBoost**: A gradient-boosted masterpiece, this model learns from its mistakes and continuously improves with each iteration.
- **CatBoost**: The king of the jungle, this model reigns supreme with its unparalleled accuracy and robustness.

After rigorous evaluation, where each model underwent intense scrutiny and meticulous fine-tuning, one emerged victorious: **CatBoost**. With an astounding accuracy of **92%**, CatBoost proved its mettle on both scaled and unscaled data, solidifying its position as the champion of our predictive arsenal.

## Web Application Deployment

To make our predictive model accessible to a wider audience, I developed a user-friendly web application using Streamlit. The application allows users to input their health parameters and receive instant predictions regarding their diabetes status. This model predicts with an accuracy of 82%, which is good. By using this web app, individuals can assess their risk of diabetes and take proactive measures towards better health.

## Conclusion

The Diabetes Prediction Project showcases the transformative potential of data-driven healthcare solutions. By harnessing the power of machine learning and deploying user-friendly applications, we aim to revolutionize diabetes risk assessment and early detection. Together, we can pave the way for a healthier future.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## .gitignore

