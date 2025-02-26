Diabetes Prediction using SVM

Overview

This repository contains a Diabetes Prediction Model built using Support Vector Machine (SVM). The model predicts whether a person is diabetic or not based on medical attributes.

Dataset

The dataset used in this project is the Pima Indians Diabetes Dataset, which consists of the following features:

Pregnancies: Number of times pregnant

Glucose: Plasma glucose concentration (mg/dL)

BloodPressure: Diastolic blood pressure (mm Hg)

SkinThickness: Triceps skin fold thickness (mm)

Insulin: 2-Hour serum insulin (mu U/ml)

BMI: Body mass index (weight in kg / (height in m)^2)

DiabetesPedigreeFunction: Diabetes pedigree function

Age: Age in years

Outcome: 0 (Non-diabetic) or 1 (Diabetic)

Model & Implementation

Algorithm: Support Vector Machine (SVM)

Libraries Used:

numpy

pandas

scikit-learn

matplotlib

seaborn

pickle (for model serialization)

Preprocessing Steps:

Handling missing values

Feature scaling using StandardScaler

Splitting data into training and testing sets (80-20 split)

Training the SVM model with hyperparameter tuning
