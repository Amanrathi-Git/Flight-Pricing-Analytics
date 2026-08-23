# ✈️ Flight Price Prediction: End-to-End Machine Learning Pipeline

[![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)](#)
[![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)](#)
[![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)](#)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)](#)
[![Matplotlib](https://img.shields.io/badge/Matplotlib-%23ffffff.svg?style=for-the-badge&logo=Matplotlib&logoColor=black)](#)

> **Predicting airline flight fares through data-driven insights and Machine Learning.**

---

## 📖 Overview
This project develops an end-to-end Machine Learning pipeline to predict airline ticket prices based on key travel variables such as carrier, route, departure/arrival schedules, cabin class, and transit stops. By combining exploratory data analysis, data encoding, regression modeling, and hyperparameter optimization, the model accurately predicts price variations on tabular travel datasets.

🔗 **Dataset:** The dataset used for this project is available on Kaggle [here](https://www.kaggle.com/datasets/shubhambathwal/flight-price-prediction?resource=download).

---

## 🛠️ Tech Stack
* **Language:** Python
* **Data Processing & Analytics:** Pandas, NumPy
* **Machine Learning:** Scikit-Learn (Random Forest Regressor, GridSearchCV)[cite: 1]
* **Statistical Methods:** SciPy[cite: 1]
* **Visualization:** Matplotlib[cite: 1]

---

## ⚙️ Full Python Implementation Script
> *Copy the entire code block below to run the complete pipeline from start to finish:*

```python
# =====================================================================
# FLIGHT PRICE PREDICTION: END-TO-END MACHINE LEARNING PIPELINE
# =====================================================================

import pandas as pd
import numpy as np
import math
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------
# 1️⃣ DATA LOADING & EXPLORATORY DATA ANALYSIS (EDA)
# ---------------------------------------------------------------------
# EXPLANATION:
# The first phase involves loading the raw dataset and understanding its underlying structure 
# and distributions. Exploratory data analysis is crucial because it helps identify categorical 
# variables, check for missing values, and understand general characteristics (such as min, 
# max, and median values) of our target variable (price) and continuous features like flight duration.

df = pd.read_csv('Clean_Dataset.csv')

print("--- Airline Value Counts ---")
print(df.airline.value_counts())

print("\n--- Price Summary Statistics ---")
print(df.price.describe())


# ---------------------------------------------------------------------
# 2️⃣ DATA PREPROCESSING & FEATURE ENGINEERING
# ---------------------------------------------------------------------
# EXPLANATION:
# Machine Learning models cannot process raw text data directly; they require strict numerical input. 
# Therefore, feature engineering and transformations are applied:
# - Feature Removal: Non-predictive identifier columns (Unnamed: 0 and flight) are completely dropped.
# - Binary Encoding: The travel class feature is mapped into binary format (1 for Business, 0 for Economy).
# - Label Encoding: The stops attribute is transformed into an ordered numerical format using pd.factorize().
# - One-Hot Encoding: Nominal categorical variables are converted into distinct boolean columns using pd.get_dummies().

df = df.drop(['Unnamed: 0', 'flight'], axis=1)

df['class'] = df['class'].apply(lambda x: 1 if x == 'Business' else 0)
df.stops = pd.factorize(df.stops)[0]

categorical_cols = ['airline', 'source_city', 'destination_city', 'arrival_time', 'departure_time']
for col in categorical_cols:
    df = df.join(pd.get_dummies(df[col], prefix=col, dtype=int)).drop(col, axis=1)


# ---------------------------------------------------------------------
# 3️⃣ TRAIN-TEST SPLIT
# ---------------------------------------------------------------------
# EXPLANATION:
# To accurately evaluate how well the model learns and generalizes, the data must be split. 
# The target variable (price) is isolated into vector y, while the remaining attributes form the 
# feature matrix x. The dataset is then divided using train_test_split, allocating 80% of the data 
# for training and holding back 20% to test the model's predictive accuracy on completely unseen data.

x, y = df.drop('price', axis=1), df.price
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)


# ---------------------------------------------------------------------
# 4️⃣ MODEL TRAINING & HYPERPARAMETER TUNING
# ---------------------------------------------------------------------
# EXPLANATION:
# A Random Forest Regressor is chosen for its powerful performance on complex, non-linear tabular datasets. 
# To maximize model accuracy and prevent overfitting, GridSearchCV is implemented to systematically test 
# combinations of hyperparameters—such as the number of trees (n_estimators), maximum tree depth (max_depth), 
# and split criteria. The grid search is fit on an optimized training sample subset to ensure smooth 
# resource allocation while maintaining high model performance.

param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [10, 20],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2],
    'max_features': ['sqrt'] 
}

reg = RandomForestRegressor(random_state=42)

grid_search = GridSearchCV(
    estimator=reg, 
    param_grid=param_grid, 
    cv=2, 
    verbose=1
)

x_train_sample = x_train[:5000]
y_train_sample = y_train[:5000]

grid_search.fit(x_train_sample, y_train_sample)
best_reg = grid_search.best_estimator_
print("\nBest Parameters Found:", grid_search.best_params_)


# ---------------------------------------------------------------------
# 5️⃣ MODEL EVALUATION
# ---------------------------------------------------------------------
# EXPLANATION:
# The optimized model is evaluated against the unseen 20% test partition to quantify its real-world 
# predictive performance. Standard regression metrics are calculated, including R^2 Score, Mean 
# Absolute Error (MAE), Mean Squared Error (MSE), and Root Mean Squared Error (RMSE).

y_pred = best_reg.predict(x_test)

print('\nR2 Score: ', r2_score(y_test, y_pred))
print('MAE: ', mean_absolute_error(y_test, y_pred))
print('MSE: ', mean_squared_error(y_test, y_pred))
print('RMSE: ', math.sqrt(mean_squared_error(y_test, y_pred)))


# ---------------------------------------------------------------------
# 6️⃣ DATA VISUALIZATION
# ---------------------------------------------------------------------
# EXPLANATION:
# Visualizing actual vs. predicted values and extracting feature importances to see what 
# drives flight price variations most.

# Scatter Plot: Actual vs Predicted
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.3)
plt.xlabel('Actual Flight Price')
plt.ylabel('Predicted Flight Price')
plt.title('Prediction VS Actual Price')
plt.show()

# Feature Importance Bar Chart
importances = dict(zip(best_reg.feature_names_in_, best_reg.feature_importances_))
sorted_importances = sorted(importances.items(), key=lambda x: x[1], reverse=True)

plt.figure(figsize=(16, 6))
plt.bar([x[0] for x in sorted_importances[:10]], [x[1] for x in sorted_importances[:10]])
plt.title('Top 10 Feature Importances')
plt.xticks(rotation=45)
plt.show()
