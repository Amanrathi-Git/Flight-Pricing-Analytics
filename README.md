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

🔗 **Dataset:** The dataset used for this project is available on Kaggle [here](https://www.kaggle.com/datasets/shubhambathwal/flight-price-prediction).

---

## 🛠️ Tech Stack
* **Language:** Python
* **Data Processing & Analytics:** Pandas, NumPy
* **Machine Learning:** Scikit-Learn (Random Forest Regressor, GridSearchCV)
* **Statistical Methods:** SciPy
* **Visualization:** Matplotlib

---

## ⚙️ Step-by-Step Implementation

### 1️⃣ Data Loading & Exploratory Data Analysis (EDA)
The dataset is loaded and examined to inspect feature distributions, summary statistics, and categorical frequencies across flight routes, airlines, and timing brackets.

```python
import pandas as pd

df = pd.read_csv('Clean_Dataset.csv')

# Inspecting flight frequency across airlines
print(df.airline.value_counts())

# Generating summary statistics for flight price
print(df.price.describe())
