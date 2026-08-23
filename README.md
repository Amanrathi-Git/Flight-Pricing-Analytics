# ✈️ Flight Price Prediction: End-to-End Machine Learning Pipeline

[![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)](#)
[![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)](#)
[![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)](#)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)](#)
[![Matplotlib](https://img.shields.io/badge/Matplotlib-%23ffffff.svg?style=for-the-badge&logo=Matplotlib&logoColor=black)](#)

> **Predicting airline flight fares through data-driven insights and Machine Learning.**

---

## 📖 Overview
This project develops an end-to-end Machine Learning pipeline to predict airline ticket prices based on key travel variables such as carrier, route, departure/arrival schedules, cabin class, and transit stops[cite: 2]. By combining exploratory data analysis, data encoding, regression modeling, and hyperparameter optimization, the model accurately predicts price variations on tabular travel datasets[cite: 2].

🔗 **Dataset:** The dataset used for this project is available on Kaggle [here](https://www.kaggle.com/datasets/shubhambathwal/flight-price-prediction).

---

## 🛠️ Tech Stack
* **Language:** Python
* **Data Processing & Analytics:** Pandas, NumPy[cite: 2]
* **Machine Learning:** Scikit-Learn (Random Forest Regressor, RandomizedSearchCV)[cite: 2]
* **Statistical Methods:** SciPy[cite: 2]
* **Visualization:** Matplotlib[cite: 2]

---

## ⚙️ Step-by-Step Implementation

### 1️⃣ Data Loading & Exploratory Data Analysis (EDA)
The dataset is loaded and examined to inspect feature distributions, summary statistics, and categorical frequencies across flight routes, airlines, and timing brackets[cite: 2].

```python
import pandas as pd

df = pd.read_csv('Clean_Dataset.csv')

# Inspecting flight frequency across airlines
print(df.airline.value_counts())

# Generating summary statistics for flight price
print(df.price.describe())
```

### 2️⃣ Data Preprocessing & Feature Engineering
Transformations are applied to clean the dataset and prepare categorical variables for tree-based regression algorithms[cite: 2]:
* **Feature Removal:** Dropped non-predictive identifier columns (`Unnamed: 0` and `flight`)[cite: 2].
* **Binary Encoding:** Converted travel `class` into binary values (`1` for Business, `0` for Economy)[cite: 2].
* **Label Encoding:** Factorized the `stops` column to represent stop counts numerically (`pd.factorize()`)[cite: 2].
* **One-Hot Encoding:** Applied dummy variable encoding via `pd.get_dummies()` across nominal categories (`airline`, `source_city`, `destination_city`, `arrival_time`, `departure_time`)[cite: 2].

```python
# Dropping redundant identifier columns
df = df.drop(['Unnamed: 0', 'flight'], axis=1)

# Binary & label encoding
df['class'] = df['class'].apply(lambda x: 1 if x == 'Business' else 0)
df.stops = pd.factorize(df.stops)[0]

# One-hot encoding nominal categorical features
categorical_cols = ['airline', 'source_city', 'destination_city', 'arrival_time', 'departure_time']
for col in categorical_cols:
    df = df.join(pd.get_dummies(df[col], prefix=col, dtype=int)).drop(col, axis=1)
```

### 3️⃣ Train-Test Split
The dataset is split into feature matrix ($X$) and target vector ($y$), reserving 80% for training and 20% for testing[cite: 2].

```python
from sklearn.model_selection import train_test_split

x, y = df.drop('price', axis=1), df.price
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
```

### 4️⃣ Model Training & Hyperparameter Tuning
A **Random Forest Regressor** is trained using `RandomizedSearchCV` across a distributed parameter grid to find optimal tree parameters[cite: 2].

```python
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint

param_dist = {
    'n_estimators': randint(100, 300),
    'max_depth': [None, 10, 20, 30, 40, 50],
    'min_samples_split': randint(2, 11),
    'min_samples_leaf': randint(1, 5),
    'max_features': [1.0, 'sqrt']
}

reg = RandomForestRegressor(random_state=42)

random_search = RandomizedSearchCV(
    estimator=reg,
    param_distributions=param_dist,
    n_iter=5,
    cv=3,
    scoring='neg_mean_squared_error',
    verbose=1,
    random_state=10
)

# Fitting model on sample for resource efficiency
random_search.fit(x_train[:5000], y_train[:5000])
best_regressor = random_search.best_estimator_
```

### 5️⃣ Model Evaluation
The optimized model is evaluated on the unseen test partition, achieving an $R^2$ score of **0.951**[cite: 2].

```python
import math
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

y_pred = best_regressor.predict(x_test)

print('R2 Score :', r2_score(y_test, y_pred))
print('MAE      :', mean_absolute_error(y_test, y_pred))
print('MSE      :', mean_squared_error(y_test, y_pred))
print('RMSE     :', math.sqrt(mean_squared_error(y_test, y_pred)))
```

| Metric | Result |
| :--- | :--- |
| **R² Score** | **0.951**[cite: 2] |
| **MAE** | **3,221.23**[cite: 2] |
| **MSE** | **25,444,280.20**[cite: 2] |
| **RMSE** | **5,044.23**[cite: 2] |

---

## 📊 Visualizations

### 📈 Actual vs. Predicted Price
A scatter plot demonstrating strong alignment between actual flight ticket fares and model predictions along the diagonal[cite: 2].

```python
import matplotlib.pyplot as plt

plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.3)
plt.xlabel('Actual Flight Price')
plt.ylabel('Predicted Flight Price')
plt.title('Prediction vs. Actual Price')
plt.show()
```

<div align="center">
  <!-- Drag and drop your Scatter Plot screenshot here on GitHub -->
  *[Insert Scatter Plot Image Here]*
</div>

---

### 🏆 Feature Importance
Evaluating the top 10 most influential features driving ticket price predictions (e.g., flight class, duration, and days left)[cite: 2].

```python
importances = dict(zip(best_regressor.feature_names_in_, best_regressor.feature_importances_))
sorted_importances = sorted(importances.items(), key=lambda x: x[1], reverse=True)

plt.figure(figsize=(14, 5))
plt.bar([x[0] for x in sorted_importances[:10]], [x[1] for x in sorted_importances[:10]])
plt.title('Top 10 Feature Importances')
plt.xticks(rotation=45)
plt.show()
```

<div align="center">
  <!-- Drag and drop your Bar Chart screenshot here on GitHub -->
  *[Insert Feature Importance Bar Chart Image Here]*
</div>

---

## 🚀 Getting Started

### Prerequisites
Install the required dependencies:
```bash
pip install pandas numpy scikit-learn scipy matplotlib
```

### Running the Project
1. Clone the repository:
   ```bash
   git clone [https://github.com/your-username/Flight-Pricing-Analytics.git](https://github.com/your-username/Flight-Pricing-Analytics.git)
   ```
2. Download `Clean_Dataset.csv` from [Kaggle](https://www.kaggle.com/datasets/shubhambathwal/flight-price-prediction) and place it in the project root directory.
3. Open and run the notebook:
   ```bash
   jupyter notebook Main_.ipynb
   ```
