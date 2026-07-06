# Loan Defaulter Classification & Prediction

A data science project analyzing borrower data to understand loan default risk, using exploratory data analysis, data cleaning/preprocessing, and supervised machine learning (classification and regression).

## 📊 Project Overview

This notebook walks through a complete data science workflow on a loan dataset:

**Part 1 — Data Analysis & Preprocessing**
- Exploratory analysis of loan outcomes by `home_ownership` and `verification_status`
- Descriptive statistics and skewness analysis of numerical features
- Outlier detection (IQR method) and treatment (winsorizing)
- Correlation analysis to identify key predictors of default
- Data cleaning: type conversion, missing value imputation, categorical encoding
- Feature scaling (standardization) and feature engineering (e.g. credit utilization ratio)

**Part 2 — Machine Learning Models**
- **Classification** (predicting `loan_status`): Logistic Regression, Decision Tree, Random Forest
- **Regression** (predicting `int_rate`): Linear Regression, Decision Tree Regressor
- Model evaluation using accuracy, AUC, precision, recall (classification) and MSE, MAE, R² (regression)
- Hyperparameter tuning with `GridSearchCV` and before/after performance comparison

## 🗂️ Dataset

The dataset is based on the [Loan Defaulter Classification dataset](https://www.kaggle.com/datasets/prateek146/loan-defaulter-classification) from Kaggle, modified for this assignment. It is referenced in the notebook as `NPL.csv`.

> **Note:** The dataset file is not included in this repository. Download it separately and place it in the project root as `NPL.csv` before running the notebook.

## 🛠️ Requirements

- Python 3.x
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- scipy

Install everything with:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn scipy
```

## 🚀 Usage

1. Clone this repository:
   ```bash
   git clone https://github.com/qtrerum-code/AI-model.git
   cd AI-model
   ```
2. Place `NPL.csv` in the project root.
3. Launch Jupyter and open the notebook:
   ```bash
   jupyter notebook
   ```
4. Run the cells in order (Part 1 preprocessing must run before Part 2 modeling).

## 📈 Key Results

- **Best classification model:** Random Forest (accuracy, precision, recall, and AUC all ≈ 0.83)
- **Best regression model:** Decision Tree Regressor (R² ≈ 0.97, lower error than Linear Regression)
- Hyperparameter tuning via GridSearchCV further improved model performance across the board.

## 📄 License

This project is for educational purposes (DSAI 2201 coursework).
