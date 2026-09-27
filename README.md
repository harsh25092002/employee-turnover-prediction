# Employee Turnover Prediction

A machine learning project that predicts employee attrition (whether an employee is likely to leave the company) using HR data such as department, income, satisfaction scores, and tenure.

## Overview

Employee attrition is costly for organizations. This project explores an HR dataset, trains multiple classification models, and compares their performance in predicting which employees are at risk of leaving.

## Dataset

`data/hr_employee_data.csv` contains employee records with the following features:

- Demographic & role info: `Age`, `Department`, `JobRole`, `EducationField`
- Compensation: `MonthlyIncome`, `PercentSalaryHike`
- Tenure & experience: `YearsAtCompany`, `NumCompaniesWorked`
- Satisfaction & work conditions: `JobSatisfaction`, `EnvironmentSatisfaction`, `WorkLifeBalance`, `OverTime`, `DistanceFromHome`
- Performance: `PerformanceRating`, `TrainingTimesLastYear`
- Target: `Attrition` (Yes/No)

## Project Structure
```
├── data/
│   └── hr_employee_data.csv       # Raw employee dataset
├── src/
│   ├── generate_dataset.py        # Generates/simulates the dataset
│   ├── eda.py                     # Exploratory data analysis & visualizations
│   └── train_models.py            # Model training and evaluation
├── reports/
│   ├── *.png                      # EDA and model result charts
│   └── model_comparison.csv       # Metrics for all trained models
└── requirements.txt
```

## Models Trained

Three classification models were trained and evaluated on a held-out test set:

| Model | Accuracy | Precision | Recall | F1 Score | ROC AUC |
|---|---|---|---|---|---|
| Logistic Regression | 0.860 | 0.671 | 0.712 | 0.691 | 0.914 |
| Decision Tree | 0.867 | 0.741 | 0.606 | 0.667 | 0.854 |
| **Random Forest** | **0.893** | **0.886** | 0.591 | 0.709 | **0.934** |

The **Random Forest** model achieved the best overall accuracy and ROC AUC, making it the top-performing model for this task.

## Exploratory Analysis

The `reports/` folder includes visualizations such as:
- Attrition breakdown by department
- Correlation heatmap of numeric features
- Feature importance from the Random Forest model
- Income vs. attrition trends
- Confusion matrix for the Random Forest model

## Getting Started

1. Clone the repository
```bash
   git clone https://github.com/harsh25092002/employee-turnover-prediction.git
   cd employee-turnover-prediction
```

2. Install dependencies
```bash
   pip install -r requirements.txt
```

3. Run the exploratory analysis
```bash
   python src/eda.py
```

4. Train and evaluate the models
```bash
   python src/train_models.py
```

## Tech Stack

- **Python**
- **pandas / numpy** — data processing
- **scikit-learn** — model training & evaluation (Logistic Regression, Decision Tree, Random Forest)
- **matplotlib / seaborn** — data visualization

## Future Improvements

- Hyperparameter tuning (e.g. GridSearchCV) to improve recall
- Try additional models such as XGBoost or Gradient Boosting
- Address class imbalance with techniques like SMOTE
- Deploy the best model behind a simple API for real-time predictions
