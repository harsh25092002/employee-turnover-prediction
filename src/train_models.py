"""
Trains and evaluates classification models to predict employee attrition:
Logistic Regression, Decision Tree, and Random Forest.
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

RANDOM_SEED = 42


def load_and_prepare_data(csv_path: str = "data/hr_employee_data.csv"):
    df = pd.read_csv(csv_path)
    df = df.drop(columns=["EmployeeID"])

    categorical_cols = df.select_dtypes(include="object").columns.drop("Attrition")
    df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

    label_encoder = LabelEncoder()
    df_encoded["Attrition"] = label_encoder.fit_transform(df_encoded["Attrition"])  # Yes=1, No=0

    X = df_encoded.drop(columns=["Attrition"])
    y = df_encoded["Attrition"]
    return X, y


def evaluate_model(name, model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        "model": name,
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_proba),
    }
    print(f"\n===== {name} =====")
    print(classification_report(y_test, y_pred, target_names=["No Attrition", "Attrition"]))
    return metrics, y_pred


def main():
    os.makedirs("reports", exist_ok=True)
    X, y = load_and_prepare_data()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    results = []

    # 1. Logistic Regression (benefits from scaled features)
    log_reg = LogisticRegression(max_iter=1000, random_state=RANDOM_SEED)
    log_reg.fit(X_train_scaled, y_train)
    metrics, _ = evaluate_model("Logistic Regression", log_reg, X_test_scaled, y_test)
    results.append(metrics)

    # 2. Decision Tree
    dt = DecisionTreeClassifier(max_depth=6, random_state=RANDOM_SEED)
    dt.fit(X_train, y_train)
    metrics, _ = evaluate_model("Decision Tree", dt, X_test, y_test)
    results.append(metrics)

    # 3. Random Forest
    rf = RandomForestClassifier(n_estimators=200, max_depth=8, random_state=RANDOM_SEED)
    rf.fit(X_train, y_train)
    metrics, rf_pred = evaluate_model("Random Forest", rf, X_test, y_test)
    results.append(metrics)

    # Save comparison table
    results_df = pd.DataFrame(results).set_index("model").round(3)
    results_df.to_csv("reports/model_comparison.csv")
    print("\n===== Model Comparison =====")
    print(results_df)

    # Confusion matrix for the best model (Random Forest)
    ConfusionMatrixDisplay.from_predictions(y_test, rf_pred, display_labels=["No Attrition", "Attrition"])
    plt.title("Random Forest - Confusion Matrix")
    plt.tight_layout()
    plt.savefig("reports/random_forest_confusion_matrix.png")
    plt.close()

    # Feature importance from Random Forest
    importances = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False).head(10)
    plt.figure(figsize=(8, 5))
    importances.plot(kind="barh")
    plt.gca().invert_yaxis()
    plt.title("Top 10 Feature Importances (Random Forest)")
    plt.tight_layout()
    plt.savefig("reports/feature_importances.png")
    plt.close()

    print("\nSaved model comparison and plots to 'reports/'")


if __name__ == "__main__":
    main()
