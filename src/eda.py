"""
Exploratory Data Analysis for the employee attrition dataset.
Generates summary statistics and saves visualization plots to the reports/ folder.
"""

import os

import matplotlib
matplotlib.use("Agg")  # headless rendering
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def run_eda(csv_path: str = "data/hr_employee_data.csv", output_dir: str = "reports"):
    os.makedirs(output_dir, exist_ok=True)
    df = pd.read_csv(csv_path)

    print("Dataset shape:", df.shape)
    print("\nMissing values:\n", df.isnull().sum())
    print("\nAttrition rate:\n", df["Attrition"].value_counts(normalize=True))

    # Attrition count plot
    plt.figure(figsize=(5, 4))
    sns.countplot(data=df, x="Attrition")
    plt.title("Attrition Count")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/attrition_count.png")
    plt.close()

    # Attrition by department
    plt.figure(figsize=(7, 4))
    sns.countplot(data=df, x="Department", hue="Attrition")
    plt.title("Attrition by Department")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/attrition_by_department.png")
    plt.close()

    # Monthly income distribution by attrition
    plt.figure(figsize=(6, 4))
    sns.boxplot(data=df, x="Attrition", y="MonthlyIncome")
    plt.title("Monthly Income vs Attrition")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/income_vs_attrition.png")
    plt.close()

    # Correlation heatmap for numeric features
    numeric_df = df.select_dtypes(include="number").drop(columns=["EmployeeID"])
    plt.figure(figsize=(9, 7))
    sns.heatmap(numeric_df.corr(), annot=False, cmap="coolwarm")
    plt.title("Feature Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/correlation_heatmap.png")
    plt.close()

    print(f"\nSaved EDA plots to '{output_dir}/'")


if __name__ == "__main__":
    run_eda()
