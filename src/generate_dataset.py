"""
Generates a synthetic but realistic HR employee-attrition dataset.

Public HR-attrition datasets (e.g. IBM's) are commonly used for this kind of
project. This script builds a dataset with the same style of features so the
project is fully self-contained and reproducible without needing an external
download.
"""

import numpy as np
import pandas as pd

RANDOM_SEED = 42


def generate_hr_dataset(n_employees: int = 1500) -> pd.DataFrame:
    rng = np.random.default_rng(RANDOM_SEED)

    departments = ["Sales", "R&D", "HR", "Engineering", "Support"]
    job_roles = ["Executive", "Manager", "Analyst", "Technician", "Representative"]
    education_fields = ["Life Sciences", "Marketing", "Technical Degree", "HR", "Other"]

    age = rng.integers(21, 60, n_employees)
    monthly_income = rng.integers(15000, 150000, n_employees)
    years_at_company = rng.integers(0, 25, n_employees)
    distance_from_home = rng.integers(1, 30, n_employees)
    job_satisfaction = rng.integers(1, 5, n_employees)
    environment_satisfaction = rng.integers(1, 5, n_employees)
    work_life_balance = rng.integers(1, 5, n_employees)
    overtime = rng.choice(["Yes", "No"], n_employees, p=[0.3, 0.7])
    num_companies_worked = rng.integers(0, 8, n_employees)
    training_times_last_year = rng.integers(0, 6, n_employees)
    percent_salary_hike = rng.integers(10, 25, n_employees)
    department = rng.choice(departments, n_employees)
    job_role = rng.choice(job_roles, n_employees)
    education_field = rng.choice(education_fields, n_employees)
    performance_rating = rng.integers(1, 5, n_employees)

    # Build a latent "attrition risk score" from the features so that labels
    # are correlated with realistic drivers of attrition, then binarize it.
    risk_score = (
        (overtime == "Yes").astype(int) * 1.4
        + (job_satisfaction <= 2).astype(int) * 1.1
        + (work_life_balance <= 2).astype(int) * 0.9
        + (distance_from_home > 15).astype(int) * 0.6
        + (years_at_company < 2).astype(int) * 0.8
        + (monthly_income < 30000).astype(int) * 0.7
        + (num_companies_worked > 4).astype(int) * 0.5
        - (environment_satisfaction >= 4).astype(int) * 0.6
        - (years_at_company > 10).astype(int) * 0.7
        + rng.normal(0, 0.6, n_employees)
    )

    threshold = np.quantile(risk_score, 0.78)  # roughly 22% attrition rate
    attrition = np.where(risk_score > threshold, "Yes", "No")

    df = pd.DataFrame({
        "EmployeeID": np.arange(1, n_employees + 1),
        "Age": age,
        "Department": department,
        "JobRole": job_role,
        "EducationField": education_field,
        "MonthlyIncome": monthly_income,
        "YearsAtCompany": years_at_company,
        "DistanceFromHome": distance_from_home,
        "JobSatisfaction": job_satisfaction,
        "EnvironmentSatisfaction": environment_satisfaction,
        "WorkLifeBalance": work_life_balance,
        "OverTime": overtime,
        "NumCompaniesWorked": num_companies_worked,
        "TrainingTimesLastYear": training_times_last_year,
        "PercentSalaryHike": percent_salary_hike,
        "PerformanceRating": performance_rating,
        "Attrition": attrition,
    })

    return df


if __name__ == "__main__":
    dataset = generate_hr_dataset()
    dataset.to_csv("data/hr_employee_data.csv", index=False)
    print(f"Generated dataset with {len(dataset)} rows -> data/hr_employee_data.csv")
    print(dataset["Attrition"].value_counts(normalize=True))
