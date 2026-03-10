# Bank Customer Churn Prediction: Multi-Model Ensemble Approach

> **Project Goal:** Predict customer attrition for a banking institution using an ensemble of Gradient Boosting Decision Trees (GBDT).

---

## Project Overview
This project focuses on identifying high-risk customers who are likely to churn (exit) from a bank. It was originally developed as a graduation assignment at **Korea National Open University (KNOU)** and has been significantly enhanced using advanced ensemble techniques and automated pipelines.

## Tech Stack
- **Language:** Python 3.x
- **Core Models:** XGBoost, CatBoost, LightGBM
- **Key Libraries:** Scikit-learn, Pandas, NumPy, Matplotlib, Seaborn
- **AI Collaborator:** Google Gemini (Code Refactoring & Optimization)

---

## Exploratory Data Analysis (EDA)
Understanding the data distribution and relationships between features is the first step in our predictive modeling.

### Target Class Distribution
![Target Distribution](./images/target_distribution.png)
*The dataset shows a natural imbalance, with approximately 20% of customers having exited. This informed our choice of using weighted loss functions in the models.*

### Feature Correlation Heatmap
![Feature Correlation](./images/correlation_heatmap.png)
*Correlation analysis helped identify redundant features and understand the linear relationships between numerical variables like Age, Balance, and Credit Score.*

---

## Key Highlights

### 1. Strategic Data Augmentation
To improve the model's ability to generalize, I merged the competition dataset with the original "Churn Modelling" dataset. This increased the training volume and allowed the model to learn more robust patterns.

### 2. Automated Preprocessing Pipeline
I utilized Scikit-learn's `ColumnTransformer` to create a professional-grade preprocessing workflow.
* **Categorical Data:** Encoded using `OneHotEncoder` with `handle_unknown='ignore'`.
* **Seamless Integration:** All transformations are applied in a single pipeline, ensuring consistency between training and test data.

### 3. High-Performance Ensemble (Soft Voting)
Instead of relying on a single algorithm, I implemented a **Soft Voting Ensemble** strategy. By averaging the predicted probabilities from **XGBoost, CatBoost, and LightGBM**, the model achieves a balanced result.
* **XGBoost:** Excellent at capturing precise linear relationships.
* **CatBoost:** Naturally handles categorical features with high accuracy.
* **LightGBM:** Fast training with high leaf-wise performance.

---

## Model Insights & Results

### Top 10 Drivers of Customer Churn
![Feature Importance](./images/feature_importance.png)
*The XGBoost importance plot reveals that **Age**, **Balance**, and **EstimatedSalary** are the strongest predictors of whether a customer will stay or leave. This insight allows for targeted marketing to specific demographic segments.*

### Final Prediction Distribution (Test Set)
![Final Prediction](./images/final_prediction_dist.png)
*The final ensemble model identified **19.37%** of the test population as high-risk churners, providing a realistic and actionable insight for the bank's retention team.*

---

## Author's Note
This enhanced version demonstrates my proficiency in building end-to-end machine learning pipelines. By refactoring my academic work with modern AI collaboration (Google Gemini), I have prepared a portfolio piece that meets the technical standards for graduate-level studies and professional data science roles in the U.S.
