# Customer Churn Prediction (Machine Learning)

This project aims to predict customer churn using the IBM Telco Customer Churn dataset.
A supervised machine learning approach is applied to identify key factors influencing
customer attrition.

## Dataset
Publicly available IBM Telco Customer Churn dataset containing customer demographics,
service usage, and billing information.

## Methodology
- Data cleaning and preprocessing
- One-hot encoding of categorical variables
- Feature scaling using Min-Max normalization
- Train-test split (80/20)
- Class imbalance handling with `class_weight='balanced'`

## Model
- Random Forest Classifier
- Fixed random seed (`random_state=42`) for reproducibility

## Evaluation
- Accuracy score
- Confusion matrix
- Feature importance analysis
- Threshold optimization (0.30) to improve recall for churned customers

## Key Outcomes
- Improved recall for churn prediction after threshold adjustment
- Identification of the most influential features affecting churn
- Visual analysis using feature importance plots and correlation heatmaps

## Tools & Libraries
Python, pandas, numpy, scikit-learn, matplotlib, seaborn

![Feature Importance](figures/Feature_Importance.png)
![Churn Correlation](figures/Churn_Correlation.png)
