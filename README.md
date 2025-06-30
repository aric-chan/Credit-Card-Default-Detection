Credit Card Default Prediction

#Overview
CreditGuard is a machine learning project aimed at predicting whether a credit card client will default on their next payment. Using the Default of Credit Card Clients Dataset from Kaggle, this project employs a binary classification approach to identify default risk based on 30,000 client records with 24 features. The target variable, default.payment.next.month, indicates default (1) or non-default (0), with a class imbalance (22% default rate). The project achieved a recall score of 75.0% on the test set using logistic regression, prioritizing the identification of default cases.
Objectives

Build a robust machine learning pipeline to predict credit card default risk.
Perform exploratory data analysis (EDA) and feature engineering to identify impactful predictors.
Develop and compare multiple classification models, optimizing for recall due to class imbalance.
Analyze feature importance using SHAP to derive actionable insights for financial risk assessment.

#Dataset
The dataset contains 30,000 samples with 24 features, including:

Numerical Features: LIMIT_BAL (credit limit), AGE, bill amounts (Bill_Apr to Bill_Sep), and payment amounts (Repay_Apr to Repay_Sep).
Categorical Features: SEX, EDUCATION, MARRIAGE, and repayment statuses (RepayStat_Apr to RepayStat_Sep).
Target: DefRepayNext (binary: 0 = non-default, 1 = default).

#Key preprocessing steps:

Removed invalid entries (e.g., EDUCATION and MARRIAGE values of 0).
Consolidated EDUCATION value 6 into 5 ("Unknown") for consistency.
Identified repayment status and bill-payment differences as strong predictors through EDA.
Handled class imbalance (mean target value: 0.22) by focusing on recall as the evaluation metric.

#Methodology

#Data Preprocessing:

Used Pandas and NumPy for EDA and data cleaning.
Applied Scikit-learn's StandardScaler for numerical features, OneHotEncoder for categorical features (SEX, EDUCATION, MARRIAGE), and OrdinalEncoder for repayment status features.
Constructed a ColumnTransformer to streamline preprocessing within a Scikit-learn pipeline.


#Model Development:

Implemented classification models: Logistic Regression, Decision Trees, SVM, and KNN.
Used Scikit-learn pipelines to integrate preprocessing and modeling.
Evaluated models using cross-validation (cross_val_score, cross_validate) with recall as the primary metric due to class imbalance.


#Hyperparameter Optimization:

Conducted grid search to tune hyperparameters (e.g., C for Logistic Regression, n_neighbors for KNN).
Found Logistic Regression to be effective, balancing simplicity and performance.


#Model Evaluation:

Achieved a test recall score of 75.0% with the optimized Logistic Regression model.
Visualized performance using a confusion matrix with Seaborn to assess true positives and false negatives.


#Feature Analysis:

Used SHAP plots to identify recent repayment status (RepayStat_Sep) as the top predictor of default.
Noted potential multicollinearity among engineered features, which reduced their impact in SHAP analysis.



#Results

Test Recall Score: 75.0% (Logistic Regression), effectively identifying default cases in an imbalanced dataset.
Key Insight: Simpler models like Logistic Regression outperformed complex models, which risked overfitting due to numerous hyperparameters.
Feature Importance: Recent repayment status was the most influential predictor, highlighting the importance of recent payment behavior in default risk.
Takeaway: Thorough data understanding and preprocessing are critical for effective modeling, and simpler models can yield robust results in imbalanced datasets.
