# Credit Card Default Prediction


## Overview
This is a machine learning project designed to predict credit card default risk using the [Default of Credit Card Clients Dataset](https://www.kaggle.com/datasets/uciml/default-of-credit-card-clients-dataset) from Kaggle. The project tackles a binary classification problem, predicting whether a client will default on their next payment (`default.payment.next.month`) using a dataset of 30,000 records and 24 features. Despite a class imbalance (22% default rate), the optimized Logistic Regression model achieved a recall score of 75.0%, effectively identifying default cases.

## Objectives
- Develop a robust machine learning pipeline for credit card default prediction.
- Conduct exploratory data analysis (EDA) and feature engineering to identify key predictors.
- Compare multiple classification models, optimizing for recall to handle class imbalance.
- Analyze feature importance using SHAP to provide financial insights.

## Dataset
The dataset includes 30,000 samples with 24 features:
- **Numerical Features**: Credit limit (`LIMIT_BAL`), age (`AGE`), bill amounts (`Bill_Apr` to `Bill_Sep`), payment amounts (`Repay_Apr` to `Repay_Sep`).
- **Categorical Features**: Gender (`SEX`), education (`EDUCATION`), marriage (`MARRIAGE`), repayment status (`RepayStat_Apr` to `RepayStat_Sep`).
- **Target**: `DefRepayNext` (0 = non-default, 1 = default).

**Preprocessing**:
- Removed invalid entries (e.g., `EDUCATION` and `MARRIAGE` values of 0).
- Consolidated `EDUCATION` value 6 into 5 ("Unknown") for consistency.
- Identified repayment status and bill-payment differences as key predictors.
- Addressed class imbalance (mean target: 0.22) by prioritizing recall.

## Methodology
1. **Data Preprocessing**:
   - Used Pandas and NumPy for EDA and data cleaning.
   - Applied Scikit-learn's `StandardScaler` for numerical features, `OneHotEncoder` for categorical features (`SEX`, `EDUCATION`, `MARRIAGE`), and `OrdinalEncoder` for repayment status.
   - Built a `ColumnTransformer` within a Scikit-learn pipeline for streamlined preprocessing.

2. **Model Development**:
   - Implemented classification models: Logistic Regression, Decision Trees, SVM, and KNN.
   - Used Scikit-learn pipelines to integrate preprocessing and modeling.
   - Evaluated models with cross-validation (`cross_val_score`, `cross_validate`), focusing on recall.

3. **Hyperparameter Optimization**:
   - Performed grid search to tune hyperparameters (e.g., `C` for Logistic Regression, `n_neighbors` for KNN).
   - Found Logistic Regression effective for balancing simplicity and performance.

4. **Model Evaluation**:
   - Achieved a test recall score of 75.0% with Logistic Regression.
   - Visualized performance using a confusion matrix with Seaborn.

5. **Feature Analysis**:
   - Used SHAP plots to identify recent repayment status (`RepayStat_Sep`) as the top predictor.
   - Noted potential multicollinearity among engineered features, limiting their impact.

## Results
- **Test Recall Score**: 75.0% (Logistic Regression), effectively identifying default cases in an imbalanced dataset.
- **Key Finding**: Recent repayment status was the most influential predictor, highlighting its role in default risk.
- **Insight**: Simpler models like Logistic Regression outperformed complex models, which risked overfitting due to excessive hyperparameters.
- **Takeaway**: Thorough data understanding and preprocessing are critical for effective modeling in imbalanced datasets.

## Usage
- **EDA**: Explore feature distributions and class imbalance in the notebook's EDA section.
- **Model Training**: The pipeline automates preprocessing and training. Modify grid search parameters to experiment with hyperparameters.
- **Evaluation**: Review the confusion matrix and SHAP plots for model performance and feature importance.
- **Extending**: Add models (e.g., Random Forest) or explore feature selection to enhance performance.

## Future Improvements
- Incorporate ensemble methods (e.g., Random Forest, CatBoost) to potentially improve recall.
- Apply feature selection techniques (e.g., Recursive Feature Elimination) to address multicollinearity.
- Deploy the model in a web application with a user-friendly interface for real-time predictions.

## Acknowledgments
- Dataset: [UCI Machine Learning Repository](https://www.kaggle.com/datasets/uciml/default-of-credit-card-clients-dataset)
- Course: CPSC 330 - Applied Machine Learning, UBC (2025)
