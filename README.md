🧠 Project Overview
This project involves building a robust machine learning-based decision support model that assists universities in evaluating scholarship eligibility for students based on academic, socioeconomic, and extracurricular factors.

The model aims to:

Improve accuracy of scholarship award prediction.

Address issues such as overfitting and multicollinearity.

Provide transparent and explainable decision logic for admissions committees.

📌 Objectives
Optimize Random Forest Regression:

Tune hyperparameters such as n_estimators, max_depth, min_samples_split, and min_samples_leaf.

Implement cross-validation to ensure generalization and reduce overfitting.

Fix Multicollinearity for Linear Regression:

Analyze feature correlation using a heatmap and VIF (Variance Inflation Factor).

Apply feature selection or dimensionality reduction techniques (e.g., PCA, regularization) to mitigate multicollinearity and improve model interpretability.

Tune Decision Tree Regressor:

Fine-tune max_depth, min_samples_split, and min_samples_leaf to avoid overfitting.

Use grid search with cross-validation for model selection.

🧪 Approach
1. Data Preprocessing
Handled missing values using imputation techniques.

Scaled features using StandardScaler or MinMaxScaler.

Converted categorical variables using one-hot encoding.

Split the dataset into training and testing sets.

2. Exploratory Data Analysis (EDA)
Visualized data distribution and relationships using seaborn/matplotlib.

Detected multicollinearity through a correlation matrix and VIF scores.

Identified influential features via feature importance metrics.

3. Model Development & Tuning
Implemented three regressors:

Linear Regression: Addressed multicollinearity via Ridge/Lasso regularization.

Decision Tree Regressor: Controlled complexity with pruning parameters.

Random Forest Regressor: Performed extensive tuning using GridSearchCV to reduce variance and improve accuracy.

4. Model Evaluation
Evaluated models using metrics such as:

RMSE (Root Mean Squared Error)

MAE (Mean Absolute Error)

R² Score

Compared model performance on training and test sets to identify overfitting.

🧰 Technologies Used
Language: Python

Libraries: scikit-learn, pandas, numpy, matplotlib, seaborn, statsmodels

📊 Results
Random Forest achieved the best accuracy with the lowest overfitting.

Linear Regression was improved through multicollinearity fixes, making it suitable for transparent decision-making.

Decision Tree offered explainable predictions after pruning.
