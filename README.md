# Project Overview

## Objective: Predict whether a bank client will subscribe to a term deposit using demographic, financial, and marketing campaign features.
## Goal: Help the bank target potential subscribers efficiently, maximizing the effectiveness of marketing campaigns while minimizing wasted resources.

** Dataset: Bank marketing dataset (numeric + categorical features, imbalanced target).** 
Target Variable: y – binary indicator (1 = subscribed, 0 = not subscribed).



Workflow

1️⃣ Data Preprocessing

Encoding categorical variables: OneHotEncoder

Scaling numeric variables: StandardScaler

Handling imbalance: SMOTE applied to training data

Train-test split: 80/20

Outcome: Preprocessed dataset ready for model training.



2️⃣ Model Selection

Candidate models: Logistic Regression, Random Forest, XGBoost

Evaluation metrics: Accuracy, Precision, Recall, F1-score, ROC-AUC

Final model selected: XGBoost – best balance of recall and F1-score for minority class (subscribers).



3️⃣ Hyperparameter Tuning

Method: RandomizedSearchCV

Purpose: Find the best combination of XGBoost hyperparameters (n_estimators, max_depth, learning_rate, subsample, colsample_bytree, gamma, min_child_weight).

Outcome: Tuned XGBoost model with improved recall for subscribers.



4️⃣ Threshold Tuning

Default threshold: 0.5

Optimal threshold chosen: 0.55

Rationale: Maximizes recall (minority class) while keeping precision reasonable. Helps reduce the risk of missing potential subscribers in marketing campaigns.



5️⃣ Model Evaluation

Metrics Comparison Table:

Stage
Baseline
After Tuning
Threshold Tuned

Confusion Matrix: Shows that the model correctly identifies most subscribers (minority class) while maintaining high accuracy for non-subscribers.

ROC Curve & Precision-Recall Curve:

ROC-AUC = 0.924 → strong overall discriminative ability

Precision-recall curve confirms a trade-off between recall and precision for minority class



6️⃣ Feature Importance (Project-Specific)

Top 10 features influencing subscription (from your XGBoost model):

1. duration – length of the last contact (seconds)
2. previous – number of contacts performed before this campaign
3. balance – average yearly balance of the client
4. pdays – number of days since last contact from previous campaign
5. age – client age
6. contact – type of contact communication (cellular/telephone)
7. job – type of job (e.g., admin, management)
8. campaign – number of contacts performed during this campaign
9. marital – marital status
10. education – level of education

Business Insight:

Clients with longer last-contact durations and higher previous interactions are more likely to subscribe → prioritize them for follow-ups.
Financial capacity features like balance also strongly predict subscription → target marketing campaigns toward clients with higher balances.
Campaign strategy can focus on specific age groups, job types, and communication channels identified as important.
Using these top predictors helps the bank allocate marketing resources efficiently and maximize response rates.

7️⃣ Final Model & Deployment

Pipeline saved: models/xgboost_bank_pipeline.pkl

Pipeline includes: Data preprocessing + XGBoost model

Ready for: Future predictions or deployment in production (API, web service, batch scoring)



8️⃣ Business Recommendations

1. Prioritize clients flagged as potential subscribers (class 1) to maximize campaign ROI.

2. Features such as duration and balance indicate customer engagement and financial capacity – target marketing accordingly.

3. Regularly retrain the model on new data to adapt to changing client behavior.

4. Consider threshold adjustments depending on campaign risk tolerance (higher recall vs higher precision).



9️⃣ Additional Notes

Model focuses on recall for minority class (subscribers) to reduce missed opportunities.

Precision is moderately lower – meaning some predicted subscribers may not convert; acceptable for mass marketing campaigns.
<img width="987" height="32766" alt="image" src="https://github.com/user-attachments/assets/04335956-aec7-4e0c-8dbb-72daaf7438fe" />
