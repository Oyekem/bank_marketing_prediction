<a id="top"></a>

# Bank Term Deposit Prediction

**Predicting which bank clients are likely to subscribe to a term deposit using demographic, financial, and marketing data.**




## 📖 Table of Contents

1. [Objective](#objective)
2. [Dataset](#dataset)
3. [Workflow](#workflow)
   - [Data Preprocessing](#data-preprocessing)
   - [Model Selection](#model-selection)
   - [Hyperparameter Tuning](#hyperparameter-tuning)
   - [Threshold Tuning](#threshold-tuning)
   - [Model Evaluation](#model-evaluation)
   - [Feature Importance](#feature-importance)
   - [Final Model & Deployment](#final-model--deployment)
   - [Business Recommendations](#business-recommendations)
   - [Additional Notes](#additional-notes)
4. [Project Structure](#project-structure)



## 🎯 Objective

The goal of this project is to **help the bank target potential subscribers efficiently**, maximizing marketing campaign effectiveness while minimizing wasted resources.

We predict whether a client will subscribe to a term deposit (`y = 1`) using a mix of **numeric and categorical features**.



## 📊 Dataset

* **Source:** Bank marketing dataset
* **Features:** Demographic, financial, and campaign-related
* **Target Variable:** `y` (binary)

  * `1` → Subscribed
  * `0` → Not subscribed
* **Challenge:** Imbalanced dataset (subscribers are minority class)



## 🛠 Workflow

## 1️⃣ Data Preprocessing

* **Encoding:** `OneHotEncoder` for categorical variables
* **Scaling:** `StandardScaler` for numeric features
* **Imbalance Handling:** SMOTE applied to training data
* **Train-Test Split:** 80/20

**Outcome:** Preprocessed dataset ready for model training

## 2️⃣ Model Selection

* **Candidate Models:** Logistic Regression, Random Forest, XGBoost
* **Evaluation Metrics:** Accuracy, Precision, Recall, F1-score, ROC-AUC
* **Final Model:** **XGBoost** – best balance of recall and F1-score for minority class

## 3️⃣ Hyperparameter Tuning

* **Method:** `RandomizedSearchCV`
* **Tuned Parameters:** `n_estimators`, `max_depth`, `learning_rate`, `subsample`, `colsample_bytree`, `gamma`, `min_child_weight`
* **Outcome:** Improved recall for subscribers

## 4️⃣ Threshold Tuning

* **Default Threshold:** 0.5
* **Optimal Threshold:** 0.55
* **Rationale:** Increases recall for subscribers while maintaining reasonable precision, minimizing missed opportunities in marketing campaigns

## 5️⃣ Model Evaluation

* **ROC-AUC:** 0.924 → Strong discriminative ability
* **Precision-Recall Curve:** Confirms trade-off between recall and precision
* **Confusion Matrix:** Accurately identifies most subscribers while maintaining high accuracy for non-subscribers


## 6️⃣ Feature Importance

**Top 10 Predictors from XGBoost:**

1. `duration` – Length of the last contact (seconds)
2. `previous` – Number of contacts before this campaign
3. `balance` – Average yearly balance of client
4. `pdays` – Days since last contact from previous campaign
5. `age` – Client age
6. `contact` – Communication type (cellular/telephone)
7. `job` – Client job type
8. `campaign` – Contacts during this campaign
9. `marital` – Marital status
10. `education` – Education level

**Business Insight:**

* Longer contact duration & higher previous interactions → higher chance of subscription
* Higher balance → better financial capacity → prioritize for marketing
* Target marketing by age, job type, and contact method


## 7️⃣ Final Model & Deployment

* **Pipeline Saved:** `models/xgboost_bank_pipeline.pkl`
* **Includes:** Data preprocessing + XGBoost model
* **Ready For:** Future predictions, production deployment, API integration


## 8️⃣ Business Recommendations

* Prioritize clients flagged as likely subscribers to maximize ROI
* Target marketing campaigns based on **duration**, **balance**, and engagement indicators
* Regularly retrain model with new data for evolving client behavior
* Adjust thresholds depending on campaign risk tolerance


## 9️⃣ Notes

* Model focuses on **recall for minority class** → reduces missed subscribers
* Moderate precision → some false positives acceptable for broad campaigns


## 📂 Project Structure

```
├── data/
│   └── bank_marketing.csv
├── models/
│   └── xgboost_bank_pipeline.pkl
├── notebooks/
│   └── exploratory_analysis.ipynb
├── src/
│   └── preprocessing.py
│   └── modeling.py
├── README.md
└── requirements.txt
```
