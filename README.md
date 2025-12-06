## 1. Introduction

### 1.1 Background & Motivation

Healthcare fraud is a major source of financial loss in public insurance programs such as Medicare, with estimated losses exceeding **$68 billion per year**. These fraudulent activities include unnecessary procedures, upcoded services, duplicate billing, and claims submitted for patients who did not actually receive care.

Traditional **rule-based detection systems** rely on manually crafted rules (e.g., fixed thresholds on claim amounts or simple pattern flags). While these systems can catch obvious fraud, they struggle with:

- **Evolving fraud strategies**, where providers adapt behavior to avoid known rules  
- **Complex, multi-dimensional patterns** that span patients, procedures, time, and provider behavior  
- **Scalability**, as maintaining and updating large rule sets becomes costly and inefficient  

To address these limitations, this project applies **machine learning** to detect high-risk providers in a **data-driven and explainable** way. Instead of relying only on static rules, the model learns patterns of suspicious behavior from historical claim and label data.

The ultimate motivation is to help healthcare payers prioritize investigations, reduce financial losses, and improve the integrity of healthcare delivery.

---

### 1.2 Problem Definition

We frame healthcare provider fraud detection as a **supervised binary classification problem**, where the goal is to predict whether a given provider is:

- **Fraudulent (1)**  
- **Non-Fraudulent (0)**  

Key characteristics of the problem:

- **Highly imbalanced data**: Only about **10%** of providers in the training data are labeled as fraudulent, meaning standard accuracy is not a reliable metric.
- **Multi-table relational structure**:  
  - `Train_Beneficiarydata.csv` contains **patient-level** demographic and clinical information.  
  - `Train_Inpatientdata.csv` and `Train_Outpatientdata.csv` contain **claim-level** information for hospital and outpatient visits.  
  - `Train_labels.csv` provides **provider-level fraud labels**.  
- **Provider as the modeling unit**: Each provider can be associated with **hundreds or thousands of claims** across many beneficiaries, requiring careful **merging and aggregation** to build provider-level features suitable for modeling.

Formally, given aggregated features \( x_{provider} \) engineered from all associated beneficiaries and claims, the task is to learn a function:

\[
f(x_{provider}) \rightarrow \{\text{Fraud}, \text{Non-Fraud}\}
\]

that identifies high-risk providers with good precision and recall.

---

### 1.3 Project Objectives

This project aims to design and evaluate an **end-to-end fraud detection pipeline** for healthcare providers. The main objectives are:

- **Build a complete fraud detection workflow**  
  Construct a pipeline that covers data loading, cleaning, aggregation, feature engineering, modeling, evaluation, and error analysis.

- **Engineer provider-level features from relational data**  
  Derive meaningful provider-level features from **beneficiary**, **inpatient**, and **outpatient** tables, capturing financial behavior, claim patterns, and patient characteristics.

- **Handle class imbalance explicitly**  
  Apply appropriate techniques (e.g., class weighting, resampling) to mitigate the impact of the strong imbalance between fraudulent and non-fraudulent providers.

- **Train and compare multiple models**  
  Implement baseline models (e.g., **Logistic Regression**, **Random Forest**) and a stronger **gradient boosting model** (e.g., XGBoost / LightGBM), and compare their performance using metrics suited for imbalanced data.

- **Provide interpretable and actionable results**  
  Use feature importance, model explanations, and detailed error analysis to understand:
  - Which behaviors are most associated with fraud  
  - Which types of providers are misclassified (false positives/false negatives)  
  - How the model could be used in practice to prioritize investigations

Together, these objectives support the larger goal: **detect high-risk providers in an explainable, data-driven way that is useful for real-world fraud investigation teams.**

## 2. Data Understanding & Exploration

### 2.1 Dataset Overview

The project is built on a multi-table relational dataset commonly used for healthcare fraud detection. The dataset consists of the following files:

- **Train_Beneficiarydata.csv**  
  Contains patient demographic information (age, gender), chronic condition indicators, and mortality flags.

- **Train_Inpatientdata.csv**  
  Contains inpatient hospitalization claims, including diagnosis codes, procedure codes, lengths of stay, physician information, and reimbursement amounts.

- **Train_Outpatientdata.csv**  
  Contains outpatient or clinic visit claims with similar fields to the inpatient dataset but typically shorter encounters.

- **Train_labels.csv**  
  Provides **provider-level** fraud labels (1 = fraudulent, 0 = non-fraudulent).  
  This is the target variable used for supervised learning.

These tables must be merged and aggregated at the **provider level**, which is the primary unit of prediction.

---

### 2.2 Data Relationships

The dataset is relational, with key identifiers linking the tables:

- **BeneID**  
  Links each beneficiary to their inpatient and outpatient claims.

- **Provider**  
  Links all claims and beneficiary interactions to the provider who submitted them.  
  Each provider has exactly one label in `Train_labels.csv`.


This structure requires consolidation across thousands of claim records into one row per provider through feature engineering.

---

### 2.3 Data Quality Checks

During initial exploration, several quality issues must be identified and handled:

- **Missing Values**  
  - Some chronic condition indicators are missing for certain beneficiaries.  
  - Mortality indicators may be absent or partially filled.

- **Inconsistent or Extreme Claim Amounts**  
  - Some inpatient and outpatient reimbursement amounts have extreme outliers, likely due to rare or complex procedures.  
  - Zero or negative reimbursement amounts may occur due to adjustments.

- **Duplicate Entries**  
  - Some beneficiaries appear multiple times in beneficiary data.  
  - Duplicate or near-duplicate claim entries exist in the claims tables.

- **Skewed Distributions**  
  - Claim amounts are right-skewed, which is typical in healthcare cost data.  
  - Length-of-stay distributions show long tails.  
  - Some providers file unusually high numbers of claims.

Capturing, cleaning, and understanding these inconsistencies is crucial before aggregation and modeling.

---

### 2.4 Exploratory Data Analysis (EDA)

Several important observations emerged during EDA:

- **Class Imbalance**  
  Fraudulent providers represent only ~10% of all labeled providers.  
  This confirms the need for metrics such as recall, PR-AUC, and F1-score.

- **Claim Amount Distributions**  
  Inpatient and outpatient claim costs follow **right-skewed** distributions, with some providers submitting extremely high-cost procedures.

- **Provider-Level Behavior Patterns**  
  Aggregated features show that fraudulent providers often have:  
  - Higher average claim costs  
  - Greater diversity in procedure and diagnosis codes  
  - Higher number of patients per provider  

- **Correlation Heatmap**  
  Engineered features (e.g., total reimbursement, number of diagnoses, average length of stay) exhibit correlations that indicate clusters of behavior (financial, procedural, beneficiary-related).

- **Outlier Analysis**  
  Outliers in cost, physician count, and patient volume may indicate unusual behavior but require careful interpretation to avoid punishing legitimate specialized providers.

These EDA findings guide the feature engineering and model design decisions in later sections.

## 3. Feature Engineering & Aggregation Strategy

### 3.1 Why Aggregation Is Needed

The original dataset is at the **claim level** and **beneficiary level**, while the prediction target (`Train_labels.csv`) is at the **provider level**.  

A single provider may:

- Treat hundreds of beneficiaries  
- Submit thousands of inpatient and outpatient claims  
- Generate high-dimensional, repetitive, and noisy raw data

Machine learning models require a **fixed-length feature vector per provider**.  
Thus, we aggregate inpatient, outpatient, and beneficiary information into **provider-level features**, summarizing financial behavior, care patterns, and patient characteristics.

---

### 3.2 Aggregation Choices (Required Features)

The following groups of engineered features are essential and widely used in healthcare fraud detection.  
They provide a comprehensive behavioral profile for each provider.

---

#### **A. Claim Volume Features**
Capture provider activity level:

- Number of inpatient claims  
- Number of outpatient claims  
- Total number of unique beneficiaries  
- Ratio of inpatient to outpatient claims  

These help detect unusually high utilization behavior.

---

#### **B. Financial Metrics**
Fraudulent providers often display irregular financial patterns.

Key aggregated metrics:

- **Total reimbursement amount**  
- **Average claim cost**  
- **Maximum claim amount submitted**  
- **Standard deviation of claim amounts**  
- **Ratio of inpatient-to-outpatient cost**  
- **Total reimbursement per beneficiary**

These features highlight billing intensity and variability.

---

#### **C. Behavioral & Procedural Patterns**
These describe provider practices:

- **Average length of stay (inpatient)**  
- **Number of distinct diagnosis codes**  
- **Number of distinct procedure codes**  
- **Number of unique physicians used across claims**  
- **Frequency of specific procedure categories** (if available)

Fraudulent providers often have:

- unusually diverse procedure code patterns  
- large physician networks  
- inflated lengths of stay or procedure complexity  

---

#### **D. Beneficiary-Level Features**
Irregular patterns among treated beneficiaries can indicate fraud.

Important engineered features:

- **Average patient age**  
- **Percentage of beneficiaries with chronic conditions**  
- **Number or proportion of deceased beneficiaries still receiving claims**  
  *(strong fraud indicator)*  
- **Average number of claims per beneficiary**  
- **Chronic condition mix complexity**

These help identify providers who treat unusually risky or suspicious patient populations.

---

### 3.3 Feature Cleaning & Encoding

Several preprocessing steps are required after aggregation:

- **Handling Missing Values**  
  - Impute missing chronic condition flags  
  - Replace missing diagnosis code fields with a placeholder  

- **Encoding Categorical Variables**  
  - One-hot encode categorical fields such as gender or procedure categories  
  - Convert physician IDs into counts rather than explicit categories (to avoid high dimensionality)

- **Normalization / Standardization**  
  - Financial features like claim costs often span large ranges  
  - Standardizing improves model stability for logistic regression or distance-based models  

- **Outlier Treatment**  
  - Extreme claim amounts may distort averages  
  - Winsorization or log transformation may be applied if needed

These steps ensure all features are clean, consistent, and machine-learning ready.

---

### 3.4 Final Aggregated Dataset

After feature engineering, each provider is represented by a **single row**, containing:

- Aggregated claim metrics  
- Financial summaries  
- Beneficiary characteristics  
- Procedural and physician diversity indicators  
- Behavioral patterns  
- Final fraud label  

This structured, provider-level dataset enables effective modeling and explainability.


## 4. Handling Class Imbalance

### 4.1 Problem Overview

The fraud detection dataset is **highly imbalanced**, with only about **10%** of providers labeled as fraudulent.  
This imbalance introduces several challenges:

- **Accuracy becomes misleading**  
  A naïve model predicting "non-fraud" for all providers would achieve ~90% accuracy but detect **zero fraud cases**, making it useless.

- **Models may be biased toward the majority class**  
  Without correction, many models learn to ignore minority-class signals.

- **Recall is critical**  
  Missing fraudulent providers (false negatives) leads to significant financial losses.

For these reasons, specialized imbalance-handling strategies are essential.

---

### 4.2 Techniques Considered

Three main categories of imbalance correction techniques were evaluated:

#### **A. Class Weighting**
Adjusts model training to penalize misclassification of the minority class more heavily.

- Simple and stable  
- Works well for tree-based models  
- Does not modify the dataset  
- Recommended for XGBoost, LightGBM, Random Forest, Logistic Regression  

Formula for weighted loss:

\[
w_{\text{fraud}} = \frac{N}{2 \times N_{\text{fraud}}}, \quad 
w_{\text{nonfraud}} = \frac{N}{2 \times N_{\text{nonfraud}}}
\]

---

#### **B. Oversampling (e.g., SMOTE)**  
Generates synthetic minority-class samples.

- Helps logistic regression, SVM  
- Can improve recall  
- **Risk:** Synthetic data may introduce unrealistic patterns  
- Not ideal for financial or irregular claim distributions

---

#### **C. Undersampling**
Reduces the number of majority-class samples.

- Fast and simple  
- Can remove informative samples  
- Useful only for baseline models

---

### 4.3 Chosen Strategy & Justification

For this project, we adopted **class weighting** as the primary imbalance-handling method because:

- It maintains the original data distribution and provider behavior.  
- It avoids creating artificial synthetic patterns that reduce interpretability.  
- It naturally integrates with tree-based models (e.g., Gradient Boosting, Random Forest).  
- It supports strong performance on recall and F1-score without destabilizing the feature space.

Oversampling was tested as part of experimentation but not selected for final modeling due to interpretability and reliability concerns.

---

### 4.4 Impact on Evaluation

Using class weighting improves:

- **Recall of fraudulent providers**  
- **F1-score**, especially for the minority class  
- **PR-AUC**, the most meaningful metric for imbalanced classification  

This ensures the model identifies a meaningful proportion of fraudulent providers while keeping false positives at a manageable level.


## 5. Modeling

### 5.1 Baseline Models

To establish a performance baseline and understand the data characteristics, we trained two foundational models:

#### **A. Logistic Regression**
- Interpretable linear model suitable for tabular data  
- Provides direct insight into feature importance via coefficients  
- Useful for understanding directionality (positive/negative impact of features)  
- Serves as a sanity check for whether aggregated features contain predictive signals

Given the imbalanced nature of the dataset, logistic regression was trained with:
- Class weights enabled  
- Standardized numerical features  

---

#### **B. Random Forest**
- Robust non-linear model suitable for datasets with mixed feature types  
- Handles outliers and skewed distributions better than linear models  
- Naturally captures feature interactions  
- Provides feature importance based on impurity reduction  

This model helps establish a non-linear baseline and identify which feature groups carry most predictive power.

---

### 5.2 Primary Model: Gradient Boosting (XGBoost / LightGBM)

The final model selected for deployment was a **Gradient Boosting Machine (GBM)**, due to its strong performance on structured, financial, and behavioral datasets.

#### **Why Gradient Boosting?**
- Excellent performance on tabular data  
- Captures non-linear patterns and interactions between features  
- Handles skewed distributions without heavy preprocessing  
- Supports class weighting  
- Provides interpretable outputs using feature importance and SHAP values  
- Robust to noise in aggregated financial metrics  

GBM models are widely used in real-world fraud detection systems because they balance **accuracy, recall, and interpretability**.

---

### 5.3 Hyperparameter Tuning

To optimize model performance, we performed systematic hyperparameter tuning using **randomized search** and cross-validation.

Key parameters tuned:

- **n_estimators** — number of boosting rounds  
- **max_depth** — depth of each tree (controls model complexity)  
- **learning_rate** — shrinkage applied to each tree’s contribution  
- **subsample** — fraction of samples used per tree  
- **colsample_bytree** — fraction of features used per tree  
- **min_child_weight / min_data_in_leaf** — controls overfitting  
- **scale_pos_weight** — class imbalance ratio  

Cross-validation strategy:

- **5-fold cross-validation**  
- Stratified folds to preserve fraud distribution  

This provides stable estimates of generalization performance.

---

### 5.4 Model Interpretability

Understanding *why* a provider is flagged as fraudulent is crucial for real-world deployment.

We used:

- **Feature Importance**  
  Identifies which aggregated behaviors most strongly influence predictions.

- **SHAP (SHapley Additive exPlanations)**  
  Provides local explanations for individual predictions.  
  Investigators can review whether a provider was flagged due to:
  - unusually high claim costs  
  - excessive procedure diversity  
  - abnormal inpatient/outpatient ratios  
  - patients with inconsistent chronic condition patterns  

These interpretability tools bridge the gap between machine learning outputs and human-led fraud investigations.


## 6. Evaluation

### 6.1 Validation Approach

To ensure reliable performance assessment—especially under severe class imbalance—we used **stratified 5-fold cross-validation**.  
This approach:

- Preserves the fraud vs. non-fraud ratio in each fold  
- Reduces variance in evaluation metrics  
- Prevents overly optimistic results from lucky/non-representative splits  

For final testing, an 80/20 train-test split was used after confirming cross-validation stability.

---

### 6.2 Evaluation Metrics

Because fraud detection is an imbalanced classification problem, accuracy is not meaningful.  
Instead, we rely on metrics that directly assess minority-class performance:

- **Precision (Fraud class)**  
  Measures how many flagged providers are actually fraudulent.  
  High precision → fewer wasted investigations.

- **Recall (Fraud class)**  
  Measures how many fraudulent providers are successfully detected.  
  High recall → fewer missed fraudulent actors.

- **F1-Score**  
  Harmonic mean of precision and recall.  
  Useful for comparing models under imbalance.

- **ROC-AUC**  
  Measures ranking performance but can be misleading under high imbalance.

- **PR-AUC (Recommended)**  
  Most informative metric → focuses directly on minority-class predictions.

The chosen primary metric: **PR-AUC**, supported by F1 and recall.

---

### 6.3 Confusion Matrix Interpretation

A confusion matrix provides deeper insight into model behavior:

- **True Positives (TP):** Fraud correctly detected  
- **False Positives (FP):** Non-fraud flagged as fraud  
- **True Negatives (TN):** Non-fraud correctly ignored  
- **False Negatives (FN):** Fraud that the model failed to detect  

Interpretation for fraud detection:

- **High FP** → Waste investigation resources  
- **High FN** → Missed fraud → Financial losses

Given domain sensitivity, the priority is **minimizing FN** (maximizing recall), while maintaining an acceptable FP rate.

---

### 6.4 Error Analysis

Detailed error analysis helps understand weaknesses and potential improvements.

#### **False Positives (examples):**
- Some legitimate providers were flagged due to:
  - Very high claim diversity  
  - Large numbers of distinct physicians  
  - Exceptional patient populations (e.g., complex chronic conditions)  
- These providers may operate in specialized or high-complexity fields, mimicking fraud patterns unintentionally.

#### **False Negatives (examples):**
- Some fraudulent providers were missed because:
  - They had mostly normal behavior except for a few suspicious high-cost claims  
  - Their patient volume was low, reducing statistical signals  
  - Fraudulent behavior was subtle and blended into typical patterns

This analysis highlights the need for future improvements such as:
- More granular time-series modeling  
- Referral-pattern network features  
- Anomaly detection alongside supervised modeling

---

### 6.5 Model Limitations

The following limitations should be acknowledged:

- **Complex multi-table relationships** mean some subtle interactions may be lost during aggregation.
- **Some fraud behaviors are intentional and adversarial**, making them hard to detect with supervised ML alone.
- **Imbalanced data** still challenges model sensitivity, even with class weighting.
- **Static tabular features** may miss dynamic fraud patterns (e.g., abrupt spikes in claim volume).

These limitations motivate the future work and improvements described in Section 7.


