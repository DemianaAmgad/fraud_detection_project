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
