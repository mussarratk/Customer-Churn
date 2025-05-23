# Telecom Customer-Churn Project AI/ML- Framework AML SDK V2
- Use : Churn_Modeling.csv
in 3 framework - https://www.coursera.org/learn/foundations-of-ai-and-machine-learning/supplement/XtdhM/practice-activity-implementing-a-model-for-business-deployment
https://docs.google.com/spreadsheets/d/1u_25Gd_lli0m9SQwrfNZfDM0U4rJoiFz8gLRuFps1us/edit?gid=1228681343#gid=1228681343
https://www.coursera.org/learn/foundations-of-ai-and-machine-learning/lecture/y5Vv5/introduction-to-deployment-platforms

-------------------------------------------------------------------------------------------------------------------------
- component

![image](https://github.com/user-attachments/assets/950f88ac-b0d4-42ca-932a-7e6430638bdf)
![image](https://github.com/user-attachments/assets/04553556-c10f-41fe-a96f-2bf6d51b97bb)
![image](https://github.com/user-attachments/assets/cfe8d3b4-a3b6-47ab-85a0-abee60d25bd4)

![image](https://github.com/user-attachments/assets/360f2be0-2478-4f2b-baec-223c89947924)
![image](https://github.com/user-attachments/assets/555476f7-7fcd-41fd-afb4-aa88faf3fb71)

- Pipeline jobs
![image](https://github.com/user-attachments/assets/ce201a36-d2c4-4601-8e6f-74d767da46db)

![image](https://github.com/user-attachments/assets/aebf7857-30f5-4f29-a6f2-4977f5ea7421)
![image](https://github.com/user-attachments/assets/bf4cd8f4-9d59-4bfb-b7dc-ca1a9e93c4a0)

![image](https://github.com/user-attachments/assets/4f0da376-ba1f-4507-b1c7-2ee90be43efc)


- Experiment Jobs
![image](https://github.com/user-attachments/assets/fac1439d-3881-45f5-93dc-230da88f8878)


- Data
  -- churn csv I have created dataset by uploading csv
  -- churn_blob_csv_default - I have created external blob storage and container there and uploaded the dataset
![image](https://github.com/user-attachments/assets/d8c2416c-4d86-40e2-ad40-4c3ecf160e08)
![image](https://github.com/user-attachments/assets/dafac8fc-9985-46b4-b366-d834e52524b8)
![image](https://github.com/user-attachments/assets/27fcb838-3a00-4f35-8e6b-a22bb682a470)
![image](https://github.com/user-attachments/assets/7f286199-0c1d-4bda-9685-baed09a2931e)
![image](https://github.com/user-attachments/assets/cc9a127c-bd06-4f3d-95bc-02601679544b)
![image](https://github.com/user-attachments/assets/54ca06ab-4f87-4869-8eed-28f92eb8ee32)

![image](https://github.com/user-attachments/assets/33c769aa-c381-4f47-be96-1b370087cf90)


 - Custom Environment created

![image](https://github.com/user-attachments/assets/1947dde9-63eb-4633-950c-2359267740c5)
![image](https://github.com/user-attachments/assets/0f6e6264-15bc-45bb-b208-de8e6a18f15a)
![image](https://github.com/user-attachments/assets/d492cd0c-f177-4c06-a557-7e3c8a79b7da)


<details>

🔍 Understanding the Confusion Matrix
The heatmap you shared is a confusion matrix, used to evaluate classification model performance:

Predicted No (0)	Predicted Yes (1)
Actual No	1200 (TN)	230 (FP)
Actual Yes	140 (FN)	240 (TP)

Accuracy: (TP + TN) / Total = (1200 + 240) / (1200 + 230 + 140 + 240) ≈ 0.80

Precision (for Churn=1): TP / (TP + FP) = 240 / (240 + 230) ≈ 0.51

Recall (for Churn=1): TP / (TP + FN) = 240 / (240 + 140) ≈ 0.63

F1-score is moderate → indicating some imbalance or difficulty in predicting churn.

📊 Feature Correlation with Churn
You shared a sorted correlation list. The top negative correlations (features reducing churn likelihood) include:

Feature	Correlation
tenure	-0.35
Contract_Two year	-0.30
DeviceProtection_No internet service	-0.22
OnlineSecurity_Yes	-0.17
TechSupport_Yes	-0.16

Top positive correlations (features increasing churn likelihood):

Feature	Correlation
Contract_Month-to-month	+0.40
OnlineSecurity_No	+0.34
TechSupport_No	+0.33

✅ Top 5 Features for Churn Prediction
Based on absolute correlation:

tenure (↓ churn with more tenure)

Contract_Month-to-month (↑ churn)

OnlineSecurity_No (↑ churn)

TechSupport_No (↑ churn)

Contract_Two year (↓ churn)

These features are very predictive and should be prioritized.

🔧 Which Models to Use for Churn Prediction?
Given:

Binary classification

Some imbalance in churned vs non-churned

Mix of categorical and numerical features

Recommended models:

Random Forest Classifier

Handles both types of features well

Robust to overfitting

Easy to interpret via feature importance

XGBoost or LightGBM

Excellent performance on tabular data

Great for imbalanced data (can tune scale_pos_weight)

Supports built-in handling of missing data

Logistic Regression (baseline)

Quick to implement and benchmark

Useful for interpretability

⚙️ Hyperparameter Tuning Tips
Use GridSearchCV or RandomizedSearchCV. For imbalanced classification, focus on:

For RandomForest:

python
params = {
  'n_estimators': [100, 200],
  'max_depth': [10, 20, None],
  'min_samples_split': [2, 5],
  'class_weight': ['balanced']
}


For XGBoost:

params = {
  'n_estimators': [100, 300],
  'max_depth': [3, 6],
  'learning_rate': [0.01, 0.1],
  'scale_pos_weight': [1, 3, 5]  # important for imbalance
}

Let’s break this down in a structured way to help you understand what's going on with your churn prediction project, how to interpret the results, and which algorithms and tuning strategies can optimize your model.

---

### 🔍 **Understanding the Confusion Matrix**

The heatmap you shared is a **confusion matrix**, used to evaluate classification model performance:

|                | Predicted No (0) | Predicted Yes (1) |
| -------------- | ---------------- | ----------------- |
| **Actual No**  | 1200 (TN)        | 230 (FP)          |
| **Actual Yes** | 140 (FN)         | 240 (TP)          |

* **Accuracy**: `(TP + TN) / Total = (1200 + 240) / (1200 + 230 + 140 + 240) ≈ 0.80`
* **Precision (for Churn=1)**: `TP / (TP + FP) = 240 / (240 + 230) ≈ 0.51`
* **Recall (for Churn=1)**: `TP / (TP + FN) = 240 / (240 + 140) ≈ 0.63`
* **F1-score** is moderate → indicating some imbalance or difficulty in predicting churn.

---

### 📊 **Feature Correlation with Churn**

You shared a sorted correlation list. The top **negative correlations** (features reducing churn likelihood) include:

| Feature                                | Correlation |
| -------------------------------------- | ----------- |
| `tenure`                               | -0.35       |
| `Contract_Two year`                    | -0.30       |
| `DeviceProtection_No internet service` | -0.22       |
| `OnlineSecurity_Yes`                   | -0.17       |
| `TechSupport_Yes`                      | -0.16       |

Top **positive correlations** (features increasing churn likelihood):

| Feature                   | Correlation |
| ------------------------- | ----------- |
| `Contract_Month-to-month` | +0.40       |
| `OnlineSecurity_No`       | +0.34       |
| `TechSupport_No`          | +0.33       |

### ✅ **Top 5 Features for Churn Prediction**

Based on absolute correlation:

1. `tenure` (↓ churn with more tenure)
2. `Contract_Month-to-month` (↑ churn)
3. `OnlineSecurity_No` (↑ churn)
4. `TechSupport_No` (↑ churn)
5. `Contract_Two year` (↓ churn)

These features are very predictive and should be prioritized.

---

### 🔧 **Which Models to Use for Churn Prediction?**

Given:

* Binary classification
* Some imbalance in churned vs non-churned
* Mix of categorical and numerical features

**Recommended models:**

1. **Random Forest Classifier**

   * Handles both types of features well
   * Robust to overfitting
   * Easy to interpret via feature importance

2. **XGBoost or LightGBM**

   * Excellent performance on tabular data
   * Great for imbalanced data (can tune `scale_pos_weight`)
   * Supports built-in handling of missing data

3. **Logistic Regression (baseline)**

   * Quick to implement and benchmark
   * Useful for interpretability

---

### ⚙️ **Hyperparameter Tuning Tips**

Use `GridSearchCV` or `RandomizedSearchCV`. For imbalanced classification, focus on:

**For RandomForest:**

```python
params = {
  'n_estimators': [100, 200],
  'max_depth': [10, 20, None],
  'min_samples_split': [2, 5],
  'class_weight': ['balanced']
}
```

**For XGBoost:**

```python
params = {
  'n_estimators': [100, 300],
  'max_depth': [3, 6],
  'learning_rate': [0.01, 0.1],
  'scale_pos_weight': [1, 3, 5]  # important for imbalance
}
```

---

### 🧠 Evaluation Metrics

Since your model might struggle more with minority class (churned customers), use:

* **AUC-ROC**: Already mentioned (0.84, quite good!)
* **F1-score**: Better than accuracy when imbalance exists
* **Precision-Recall curve**: For assessing performance on positive (churn) class

---

Would you like help with writing optimized code for modeling or hyperparameter tuning using `RandomForest`, `XGBoost`, or as an Azure ML pipeline?


  
</details>

























--------------------------------------------------------------------------------------------------------------------------------------
<details>
  


- pipeline -
- Model development
  - - framework choice - the ability of framework to scale with project demands
  - - Model lifecycle
- deployment platform - AKS (Containerized application)- ACI - on cloud scalability: autoscaling, secure your model and integrate with other systems(DB, API, WServices), performance: speed - Powerful Solution
  - - Azure service - would you use to deploy an ML model as a web service that can be accessed by other applications via HTTP requests
- Maintain: App Insights and Monitor (Data Drift) - Continuous monitoring helps ensure the model remains accurate and effective, allowing for timely adjustments if performance issues arise.

  * AutoML - automatically select and tunes best performing model

  - Data access - API, Webscraping, DB, sensor, IoT git, link, external, open source
  
      *  RAG enables AI to retrieve fresh (real-time), relevant data from external sources, enhancing the generated content’s accuracy and relevance.
      *  https://www.coursera.org/learn/foundations-of-ai-and-machine-learning/supplement/AMEqp/comparison-of-data-sources-for-rag-and-traditional-ml-pipelines

 - The main purpose of data encryption is to safeguard sensitive data from unauthorized access, ensuring privacy and security.RBAC is a widely used method for managing user access to sensitive data based on their roles within the organization.
 - Data Management
  - - Data quality -To improve the accuracy and reliability of the AI models
   -- Data governance ensures responsible usage of data, compliance with regulations, and overall data quality throughout its life cycle.
   -- https://www.coursera.org/learn/foundations-of-ai-and-machine-learning/supplement/Nu3A7/practice-activity-auditing-ml-code-for-security-vulnerabilities
   -- https://www.coursera.org/learn/foundations-of-ai-and-machine-learning/supplement/Nu3A7/practice-activity-auditing-ml-code-for-security-vulnerabilities

- ML framework
  * Tensorflow, Pytorch, Keras, Scikit-learn, Apache Spark MLlib, Azure ML SDK
  * Define - Model Type : deep learning task - tensflow/pytorch - large scale - GPU/TPU - cloud - edge
            -- classical ml algorithm - DT, SVM - small/medium - CPU - sk-learn
    
- Azure
    - https://www.coursera.org/learn/foundations-of-ai-and-machine-learning/supplement/qJvih/selecting-the-right-model-deployment-strategy-in-microsoft-azure
 
- Pretrained LLM Model -  Pretrained LLMs can be fine-tuned for customer service tasks, allowing them to understand and respond to queries quickly and accurately, leading to improved customer satisfaction.
    -- T5 - analyze
    -- gpt n bert

- Implementing Models - prep, deploy, monitor - https://www.coursera.org/learn/foundations-of-ai-and-machine-learning/supplement/mwEGm/introduction-to-implementing-models




</details>
