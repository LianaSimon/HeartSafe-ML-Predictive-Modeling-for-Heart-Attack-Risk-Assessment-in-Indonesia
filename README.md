# 🫀 HeartSafe ML: Predictive Modeling for Heart Attack Risk Assessment in Indonesia

##### “Leveraging AI to detect heart attack risk early and support preventive healthcare decisions.”



## 📌 Project Overview:

This project aims to develop a machine learning model that can accurately predict the risk of a heart attack using a rich healthcare dataset collected from patients in Indonesia. By leveraging clinical, lifestyle, and demographic indicators, we aim to identify at-risk individuals and support early intervention strategies in the healthcare system. With over 158,000 records and 28 features, this end-to-end solution includes data cleaning, preprocessing, outlier analysis, feature scaling, model training, evaluation, and performance comparison. The goal is to assist healthcare professionals in identifying high-risk individuals for timely intervention.



## 🎯 Objectives:

To build a supervised classification model that accurately identifies individuals at risk of heart attack using demographic, lifestyle, and clinical health indicators.


* Clean and preprocess a real-world healthcare dataset

* Handle missing values and outliers appropriately

* Encode categorical variables and scale numeric features

* Split the dataset into training and testing subsets

* Prepare the data for machine learning modeling 



##### 📁 Dataset Details



| Attribute           | Value                                                                          |
| ------------------- | ------------------------------------------------------------------------------ |
| **Name**            | Heart Attack Prediction in Indonesia                                           |
| **Source**          | [DeepDataLake Dataset #55](https://deepdatalake.com/details.php?dataset_id=55) |
| **Rows**            | 158,355                                                                        |
| **Columns**         | 28 (features + target)                                                         |
| **Target Variable** | `heart_attack` (0 = No, 1 = Yes)                                               |
| **Type**            | Supervised Classification                                                      |





### 🛠️ Stage 1: Dataset Collection


* Imported CSV data using pandas. Checked the head().

<img width="978" height="387" alt="image" src="https://github.com/user-attachments/assets/0fc510fc-ad21-4a32-ad4f-01bb58e7a0ea" />


* Explored structure using .info(),shape(), column names.

<img width="978" height="147" alt="image" src="https://github.com/user-attachments/assets/bbe19220-78f5-41cc-bd89-8459d307bc76" />

<img width="978" height="511" alt="image" src="https://github.com/user-attachments/assets/4cb5a970-3c04-4a80-8672-daf4bc3d4a26" />

<img width="978" height="607" alt="image" src="https://github.com/user-attachments/assets/5f83f336-5df6-4410-b939-30bcf496e015" />

Identified 28 features including demographic, clinical, and lifestyle variables


* Removed duplicate records

<img width="978" height="140" alt="image" src="https://github.com/user-attachments/assets/97ac31c3-e767-4e13-a587-463e9e503345" />

Verified structure and feature relevance



### 🧹 Stage 2: Preprocessing & Cleaning


✅ Missing Values

* alcohol_consumption had ~60% missing values → filled with 'Unknown'

* Preserved structure and retained potential behavioral insight

<img width="978" height="413" alt="image" src="https://github.com/user-attachments/assets/6139aef3-fc0f-4fdd-87d8-4ac50310d9f0" />

<img width="978" height="537" alt="image" src="https://github.com/user-attachments/assets/5bd0b185-43f3-4298-bc34-892e24228c84" />


✅ Categorical Encoding

* One-hot encoded all categorical variables

* Used drop_first=True to prevent multicollinearity

<img width="978" height="508" alt="image" src="https://github.com/user-attachments/assets/5f811288-627c-480d-a485-e5e2b2e93576" />


✅ Outlier Analysis

* Visualized numeric features with boxplots

* Outliers not removed, as they may indicate true medical risks (e.g., high blood pressure)

<img width="978" height="477" alt="image" src="https://github.com/user-attachments/assets/69e76834-c3aa-4dc5-9d5d-b1ccec8f0493" />

<img width="978" height="452" alt="image" src="https://github.com/user-attachments/assets/709ea2b7-919a-427b-8344-2867057141b9" />

<img width="978" height="459" alt="image" src="https://github.com/user-attachments/assets/0dc81225-b3f7-403a-98e8-a520641439f7" />

<img width="978" height="450" alt="image" src="https://github.com/user-attachments/assets/d3e7a7ba-2799-4a19-9880-d38c346f7b4e" />

<img width="978" height="452" alt="image" src="https://github.com/user-attachments/assets/266c1427-9608-4f2c-9202-81bba9ac123c" />

<img width="978" height="454" alt="image" src="https://github.com/user-attachments/assets/54f4438d-2a41-4c25-a86b-fa93a1b7e43c" />

<img width="978" height="448" alt="image" src="https://github.com/user-attachments/assets/8744bc27-7e2a-46f8-bd31-13c664939478" />

<img width="978" height="441" alt="image" src="https://github.com/user-attachments/assets/dc1e192a-ac63-4b9f-a9d1-01a9c09b4869" />

<img width="978" height="474" alt="image" src="https://github.com/user-attachments/assets/50599d4a-f0e1-4807-bfd3-5de9d0e072d7" />

<img width="978" height="303" alt="image" src="https://github.com/user-attachments/assets/26c3fea8-25a2-4e0b-879b-2d888421fcd2" />

<img width="978" height="612" alt="image" src="https://github.com/user-attachments/assets/a7a3ac6f-abba-4fcc-b29e-e09a73b62c6f" />


✅ Feature Scaling

* Standardized numeric features using StandardScaler

<img width="978" height="210" alt="image" src="https://github.com/user-attachments/assets/28d0ed75-f37e-45ac-aac4-d9ae5f415034" />


* Implemented using a Pipeline + ColumnTransformer

<img width="978" height="320" alt="image" src="https://github.com/user-attachments/assets/ed48b60f-7375-4421-84c2-3e1d2a878c72" />


✅ Train/Test Split

* Used train_test_split() with stratify=y to maintain class balance

* 80% training / 20% testing

<img width="978" height="229" alt="image" src="https://github.com/user-attachments/assets/414261ac-068b-4de2-b2ff-3928d692fb34" />



### 🤖 Stage 3: Machine Learning Modeling & Evaluation

Models Implemented:


* Logistic Regression

* Decision Tree Classifier

* Random Forest Classifier

* Support Vector Machine (SVM)

* Gradient Boosting Classifier



Evaluation Metrics:


* Accuracy

* Precision

* Recall

* F1-score

* ROC-AUC Score

<img width="978" height="428" alt="image" src="https://github.com/user-attachments/assets/ea1b93b2-5b0c-4c55-a89f-2332b13e298d" />

  
*****
Model Evaluation Example Output
Classification Report (Precision, Recall, F1-score)

Confusion Matrix

ROC-AUC Score for each model

Summary comparison table (sorted by ROC-AUC)

🔍 Sample Model Performance Summary (example)
Model	ROC-AUC
Random Forest	0.92
Gradient Boosting	0.91
Logistic Regression	0.88
SVM	0.87
Decision Tree	0.84

✅ Project Status
✅ Data collected and cleaned

✅ ML models built and compared

🔄 Next: Hyperparameter tuning, model selection, and deployment

💬 Key Learnings
Healthcare data often includes meaningful outliers — not just noise

Pipelines make preprocessing reproducible and scalable

One-hot encoding and stratified splitting are crucial for classification



