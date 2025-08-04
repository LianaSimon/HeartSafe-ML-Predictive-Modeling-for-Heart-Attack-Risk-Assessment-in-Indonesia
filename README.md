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



🧼 Stage 2: Preprocessing & Cleaning
🔹 Handling Missing Values
Missing values in alcohol_consumption (~60%) were filled with "Unknown" to retain behavioral information

No other columns had significant missing data

🔹 Categorical Encoding
Identified object-type columns and applied one-hot encoding

drop_first=True used to prevent multicollinearity

🔹 Outlier Detection
Used boxplots to visualize potential outliers in all numeric features

No removal was applied due to the clinical significance of outliers in healthcare (e.g., high blood pressure may indicate actual heart attack risk)

🔹 Feature Scaling
Applied StandardScaler only to numeric features using a Pipeline with ColumnTransformer

This standardization ensures consistent scale for models sensitive to feature magnitude

🔹 Train/Test Split
Split data into 80% training and 20% testing sets

Used stratified sampling to maintain class balance in heart_attack.


🚀 Stage 3: Model Building & Evaluation — Overview

✅ Objective:
Build and evaluate multiple classification models to predict heart attack risk based on the preprocessed healthcare data.


✅ ML Workflow Plan
🔹 1. Models to Implement:
We'll use these popular supervised classification algorithms:

Logistic Regression

Decision Tree Classifier

Random Forest Classifier

Support Vector Machine (SVM)

Gradient Boosting (e.g., XGBoost or GradientBoostingClassifier)

🔹 2. Evaluation Metrics:
We'll compare the models using:

Accuracy

Precision

Recall

F1-score

ROC-AUC

🔹 3. Best Model Selection:
After evaluating all models, we’ll:

Compare them side-by-side in a summary table

Select the best-performing one based on ROC-AUC and F1-score

Use it for prediction on test data


