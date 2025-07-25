#🫀 HeartSafe ML: Predictive Modeling for Heart Attack Risk Assessment in Indonesia

### “Leveraging AI to detect heart attack risk early and support preventive healthcare decisions.”


📌 Project Overview
This project aims to develop a machine learning model that can accurately predict the risk of a heart attack using a rich healthcare dataset collected from patients in Indonesia. By leveraging clinical, lifestyle, and demographic indicators, we aim to identify at-risk individuals and support early intervention strategies in the healthcare system.



🎯 Objectives
Clean and preprocess a real-world healthcare dataset

Handle missing values and outliers appropriately

Encode categorical variables and scale numeric features

Split the dataset into training and testing subsets

Prepare the data for machine learning modeling (Stage 3)



📁 Dataset

Source: DeepDataLake Dataset ID 55

Name: Heart Attack Prediction in Indonesia

Rows: 158,355

Columns: 28 (including the target heart_attack)

Type: Classification (0 = No Heart Attack, 1 = Heart Attack)


🔧 Stage 1: Dataset Collection
Imported the dataset using pandas

Inspected column names, data types, and missing values

Checked for duplicate records

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

Used stratified sampling to maintain class balance in heart_attack

✅ Current Project Status
✔ Preprocessing complete
⏳ Ready to proceed to Stage 3: Machine Learning Modeling & Evaluation


