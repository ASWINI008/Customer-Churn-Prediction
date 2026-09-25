# Customer Churn Analysis & Prediction

[![Live Demo](https://img.shields.io/badge/Live%20Demo-Streamlit-red?logo=streamlit)](https://customer-churn-prediction-dv8ca25dnusnszei8wyp9y.streamlit.app/)
[![Python](https://img.shields.io/badge/Python-3.x-blue?logo=python)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?logo=streamlit)](https://streamlit.io/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Machine%20Learning-orange)](https://scikit-learn.org/)

## 🌐 Live Demo

**Try the application here:**

👉 https://customer-churn-prediction-dv8ca25dnusnszei8wyp9y.streamlit.app/

---

## 📌 Project Overview

**Customer Churn Analysis & Prediction** is a Data Science and Machine Learning web application developed to analyze customer behavior, understand churn patterns, and predict whether a customer is likely to leave a service.

The project uses customer information such as tenure, contract type, monthly charges, total charges, internet service, payment method, technical support, online security, and other customer attributes.

A **Logistic Regression** Machine Learning model is used to predict customer churn.

The application is built using **Python and Streamlit** and provides an interactive interface for data analysis, visualization, customer prediction, and bulk prediction using an uploaded CSV file.

---

## 🎯 Objectives

The main objectives of this project are:

- Analyze customer data using Data Science techniques.
- Understand customer churn patterns.
- Identify customer groups with different churn behavior.
- Visualize important customer and subscription information.
- Predict whether a customer is likely to churn.
- Calculate the probability of customer churn.
- Classify customers into different risk levels.
- Allow users to upload their own customer dataset.
- Generate churn predictions for multiple customers.
- Provide downloadable prediction results.

---

## ✨ Features

### 1. 📊 Overview Dashboard

The Overview Dashboard provides a quick summary of the customer dataset.

It displays:

- Total Customers
- Churned Customers
- Active Customers
- Churn Rate
- Average Tenure
- Average Monthly Charges
- Average Total Charges
- Churn Distribution

The dashboard uses charts and summary statistics to make the data easier to understand.

---

### 2. 👥 Customer Data

The Customer Data section allows users to explore the available customer records.

Users can filter the data based on:

- Contract
- Internet Service
- Payment Method
- Churn Status

The filtered customer data can also be downloaded as a CSV file.

---

### 3. 📈 Churn Analysis

The Churn Analysis section explores the relationship between customer attributes and churn.

The analysis includes:

- Churn by Contract Type
- Churn by Internet Service
- Churn by Payment Method
- Churn by Tenure
- Monthly Charges
- Customer Characteristics

The results are displayed using simple and understandable visualizations.

---

### 4. 💡 Customer Insights

The Customer Insights section provides data-driven observations from the dataset.

Important factors analyzed include:

- Contract Type
- Customer Tenure
- Monthly Charges
- Internet Service
- Technical Support
- Online Security
- Payment Method

The insights are based on the actual values available in the dataset.

---

### 5. 🔮 Customer Churn Prediction

Users can enter individual customer information and obtain a churn prediction.

The prediction provides:

- Churn Prediction
- Churn Probability
- Risk Level

The customer is classified into:

| Risk Level | Churn Probability |
|------------|-------------------|
| Low Risk | 0% – 30% |
| Medium Risk | 31% – 60% |
| High Risk | 61% – 100% |

---

### 6. 🔄 What-If Churn Simulator

The What-If Simulator allows users to modify customer details and observe changes in the predicted churn probability.

Users can modify values such as:

- Contract
- Tenure
- Monthly Charges
- Internet Service
- Technical Support
- Online Security
- Payment Method

The application then generates an updated churn probability using the Machine Learning model.

---

### 7. 👤 Customer Segmentation

The application provides different customer groups based on their characteristics and behavior.

Examples include:

- New Customers
- Long-term Customers
- High-value Customers
- High Monthly Cost Customers
- At-risk Customers

This helps in understanding different customer groups in the dataset.

---

### 8. 📤 Upload & Predict

The **Upload & Predict** feature allows users to upload their own customer CSV file.

The application performs the following steps:

```text
Upload CSV
     ↓
Validate Data
     ↓
Preview Data
     ↓
Preprocess Data
     ↓
Apply Machine Learning Model
     ↓
Predict Churn
     ↓
Calculate Churn Probability
     ↓
Assign Risk Level
     ↓
Download Results
```

The prediction results include:

- Predicted Churn
- Churn Probability
- Risk Level

The complete results can be downloaded as a CSV file.

---

## 🤖 Machine Learning

The project uses **Logistic Regression** for customer churn prediction.

Logistic Regression is used because churn prediction is a classification problem where the customer can have one of two outcomes:

```text
Churn
Stay
```

### Machine Learning Workflow

```text
Customer Dataset
       ↓
Data Cleaning
       ↓
Exploratory Data Analysis
       ↓
Feature Selection
       ↓
Categorical Encoding
       ↓
Train-Test Split
       ↓
Logistic Regression
       ↓
Model Evaluation
       ↓
Churn Prediction
```

---

## 📊 Dataset

The dataset contains customer information related to their subscription, billing, and service usage.

### Important Dataset Attributes

| Feature | Description |
|---------|-------------|
| CustomerID | Unique customer identifier |
| Gender | Customer gender |
| Age | Customer age |
| SeniorCitizen | Senior citizen indicator |
| Tenure | Number of months with the service |
| MonthlyCharges | Monthly service charges |
| TotalCharges | Total charges paid by the customer |
| Contract | Type of customer contract |
| PaymentMethod | Customer payment method |
| InternetService | Type of internet service |
| TechSupport | Technical support availability |
| OnlineSecurity | Online security availability |
| PaperlessBilling | Paperless billing status |
| Churn | Customer churn status |

---

## 🧠 Model Inputs

The prediction model uses customer attributes such as:

```text
Gender
Age
SeniorCitizen
Tenure
MonthlyCharges
TotalCharges
Contract
PaymentMethod
InternetService
TechSupport
OnlineSecurity
PaperlessBilling
```

The same preprocessing approach is used for individual prediction and uploaded CSV prediction.

---

## 📈 Model Evaluation

The Logistic Regression model can be evaluated using:

- Accuracy
- Confusion Matrix
- Classification Report

These evaluation metrics are used to understand the performance of the classification model.

---

## 🛠️ Technologies Used

### Programming Language

- Python

### Data Processing

- Pandas
- NumPy

### Data Visualization

- Matplotlib
- Seaborn

### Machine Learning

- Scikit-learn

### Web Application

- Streamlit

### Deployment

- Streamlit Community Cloud

---

## 📁 Project Structure

```text
customer-churn-analysis/
│
├── app.py
├── model.py
├── customer_churn.csv
├── requirements.txt
├── README.md
│
└── assets/
    └── screenshots/
```

---

## ⚙️ Installation and Setup

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/customer-churn-analysis.git
```

### 2. Navigate to the Project Folder

```bash
cd customer-churn-analysis
```

### 3. Install Required Libraries

```bash
pip install -r requirements.txt
```

### 4. Run the Application

```bash
streamlit run app.py
```

The application will open in your default web browser.

---

## 📦 Requirements

The main libraries used in this project are:

```text
streamlit
pandas
numpy
matplotlib
seaborn
scikit-learn
```

These dependencies are also included in the `requirements.txt` file.

---

## 📤 Using Your Own Customer Data

The application provides an **Upload & Predict** feature for analyzing external customer data.

The uploaded CSV should contain the required customer attributes.

### Required Columns

```text
Gender
Age
SeniorCitizen
Tenure
MonthlyCharges
TotalCharges
Contract
PaymentMethod
InternetService
TechSupport
OnlineSecurity
PaperlessBilling
```

### Process

1. Upload the CSV file.
2. The application validates the required columns.
3. A preview of the uploaded data is displayed.
4. The data is processed using the existing preprocessing pipeline.
5. The Logistic Regression model generates predictions.
6. Churn probability is calculated.
7. A risk level is assigned.
8. The prediction results can be downloaded.

---

## 📥 Prediction Output

The uploaded customer data is returned with additional prediction information.

Additional columns include:

```text
Predicted Churn
Churn Probability
Risk Level
```

### Example

| Customer | Predicted Churn | Probability | Risk Level |
|----------|-----------------|-------------|------------|
| C001 | Yes | 72.4% | High |
| C002 | No | 18.7% | Low |
| C003 | Yes | 54.2% | Medium |

---

## 💼 Example Use Case

A company can use this application to analyze customer data and identify customers who have a higher estimated probability of leaving the service.

For example, the application can help identify:

- Customers with higher churn probability
- Customers with lower churn probability
- Contract groups with different churn rates
- Customer groups with different behavior
- Service-related churn patterns

The results can be exported for further analysis.

---

## 🔍 Project Workflow

The complete application follows this workflow:

```text
                    CUSTOMER DATA
                         │
                         ▼
                 DATA PREPROCESSING
                         │
                         ▼
              EXPLORATORY DATA ANALYSIS
                         │
                         ▼
                  DATA VISUALIZATION
                         │
                         ▼
                 FEATURE ENGINEERING
                         │
                         ▼
                 LOGISTIC REGRESSION
                         │
                         ▼
                  MODEL EVALUATION
                         │
                         ▼
                 CHURN PREDICTION
                         │
              ┌──────────┴──────────┐
              ▼                     ▼
       INDIVIDUAL USER        CSV UPLOAD
         PREDICTION            PREDICTION
              │                     │
              └──────────┬──────────┘
                         ▼
                  RISK CLASSIFICATION
                         │
                         ▼
                  DOWNLOAD RESULTS
```

---

## 🌐 Deployment

The application is deployed using **Streamlit Community Cloud**.

### Live Application

**Customer Churn Analysis & Prediction**

https://customer-churn-prediction-dv8ca25dnusnszei8wyp9y.streamlit.app/

The deployed application can be accessed directly through a web browser without installing Python or the required libraries locally.

---

## 🚀 Future Enhancements

Possible future improvements include:

- Comparing Logistic Regression with other classification algorithms.
- Improving model performance through feature engineering.
- Adding advanced customer segmentation.
- Adding real-time database integration.
- Adding automated reports.
- Adding monthly churn trend analysis.
- Adding model explainability.
- Deploying the application with a production database.
- Adding more customer behavior features.

---

## 📸 Application

The project contains an interactive web interface with:

- Dashboard
- Customer Data
- Churn Analysis
- Customer Insights
- Prediction
- What-If Simulator
- Customer Segmentation
- Upload & Predict
- Reports

---

## 👩‍💻 Author

**ASWINI S**

B.E. Computer Science and Engineering

**KIT – Kalaignar Karunanidhi Institute of Technology**

---

## 🔗 Project Links

### Live Application

https://customer-churn-prediction-dv8ca25dnusnszei8wyp9y.streamlit.app/

### GitHub Repository

Add your GitHub repository link here.

---

## 📄 License

This project is developed for educational and academic purposes.
