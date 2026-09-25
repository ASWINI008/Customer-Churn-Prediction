# Customer Churn Analysis and Prediction Dashboard

A human-designed, professional Data Science and Machine Learning analytics web application built with **Python**, **Pandas**, **NumPy**, **Matplotlib**, **Seaborn**, **Scikit-learn**, and **Streamlit**.

Designed specifically to emulate a real college/business analytics project (`#F8FAFC` background, pure white cards with light borders, navy headings, charcoal body text, and professional blue accents), this application provides deep dataset exploration, risk segmentation, scenario simulations, customer behavioral segmentation, and machine learning classification using a **Logistic Regression** model.

---

## 📌 Project Overview & Objectives

In subscription-based industries like telecommunications, customer retention is vital to long-term profitability. Retaining an existing customer is significantly less expensive than acquiring a new one.

This application enables users to:
1. **Explore Churn Patterns**: Dynamic KPI cards, Donut chart, Customer Snapshot, and compact Activity tables.
2. **Filter & Search Customer Profiles**: Search individual Customer IDs to view full demographic profiles, current churn status, and risk scores.
3. **Analyze Service Drivers**: Bar charts and statistical observations across Contract types, Internet Services, Payment Methods, and Tenure brackets.
4. **Segment Risk Distribution**: Categorize dataset records into **Low Risk** (0–30%), **Medium Risk** (31–60%), and **High Risk** (61–100%) groups.
5. **Simulate Scenarios (What-If)**: Modify subscription features in real time to calculate probability deltas.
6. **Classify Business Segments**: Evaluate New Customers, Loyal Customers, High Value Customers, High Monthly Cost, and At-Risk segments.
7. **Predict Customer Churn**: Logistic Regression classification model with model performance evaluation (80/20 train/test split, accuracy, confusion matrix, classification report).
8. **Export Reports**: Download raw CSV dataset, scored predictions, and executive text summary reports.

---

## 🛠️ Tech Stack

* **Python 3.10+**: Core programming language
* **Pandas**: Data manipulation and aggregation
* **NumPy**: Vectorized calculations and numerical ops
* **Matplotlib & Seaborn**: Statistical visualizations and distribution plots
* **Scikit-learn**: Preprocessing pipelines (`OneHotEncoder`, `StandardScaler`) and Logistic Regression model training
* **Streamlit**: Light academic analytics web dashboard

---

## 📊 Dataset Description

The project uses `customer_churn.csv`, containing **750 customer records** with 14 attributes:

| Column Name | Data Type | Description |
| :--- | :--- | :--- |
| `CustomerID` | String | Unique identification code for each customer |
| `Gender` | Categorical | `Male` or `Female` |
| `Age` | Integer | Customer age (18 to 80 years) |
| `Tenure` | Integer | Number of months the customer has stayed with the company |
| `MonthlyCharges` | Float | Monthly bill amount ($18.5 to $118.5) |
| `TotalCharges` | Float | Cumulative bill amount over customer tenure |
| `Contract` | Categorical | Contract term (`Month-to-month`, `One year`, `Two year`) |
| `PaymentMethod` | Categorical | `Electronic check`, `Mailed check`, `Bank transfer (automatic)`, `Credit card (automatic)` |
| `InternetService` | Categorical | Internet provider type (`DSL`, `Fiber optic`, `No`) |
| `TechSupport` | Categorical | Technical support subscription (`Yes`, `No`, `No internet service`) |
| `OnlineSecurity` | Categorical | Cyber security subscription (`Yes`, `No`, `No internet service`) |
| `PaperlessBilling` | Categorical | Paperless billing indicator (`Yes`, `No`) |
| `SeniorCitizen` | Binary | Senior citizen indicator (`0` = No, `1` = Yes) |
| `Churn` | Categorical | Target label (`Yes` = Left service, `No` = Retained) |

---

## 🧭 Dashboard Navigation Structure

The sidebar provides access to 8 sections:

1. **Overview**: Executive top header, KPI row, 60/40 Donut vs. Snapshot split, Churn by Contract & Internet Service charts, and Customer Activity table.
2. **Customers**: Data table with search/filters, CSV export, and 3-column Customer Profile card.
3. **Churn Analysis**: Categorical distribution bar charts + Churn Risk Distribution section (Low/Medium/High risk counts and horizontal bar chart).
4. **Customer Insights**: Key churn factors horizontal comparison chart and empirical non-causal observations.
5. **Prediction**: 3-section input form (`Customer Details`, `Subscription`, `Services & Billing`), single primary action button, result panel, and Model Performance section.
6. **What-If Simulator**: 2-column scenario input vs. output simulator with explicit probability delta calculation (e.g. "Churn probability changed from 42.1% to 28.7%").
7. **Customer Segmentation**: Business segments table (New Customers, Loyal Customers, High Value, High Monthly Cost, At-Risk Customers) with customer counts, avg monthly charges, avg tenure, and churn rates.
8. **Reports**: Executive summary cards and CSV/TXT export downloads.

---

## 🚀 How to Run the Application

```powershell
# 1. Navigate to project directory
cd "e:\DATA SCIENCE PROJECT\customer-churn-analysis"

# 2. Install required Python packages
python -m pip install -r requirements.txt

# 3. Run the Streamlit application
python -m streamlit run app.py
```

Access the dashboard in your browser at `http://localhost:8501`.
