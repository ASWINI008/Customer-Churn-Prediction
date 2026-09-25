"""
model.py - Machine Learning Pipeline for Customer Churn Prediction

This module handles:
1. Dataset loading and feature separation
2. Categorical encoding using OneHotEncoder & Numerical scaling using StandardScaler
3. Model training using Logistic Regression
4. Evaluation metrics (Accuracy, Confusion Matrix, Classification Report)
5. Single customer churn risk prediction, batch dataset risk classification, and business segmentation
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


# Define features
NUMERICAL_FEATURES = ['Age', 'Tenure', 'MonthlyCharges', 'TotalCharges', 'SeniorCitizen']
CATEGORICAL_FEATURES = [
    'Gender', 'Contract', 'PaymentMethod', 'InternetService', 
    'TechSupport', 'OnlineSecurity', 'PaperlessBilling'
]
TARGET_COLUMN = 'Churn'


def get_dataset_path():
    """Returns absolute path to customer_churn.csv."""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base_dir, 'customer_churn.csv')


def load_data():
    """Loads the customer churn dataset."""
    csv_path = get_dataset_path()
    df = pd.read_csv(csv_path)
    return df


def build_pipeline():
    """
    Creates a scikit-learn preprocessing and Logistic Regression pipeline.
    - OneHotEncoder for categorical features
    - StandardScaler for numerical features
    - LogisticRegression classifier
    """
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), NUMERICAL_FEATURES),
            ('cat', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'), CATEGORICAL_FEATURES)
        ]
    )

    model_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', LogisticRegression(random_state=42, max_iter=1000))
    ])

    return model_pipeline


def train_and_evaluate_model():
    """
    Trains the Logistic Regression model on 80% data and evaluates on 20% test data.
    
    Returns:
        pipeline: Trained sklearn Pipeline
        metrics: Dictionary containing accuracy, confusion_matrix, classification_report, test set counts
    """
    df = load_data()

    # Step 1: Feature Matrix (X) and Target Vector (y)
    X = df[NUMERICAL_FEATURES + CATEGORICAL_FEATURES]
    y = df[TARGET_COLUMN].map({'Yes': 1, 'No': 0})

    # Step 2: Train-Test Split (80% Train, 20% Test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    # Step 3: Build & Fit Model Pipeline
    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)

    # Step 4: Predict on Test Set
    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]

    # Step 5: Calculate Evaluation Metrics
    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    report_dict = classification_report(y_test, y_pred, target_names=['Stay (No)', 'Churn (Yes)'], output_dict=True)
    report_text = classification_report(y_test, y_pred, target_names=['Stay (No)', 'Churn (Yes)'])

    metrics = {
        'accuracy': accuracy,
        'confusion_matrix': cm,
        'classification_report_dict': report_dict,
        'classification_report_text': report_text,
        'train_samples': len(X_train),
        'test_samples': len(X_test)
    }

    return pipeline, metrics


def predict_single_customer(pipeline, customer_data_dict):
    """
    Predicts churn status, probability, and risk level for a single customer input dictionary.
    
    Args:
        pipeline: Trained sklearn Pipeline
        customer_data_dict: Dictionary with feature values matching dataset columns
        
    Returns:
        prediction_label: 'Yes' or 'No'
        churn_probability: float (0.0 to 1.0)
        risk_level: 'Low Risk', 'Medium Risk', or 'High Risk'
    """
    input_df = pd.DataFrame([customer_data_dict])
    pred_code = pipeline.predict(input_df)[0]
    prob = pipeline.predict_proba(input_df)[0][1]

    prediction_label = 'Yes' if pred_code == 1 else 'No'
    
    if prob <= 0.30:
        risk_level = 'Low Risk'
    elif prob <= 0.60:
        risk_level = 'Medium Risk'
    else:
        risk_level = 'High Risk'

    return prediction_label, float(prob), risk_level


def batch_predict_churn(df, pipeline):
    """
    Generates predicted probabilities and risk levels for an entire dataframe.
    
    Returns:
        Augmented dataframe with columns:
        - Predicted_Churn
        - Churn_Probability
        - Risk_Level
    """
    X = df[NUMERICAL_FEATURES + CATEGORICAL_FEATURES]
    probs = pipeline.predict_proba(X)[:, 1]
    preds = np.where(probs > 0.5, 'Yes', 'No')

    df_scored = df.copy()
    df_scored['Predicted_Churn'] = preds
    df_scored['Churn_Probability'] = np.round(probs * 100, 1)

    # Classify Risk Levels
    risk_conditions = [
        (probs <= 0.30),
        (probs > 0.30) & (probs <= 0.60),
        (probs > 0.60)
    ]
    risk_choices = ['Low Risk', 'Medium Risk', 'High Risk']
    df_scored['Risk_Level'] = np.select(risk_conditions, risk_choices, default='Low Risk')

    return df_scored


def compute_customer_segmentation(df_scored):
    """
    Computes business-oriented customer segment metrics based on customer behavioral attributes.
    
    Segments:
    1. New Customers (Tenure <= 12m)
    2. Loyal Customers (Tenure > 48m)
    3. High Value Customers (TotalCharges > median)
    4. High Monthly Cost (MonthlyCharges > median)
    5. At-Risk Customers (Risk_Level == 'High Risk')
    """
    tot_charges_median = df_scored['TotalCharges'].median()
    monthly_charges_median = df_scored['MonthlyCharges'].median()

    segments = {
        'New Customers (Tenure <= 12m)': df_scored[df_scored['Tenure'] <= 12],
        'Loyal Customers (Tenure > 48m)': df_scored[df_scored['Tenure'] > 48],
        'High Value Customers (TotalCharges > Median)': df_scored[df_scored['TotalCharges'] > tot_charges_median],
        'High Monthly Cost (MonthlyCharges > Median)': df_scored[df_scored['MonthlyCharges'] > monthly_charges_median],
        'At-Risk Customers (Predicted High Risk)': df_scored[df_scored['Risk_Level'] == 'High Risk']
    }

    seg_summary = []
    for seg_name, seg_df in segments.items():
        count = len(seg_df)
        avg_monthly = seg_df['MonthlyCharges'].mean() if count > 0 else 0.0
        avg_tenure = seg_df['Tenure'].mean() if count > 0 else 0.0
        churn_rate = (seg_df['Churn'] == 'Yes').mean() * 100 if count > 0 else 0.0

        seg_summary.append({
            'Segment Name': seg_name,
            'Customer Count': count,
            'Avg Monthly Charges ($)': round(avg_monthly, 2),
            'Avg Tenure (Months)': round(avg_tenure, 1),
            'Observed Churn Rate (%)': round(churn_rate, 1)
        })

    return pd.DataFrame(seg_summary)


if __name__ == '__main__':
    pipeline, metrics = train_and_evaluate_model()
    print("Model Training Completed Successfully!")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    df = load_data()
    scored = batch_predict_churn(df, pipeline)
    print("\nRisk Level Distribution across Dataset:")
    print(scored['Risk_Level'].value_counts())
    print("\nCustomer Segmentation Summary:")
    print(compute_customer_segmentation(scored))
