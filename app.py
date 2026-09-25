"""
app.py - Human-Designed Data Science Analytics Dashboard for Customer Churn Analysis & Prediction

A clean, restrained, light academic analytics web application built for telecommunications
customer churn analysis, risk segmentation, individual customer profiles, scenario simulation,
customer segmentation, and machine learning classification using Logistic Regression.
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import io

from model import (
    load_data, 
    train_and_evaluate_model, 
    predict_single_customer, 
    batch_predict_churn,
    compute_customer_segmentation,
    NUMERICAL_FEATURES, 
    CATEGORICAL_FEATURES
)

# ---------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="Customer Churn Analytics Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------------
# Human-Designed Light Analytics CSS System
# ---------------------------------------------------------
st.markdown("""
<style>
    /* =========================================================
       FIXED LIGHT THEME COLOR SYSTEM
       Page background: #F5F7FA
       Card background: #FFFFFF
       Primary text:    #172033
       Secondary text:  #4B5563
       Muted text:      #6B7280
       Border:          #D9DEE7
       Primary blue:    #2563EB
       ========================================================= */
    .stApp {
        background-color: #F5F7FA !important;
        color: #172033 !important;
        font-family: 'Segoe UI', -apple-system, BlinkMacSystemFont, Roboto, sans-serif;
    }

    /* Main Content Area */
    .main .block-container {
        padding-top: 1.2rem;
        padding-bottom: 2.5rem;
        max-width: 1250px;
    }

    /* Page Headings, Titles & Section Titles */
    h1, h2, h3, h4, h5, h6,
    .top-header-title, .form-section-header {
        color: #172033 !important;
    }
    
    p, span, div, label {
        color: #172033;
    }

    .top-header-bar {
        background-color: #FFFFFF;
        border: 1px solid #D9DEE7;
        border-radius: 6px;
        padding: 14px 20px;
        margin-bottom: 20px;
        box-shadow: 0 1px 2px rgba(0, 0, 0, 0.03);
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    .top-header-subtitle {
        font-size: 0.875rem;
        color: #4B5563 !important;
        margin-top: 2px;
    }
    .top-header-badge {
        font-size: 0.8rem;
        font-weight: 600;
        color: #2563EB !important;
        background-color: #EFF6FF;
        border: 1px solid #D9DEE7;
        padding: 4px 10px;
        border-radius: 4px;
        white-space: nowrap;
    }

    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background-color: #FFFFFF !important;
        border-right: 1px solid #D9DEE7 !important;
        width: 240px !important;
    }
    .sidebar-brand {
        padding: 12px 0 14px 0;
        border-bottom: 1px solid #D9DEE7;
        margin-bottom: 15px;
    }
    .sidebar-brand-title {
        font-size: 1.15rem;
        font-weight: 800;
        color: #2563EB !important;
        letter-spacing: 0.5px;
        margin: 0;
    }
    .sidebar-brand-sub {
        font-size: 0.75rem;
        font-weight: 600;
        color: #4B5563 !important;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin: 0;
    }
    .sidebar-footer {
        margin-top: 25px;
        padding-top: 14px;
        border-top: 1px solid #D9DEE7;
        font-size: 0.775rem;
        color: #4B5563 !important;
        line-height: 1.4;
    }
    .sidebar-footer strong {
        color: #172033 !important;
    }

    /* Sidebar Navigation Items */
    div[data-testid="stRadio"] > label {
        display: none !important;
    }
    div[data-testid="stRadio"] div[role="radiogroup"] > label {
        background-color: transparent;
        border: none;
        border-radius: 4px;
        padding: 7px 12px;
        margin-bottom: 4px;
        font-size: 0.875rem;
        font-weight: 600;
        color: #374151 !important;
        transition: all 0.15s ease;
    }
    div[data-testid="stRadio"] div[role="radiogroup"] > label:hover {
        background-color: #F3F4F6 !important;
        color: #172033 !important;
    }
    div[data-testid="stRadio"] div[role="radiogroup"] > label[aria-checked="true"] {
        background-color: #EFF6FF !important;
        color: #2563EB !important;
        border-left: 3.5px solid #2563EB !important;
        font-weight: 700 !important;
    }
    div[data-testid="stRadio"] div[role="radiogroup"] label div[data-testid="stMarkdownContainer"] p {
        color: inherit !important;
    }

    /* Compact KPI Cards */
    .kpi-card {
        background-color: #FFFFFF !important;
        border: 1px solid #D9DEE7 !important;
        border-radius: 6px;
        padding: 14px 18px;
        box-shadow: 0 1px 2px rgba(0, 0, 0, 0.03);
    }
    .kpi-card .kpi-label {
        font-size: 0.775rem;
        font-weight: 700;
        color: #4B5563 !important;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 4px;
    }
    .kpi-card .kpi-value {
        font-size: 1.6rem;
        font-weight: 800;
        color: #172033 !important;
    }

    /* Content Cards / Containers */
    .content-card {
        background-color: #FFFFFF !important;
        border: 1px solid #D9DEE7 !important;
        border-radius: 6px;
        padding: 18px;
        box-shadow: 0 1px 2px rgba(0, 0, 0, 0.02);
        margin-bottom: 16px;
    }
    
    /* Observation Boxes */
    .observation-box {
        background-color: #F5F7FA;
        border-left: 3.5px solid #2563EB;
        border-top: 1px solid #D9DEE7;
        border-right: 1px solid #D9DEE7;
        border-bottom: 1px solid #D9DEE7;
        padding: 10px 14px;
        border-radius: 4px;
        font-size: 0.875rem;
        color: #172033 !important;
        margin-top: 8px;
        margin-bottom: 16px;
    }
    .observation-box strong {
        color: #172033 !important;
    }

    /* Alert / Risk Banners */
    .risk-banner-low {
        background-color: #ECFDF5 !important;
        border: 1px solid #A7F3D0 !important;
        border-left: 4px solid #16534A !important;
        color: #166534 !important;
        padding: 14px 16px;
        border-radius: 5px;
        margin-top: 12px;
    }
    .risk-banner-low * {
        color: #166534 !important;
    }

    .risk-banner-medium {
        background-color: #FFFBEB !important;
        border: 1px solid #FDE68A !important;
        border-left: 4px solid #B45309 !important;
        color: #92400E !important;
        padding: 14px 16px;
        border-radius: 5px;
        margin-top: 12px;
    }
    .risk-banner-medium * {
        color: #92400E !important;
    }

    .risk-banner-high {
        background-color: #FEF2F2 !important;
        border: 1px solid #FECACA !important;
        border-left: 4px solid #DC2626 !important;
        color: #991B1B !important;
        padding: 14px 16px;
        border-radius: 5px;
        margin-top: 12px;
    }
    .risk-banner-high * {
        color: #991B1B !important;
    }

    /* Form Section Headers */
    .form-section-header {
        font-size: 0.825rem;
        font-weight: 700;
        color: #172033 !important;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        padding-bottom: 4px;
        border-bottom: 1px solid #D9DEE7;
        margin-bottom: 10px;
    }
    
    /* Buttons */
    .stButton>button {
        background-color: #2563EB !important;
        color: #FFFFFF !important;
        font-weight: 600 !important;
        border: none !important;
        border-radius: 5px !important;
        padding: 8px 18px !important;
        transition: background-color 0.15s ease !important;
    }
    .stButton>button:hover {
        background-color: #1D4ED8 !important;
        color: #FFFFFF !important;
    }
    .stButton>button * {
        color: #FFFFFF !important;
    }

    /* Download Buttons (Secondary) */
    .stDownloadButton>button {
        background-color: #FFFFFF !important;
        color: #172033 !important;
        border: 1px solid #D9DEE7 !important;
        font-weight: 600 !important;
        border-radius: 5px !important;
        padding: 6px 14px !important;
    }
    .stDownloadButton>button:hover {
        background-color: #EFF6FF !important;
        color: #2563EB !important;
        border-color: #2563EB !important;
    }
    .stDownloadButton>button * {
        color: inherit !important;
    }

    /* Streamlit Inputs & Labels */
    label, 
    .stTextInput label, 
    .stSelectbox label, 
    .stNumberInput label, 
    .stSlider label, 
    .stMultiSelect label,
    div[data-testid="stWidgetLabel"] p {
        color: #374151 !important;
        font-weight: 600 !important;
        font-size: 0.85rem !important;
    }
    
    div[data-baseweb="input"] input, 
    div[data-baseweb="select"] div,
    .stTextInput input, 
    .stNumberInput input,
    div[data-baseweb="base-input"] input {
        color: #172033 !important;
        background-color: #FFFFFF !important;
        border-color: #D9DEE7 !important;
    }
    
    /* Input Placeholders */
    ::placeholder,
    input::placeholder {
        color: #6B7280 !important;
        opacity: 1 !important;
    }

    /* Dropdown Menus & Popovers */
    div[data-baseweb="popover"],
    div[data-baseweb="popover"] *,
    ul[role="listbox"],
    ul[role="listbox"] *,
    li[role="option"],
    div[role="option"] {
        color: #172033 !important;
        background-color: #FFFFFF !important;
    }
    div[role="option"]:hover, 
    li[role="option"]:hover {
        background-color: #F3F4F6 !important;
        color: #172033 !important;
    }
    div[role="option"][aria-selected="true"],
    li[role="option"][aria-selected="true"] {
        background-color: #EFF6FF !important;
        color: #2563EB !important;
        font-weight: 600 !important;
    }

    /* Tables & Dataframes */
    .stDataFrame, 
    div[data-testid="stTable"] table {
        background-color: #FFFFFF !important;
        border: 1px solid #D9DEE7 !important;
    }
    div[data-testid="stTable"] th,
    div[data-testid="stTable"] th p {
        background-color: #F3F4F6 !important;
        color: #172033 !important;
        font-weight: 700 !important;
        border-bottom: 1px solid #D9DEE7 !important;
    }
    div[data-testid="stTable"] td,
    div[data-testid="stTable"] td p,
    div[data-testid="stTable"] p {
        color: #374151 !important;
    }
    div[data-testid="stTable"] tr:hover {
        background-color: #EFF6FF !important;
    }

    /* Metrics & KPI Labels */
    div[data-testid="stMetricLabel"] p {
        color: #4B5563 !important;
        font-weight: 600 !important;
    }
    div[data-testid="stMetricValue"] div {
        color: #172033 !important;
    }

    /* Tabs */
    button[data-baseweb="tab"] {
        color: #4B5563 !important;
        background-color: transparent !important;
        font-weight: 600 !important;
    }
    button[data-baseweb="tab"][aria-selected="true"] {
        color: #2563EB !important;
        background-color: #EFF6FF !important;
        font-weight: 700 !important;
        border-bottom: 2px solid #2563EB !important;
    }
    button[data-baseweb="tab"] div,
    button[data-baseweb="tab"] p {
        color: inherit !important;
    }
    
    /* Expanders */
    .stExpander summary,
    .stExpander summary *,
    details summary span {
        color: #172033 !important;
        font-weight: 600 !important;
    }
    .stExpander div[data-testid="stMarkdownContainer"] p {
        color: #374151 !important;
    }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------
# Data & Model Caching
# ---------------------------------------------------------
@st.cache_data
def get_dataset():
    return load_data()

@st.cache_resource
def get_model():
    return train_and_evaluate_model()

df = get_dataset()
pipeline, model_metrics = get_model()
df_scored = batch_predict_churn(df, pipeline)


# ---------------------------------------------------------
# Sidebar Navigation
# ---------------------------------------------------------
with st.sidebar:
    st.markdown("""
<div class="sidebar-brand">
    <div class="sidebar-brand-title">CUSTOMER CHURN</div>
    <div class="sidebar-brand-sub">Analytics Dashboard</div>
</div>
""", unsafe_allow_html=True)

    nav_option = st.radio(
        "Navigation Menu",
        [
            "Overview", 
            "Customers", 
            "Churn Analysis", 
            "Customer Insights", 
            "Prediction", 
            "Upload & Predict",
            "What-If Simulator", 
            "Customer Segmentation", 
            "Reports"
        ],
        index=0
    )

    st.markdown(f"""
<div class="sidebar-footer">
    <strong>DATA SCIENCE PROJECT</strong><br>
    <strong>Algorithm:</strong> Logistic Regression<br>
    <strong>Dataset:</strong> Customer Churn<br>
    <strong>Records:</strong> {len(df):,} Rows
</div>
""", unsafe_allow_html=True)


# ---------------------------------------------------------
# Top Header Component Helper
# ---------------------------------------------------------
def render_header(title, subtitle):
    st.markdown(f"""
<div class="top-header-bar">
    <div>
        <div class="top-header-title">{title}</div>
        <div class="top-header-subtitle">{subtitle}</div>
    </div>
    <div class="top-header-badge">
        Dataset: Customer Churn &nbsp;|&nbsp; Session: Active
    </div>
</div>
""", unsafe_allow_html=True)


# Color Palette for Charts
PRIMARY_BLUE = '#2563EB'
MUTED_RED = '#DC2626'
MUTED_AMBER = '#CA8A04'
MUTED_GREEN = '#16A34A'
CHURN_PALETTE = {'No': PRIMARY_BLUE, 'Yes': MUTED_RED}


# =========================================================
# 1. OVERVIEW DASHBOARD
# =========================================================
if nav_option == "Overview":
    render_header(
        "Customer Churn Analysis",
        "Understand customer behavior, identify churn patterns, and estimate customer risk."
    )

    # Calculate dynamic metrics
    total_cust = len(df)
    churned_cust = int((df['Churn'] == 'Yes').sum())
    active_cust = int((df['Churn'] == 'No').sum())
    churn_rate_val = (churned_cust / total_cust) * 100

    # Top KPI Row
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.markdown(f"""
<div class="kpi-card">
    <div class="kpi-label">Total Customers</div>
    <div class="kpi-value">{total_cust:,}</div>
</div>
""", unsafe_allow_html=True)
    with k2:
        st.markdown(f"""
<div class="kpi-card">
    <div class="kpi-label">Churned Customers</div>
    <div class="kpi-value" style="color: #DC2626;">{churned_cust:,}</div>
</div>
""", unsafe_allow_html=True)
    with k3:
        st.markdown(f"""
<div class="kpi-card">
    <div class="kpi-label">Active Customers</div>
    <div class="kpi-value" style="color: #16A34A;">{active_cust:,}</div>
</div>
""", unsafe_allow_html=True)
    with k4:
        st.markdown(f"""
<div class="kpi-card">
    <div class="kpi-label">Churn Rate</div>
    <div class="kpi-value" style="color: #2563EB;">{churn_rate_val:.1f}%</div>
</div>
""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # 60% Left Donut Chart & 40% Right Customer Snapshot
    left_60, right_40 = st.columns([1.2, 0.8])

    with left_60:
        st.subheader("Churn Overview")
        fig, ax = plt.subplots(figsize=(5.5, 3.2))
        churn_counts = df['Churn'].value_counts()
        labels = [f"Stayed ({churn_counts.get('No', 0)})", f"Churned ({churn_counts.get('Yes', 0)})"]
        colors = [PRIMARY_BLUE, MUTED_RED]
        
        wedges, texts, autotexts = ax.pie(
            churn_counts, 
            labels=labels, 
            autopct='%1.1f%%',
            startangle=90,
            colors=colors,
            pctdistance=0.75,
            textprops=dict(color="#0F172A", fontsize=9, weight='bold')
        )
        centre_circle = plt.Circle((0,0), 0.52, fc='white')
        fig.gca().add_artist(centre_circle)
        ax.axis('equal')
        plt.tight_layout()
        st.pyplot(fig)

    with right_40:
        st.subheader("Customer Snapshot")
        avg_tenure = df['Tenure'].mean()
        avg_monthly = df['MonthlyCharges'].mean()
        avg_total = df['TotalCharges'].mean()

        st.markdown(f"""
<div class="content-card" style="margin-top: 5px;">
    <table style="width:100%; font-size: 0.9rem; border-collapse: collapse;">
        <tr style="border-bottom: 1px solid #E2E8F0; height: 38px;">
            <td style="color: #64748B; font-weight: 600;">Average Tenure</td>
            <td style="text-align: right; font-weight: 700; color: #0F172A;">{avg_tenure:.1f} months</td>
        </tr>
        <tr style="border-bottom: 1px solid #E2E8F0; height: 38px;">
            <td style="color: #64748B; font-weight: 600;">Average Monthly Charges</td>
            <td style="text-align: right; font-weight: 700; color: #0F172A;">${avg_monthly:.2f}</td>
        </tr>
        <tr style="height: 38px;">
            <td style="color: #64748B; font-weight: 600;">Average Total Charges</td>
            <td style="text-align: right; font-weight: 700; color: #0F172A;">${avg_total:.2f}</td>
        </tr>
    </table>
</div>
""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # 50/50 Charts: Churn by Contract & Churn by Internet Service
    c_left, c_right = st.columns(2)

    with c_left:
        st.subheader("Churn by Contract")
        ct_c = pd.crosstab(df['Contract'], df['Churn'], normalize='index') * 100
        fig_c, ax_c = plt.subplots(figsize=(5.5, 3))
        ct_c['Yes'].plot(kind='bar', color=MUTED_RED, ax=ax_c, width=0.45)
        ax_c.set_ylabel("Churn Rate (%)", fontsize=8.5)
        ax_c.set_xlabel("Contract Type", fontsize=8.5)
        ax_c.set_ylim(0, max(ct_c['Yes']) + 15)
        plt.xticks(rotation=0)
        for p in ax_c.patches:
            ax_c.annotate(f"{p.get_height():.1f}%", (p.get_x() + p.get_width() / 2., p.get_height()),
                         ha='center', va='bottom', fontsize=8.5, xytext=(0, 2), textcoords='offset points')
        plt.tight_layout()
        st.pyplot(fig_c)

    with c_right:
        st.subheader("Churn by Internet Service")
        ct_is = pd.crosstab(df['InternetService'], df['Churn'], normalize='index') * 100
        fig_is, ax_is = plt.subplots(figsize=(5.5, 3))
        ct_is['Yes'].plot(kind='bar', color=PRIMARY_BLUE, ax=ax_is, width=0.45)
        ax_is.set_ylabel("Churn Rate (%)", fontsize=8.5)
        ax_is.set_xlabel("Internet Service Provider", fontsize=8.5)
        ax_is.set_ylim(0, max(ct_is['Yes']) + 15)
        plt.xticks(rotation=0)
        for p in ax_is.patches:
            ax_is.annotate(f"{p.get_height():.1f}%", (p.get_x() + p.get_width() / 2., p.get_height()),
                         ha='center', va='bottom', fontsize=8.5, xytext=(0, 2), textcoords='offset points')
        plt.tight_layout()
        st.pyplot(fig_is)

    st.markdown("<br>", unsafe_allow_html=True)
    st.subheader("Recent Customer Activity")
    disp_cols = ['CustomerID', 'Gender', 'Age', 'Tenure', 'Contract', 'MonthlyCharges', 'TotalCharges', 'Churn']
    st.dataframe(df[disp_cols].head(8), use_container_width=True, height=270)


# =========================================================
# 2. CUSTOMERS PAGE (DIRECTORY + PROFILE)
# =========================================================
elif nav_option == "Customers":
    render_header(
        "Customer Directory & Profile Search",
        "Inspect customer records, apply multi-attribute filters, and view individual profile details."
    )

    cust_tab1, cust_tab2 = st.tabs(["Customer Data Table", "Individual Customer Profile"])

    with cust_tab1:
        st.markdown("<br>", unsafe_allow_html=True)
        # Top filter row
        col_f1, col_f2, col_f3, col_f4, col_f5 = st.columns(5)
        with col_f1:
            search_id = st.text_input("Customer ID Search", placeholder="e.g. 7590...")
        with col_f2:
            contract_f = st.selectbox("Contract", ["All"] + list(df['Contract'].unique()))
        with col_f3:
            internet_f = st.selectbox("Internet Service", ["All"] + list(df['InternetService'].unique()))
        with col_f4:
            payment_f = st.selectbox("Payment Method", ["All"] + list(df['PaymentMethod'].unique()))
        with col_f5:
            churn_f = st.selectbox("Churn Status", ["All", "Yes", "No"])

        # Filter dataset
        filt = df_scored.copy()
        if search_id:
            filt = filt[filt['CustomerID'].str.contains(search_id, case=False, na=False)]
        if contract_f != "All":
            filt = filt[filt['Contract'] == contract_f]
        if internet_f != "All":
            filt = filt[filt['InternetService'] == internet_f]
        if payment_f != "All":
            filt = filt[filt['PaymentMethod'] == payment_f]
        if churn_f != "All":
            filt = filt[filt['Churn'] == churn_f]

        top_col1, top_col2 = st.columns([0.8, 0.2])
        with top_col1:
            st.markdown(f"**Filtered Customers:** `{len(filt):,}`")
        with top_col2:
            csv_b = filt.to_csv(index=False).encode('utf-8')
            st.download_button("Download CSV", csv_b, "customers_filtered.csv", "text/csv")

        st.dataframe(
            filt[['CustomerID', 'Gender', 'Age', 'Tenure', 'Contract', 'InternetService', 'PaymentMethod', 'MonthlyCharges', 'TotalCharges', 'Churn', 'Churn_Probability', 'Risk_Level']],
            use_container_width=True,
            height=380
        )

    with cust_tab2:
        st.markdown("<br>", unsafe_allow_html=True)
        st.subheader("Customer Profile Lookup")

        sel_id = st.selectbox("Select Customer ID to View Profile", options=df_scored['CustomerID'].values)
        row = df_scored[df_scored['CustomerID'] == sel_id].iloc[0]

        # 3-Column Profile Layout: LEFT (Info), CENTER (Subscription), RIGHT (Billing & Services)
        p_c1, p_c2, p_c3 = st.columns(3)

        with p_c1:
            st.markdown(f"""
<div class="content-card">
    <div class="form-section-header">CUSTOMER INFORMATION</div>
    <table style="width: 100%; font-size: 0.875rem; border-collapse: collapse;">
        <tr style="border-bottom: 1px solid #F1F5F9; height: 32px;">
            <td style="color: #64748B;">Customer ID</td>
            <td style="text-align: right; font-weight: 700;">{row['CustomerID']}</td>
        </tr>
        <tr style="border-bottom: 1px solid #F1F5F9; height: 32px;">
            <td style="color: #64748B;">Age / Gender</td>
            <td style="text-align: right; font-weight: 600;">{row['Age']} yrs / {row['Gender']}</td>
        </tr>
        <tr style="height: 32px;">
            <td style="color: #64748B;">Senior Citizen</td>
            <td style="text-align: right; font-weight: 600;">{'Yes (1)' if row['SeniorCitizen'] == 1 else 'No (0)'}</td>
        </tr>
    </table>
</div>
""", unsafe_allow_html=True)

        with p_c2:
            st.markdown(f"""
<div class="content-card">
    <div class="form-section-header">SUBSCRIPTION</div>
    <table style="width: 100%; font-size: 0.875rem; border-collapse: collapse;">
        <tr style="border-bottom: 1px solid #F1F5F9; height: 32px;">
            <td style="color: #64748B;">Tenure</td>
            <td style="text-align: right; font-weight: 700;">{row['Tenure']} months</td>
        </tr>
        <tr style="border-bottom: 1px solid #F1F5F9; height: 32px;">
            <td style="color: #64748B;">Contract</td>
            <td style="text-align: right; font-weight: 600;">{row['Contract']}</td>
        </tr>
        <tr style="height: 32px;">
            <td style="color: #64748B;">Internet Service</td>
            <td style="text-align: right; font-weight: 600;">{row['InternetService']}</td>
        </tr>
    </table>
</div>
""", unsafe_allow_html=True)

        with p_c3:
            st.markdown(f"""
<div class="content-card">
    <div class="form-section-header">BILLING & SERVICES</div>
    <table style="width: 100%; font-size: 0.875rem; border-collapse: collapse;">
        <tr style="border-bottom: 1px solid #F1F5F9; height: 32px;">
            <td style="color: #64748B;">Monthly Charges</td>
            <td style="text-align: right; font-weight: 700;">${row['MonthlyCharges']:.2f}</td>
        </tr>
        <tr style="border-bottom: 1px solid #F1F5F9; height: 32px;">
            <td style="color: #64748B;">Total Charges</td>
            <td style="text-align: right; font-weight: 700;">${row['TotalCharges']:.2f}</td>
        </tr>
        <tr style="border-bottom: 1px solid #F1F5F9; height: 32px;">
            <td style="color: #64748B;">Payment Method</td>
            <td style="text-align: right; font-weight: 600;">{row['PaymentMethod']}</td>
        </tr>
        <tr style="border-bottom: 1px solid #F1F5F9; height: 32px;">
            <td style="color: #64748B;">Tech Support</td>
            <td style="text-align: right; font-weight: 600;">{row['TechSupport']}</td>
        </tr>
        <tr style="height: 32px;">
            <td style="color: #64748B;">Online Security</td>
            <td style="text-align: right; font-weight: 600;">{row['OnlineSecurity']}</td>
        </tr>
    </table>
</div>
""", unsafe_allow_html=True)

        # Profile Status & Risk Result Card
        r_lvl = row['Risk_Level']
        b_class = "risk-banner-low" if r_lvl == "Low Risk" else ("risk-banner-medium" if r_lvl == "Medium Risk" else "risk-banner-high")
        st_color = MUTED_RED if row['Churn'] == "Yes" else MUTED_GREEN

        st.markdown(f"""
<div class="{b_class}">
    <div style="display: flex; justify-content: space-between; align-items: center;">
        <div>
            <div style="font-size: 0.8rem; font-weight: 700; color: #64748B; text-transform: uppercase;">Current Status</div>
            <div style="font-size: 1.35rem; font-weight: 800; color: {st_color};">{'Churned (Left Service)' if row['Churn'] == 'Yes' else 'Active (Retained)'}</div>
        </div>
        <div>
            <div style="font-size: 0.8rem; font-weight: 700; color: #64748B; text-transform: uppercase;">Churn Probability</div>
            <div style="font-size: 1.4rem; font-weight: 800; color: #0F172A;">{row['Churn_Probability']}%</div>
        </div>
        <div>
            <div style="font-size: 0.8rem; font-weight: 700; color: #64748B; text-transform: uppercase;">Assigned Risk Tier</div>
            <div style="font-size: 1.2rem; font-weight: 700; color: #0F172A;">{r_lvl}</div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.write("**Risk Meter:**")
        st.progress(float(row['Churn_Probability'] / 100.0))


# =========================================================
# 3. CHURN ANALYSIS PAGE
# =========================================================
elif nav_option == "Churn Analysis":
    render_header(
        "Churn Analysis & Categorical Breakdown",
        "Examine observed customer churn patterns across service categories and risk score distribution."
    )

    def get_crosstab_rate(col):
        ct = pd.crosstab(df[col], df['Churn'])
        ct['Total'] = ct.sum(axis=1)
        ct['Rate'] = (ct['Yes'] / ct['Total']) * 100
        return ct

    # Contract & Internet Service Grid
    ca1, ca2 = st.columns(2)
    with ca1:
        st.subheader("A. Churn by Contract")
        c_df = get_crosstab_rate('Contract')
        fig, ax = plt.subplots(figsize=(5.5, 2.8))
        sns.barplot(x=c_df.index, y=c_df['Rate'], palette=[PRIMARY_BLUE, MUTED_AMBER, MUTED_GREEN], ax=ax)
        ax.set_ylabel("Churn Rate (%)", fontsize=8.5)
        ax.set_xlabel("Contract Type", fontsize=8.5)
        ax.set_ylim(0, max(c_df['Rate']) + 15)
        for p in ax.patches:
            ax.annotate(f"{p.get_height():.1f}%", (p.get_x() + p.get_width() / 2., p.get_height()),
                         ha='center', va='bottom', fontsize=8.5, xytext=(0, 2), textcoords='offset points')
        plt.tight_layout()
        st.pyplot(fig)
        st.markdown(f"""
<div class="observation-box">
    <strong>Observed Data Insight:</strong> Month-to-month subscribers exhibit a <strong>{c_df.loc['Month-to-month', 'Rate']:.1f}%</strong> churn rate, compared to <strong>{c_df.loc['Two year', 'Rate']:.1f}%</strong> for Two-year contract holders.
</div>
""", unsafe_allow_html=True)

    with ca2:
        st.subheader("B. Churn by Internet Service")
        is_df = get_crosstab_rate('InternetService')
        fig, ax = plt.subplots(figsize=(5.5, 2.8))
        sns.barplot(x=is_df.index, y=is_df['Rate'], palette=[PRIMARY_BLUE, MUTED_RED, MUTED_GREEN], ax=ax)
        ax.set_ylabel("Churn Rate (%)", fontsize=8.5)
        ax.set_xlabel("Internet Service Provider", fontsize=8.5)
        ax.set_ylim(0, max(is_df['Rate']) + 15)
        for p in ax.patches:
            ax.annotate(f"{p.get_height():.1f}%", (p.get_x() + p.get_width() / 2., p.get_height()),
                         ha='center', va='bottom', fontsize=8.5, xytext=(0, 2), textcoords='offset points')
        plt.tight_layout()
        st.pyplot(fig)
        st.markdown(f"""
<div class="observation-box">
    <strong>Observed Data Insight:</strong> Fiber optic internet subscribers show a higher churn rate of <strong>{is_df.loc['Fiber optic', 'Rate']:.1f}%</strong> relative to DSL or No Internet service.
</div>
""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Payment Method & Tenure Bracket Grid
    ca3, ca4 = st.columns(2)
    with ca3:
        st.subheader("C. Churn by Payment Method")
        pm_df = get_crosstab_rate('PaymentMethod')
        fig, ax = plt.subplots(figsize=(5.5, 2.8))
        sns.barplot(x=pm_df.index, y=pm_df['Rate'], palette='Blues_r', ax=ax)
        ax.set_ylabel("Churn Rate (%)", fontsize=8.5)
        ax.set_xlabel("Payment Method", fontsize=8.5)
        plt.xticks(rotation=15, ha='right')
        ax.set_ylim(0, max(pm_df['Rate']) + 15)
        for p in ax.patches:
            ax.annotate(f"{p.get_height():.1f}%", (p.get_x() + p.get_width() / 2., p.get_height()),
                         ha='center', va='bottom', fontsize=8.5, xytext=(0, 2), textcoords='offset points')
        plt.tight_layout()
        st.pyplot(fig)
        st.markdown(f"""
<div class="observation-box">
    <strong>Observed Data Insight:</strong> Electronic check users present the highest churn rate at <strong>{pm_df.loc['Electronic check', 'Rate']:.1f}%</strong> among payment methods.
</div>
""", unsafe_allow_html=True)

    with ca4:
        st.subheader("D. Churn by Tenure Bracket")
        bins = [0, 12, 24, 48, 100]
        lbls = ['0–12m', '13–24m', '25–48m', '49+m']
        df_t = df.copy()
        df_t['TB'] = pd.cut(df_t['Tenure'], bins=bins, labels=lbls, right=True)
        tg_df = pd.crosstab(df_t['TB'], df_t['Churn'])
        tg_df['Rate'] = (tg_df['Yes'] / tg_df.sum(axis=1)) * 100

        fig, ax = plt.subplots(figsize=(5.5, 2.8))
        sns.barplot(x=tg_df.index, y=tg_df['Rate'], palette='Reds_r', ax=ax)
        ax.set_ylabel("Churn Rate (%)", fontsize=8.5)
        ax.set_xlabel("Tenure Bracket", fontsize=8.5)
        ax.set_ylim(0, max(tg_df['Rate']) + 15)
        for p in ax.patches:
            ax.annotate(f"{p.get_height():.1f}%", (p.get_x() + p.get_width() / 2., p.get_height()),
                         ha='center', va='bottom', fontsize=8.5, xytext=(0, 2), textcoords='offset points')
        plt.tight_layout()
        st.pyplot(fig)
        st.markdown(f"""
<div class="observation-box">
    <strong>Observed Data Insight:</strong> Customers with tenure of 0–12 months present an observed churn rate of <strong>{tg_df.loc['0–12m', 'Rate']:.1f}%</strong>.
</div>
""", unsafe_allow_html=True)

    st.markdown("<br><hr><br>", unsafe_allow_html=True)

    # FEATURE 1 — CHURN RISK DISTRIBUTION
    st.subheader("Churn Risk Distribution")
    st.markdown("Dataset customer risk segmentation derived from Logistic Regression predicted probability scores.")

    rc = df_scored['Risk_Level'].value_counts()
    low_c = int(rc.get('Low Risk', 0))
    med_c = int(rc.get('Medium Risk', 0))
    high_c = int(rc.get('High Risk', 0))

    r1, r2, r3 = st.columns(3)
    with r1:
        st.markdown(f"""
<div class="kpi-card" style="border-top: 4px solid #16A34A;">
    <div class="kpi-label">Low Risk (0–30%)</div>
    <div class="kpi-value" style="color: #16A34A;">{low_c:,} customers</div>
</div>
""", unsafe_allow_html=True)
    with r2:
        st.markdown(f"""
<div class="kpi-card" style="border-top: 4px solid #CA8A04;">
    <div class="kpi-label">Medium Risk (31–60%)</div>
    <div class="kpi-value" style="color: #CA8A04;">{med_c:,} customers</div>
</div>
""", unsafe_allow_html=True)
    with r3:
        st.markdown(f"""
<div class="kpi-card" style="border-top: 4px solid #DC2626;">
    <div class="kpi-label">High Risk (61–100%)</div>
    <div class="kpi-value" style="color: #DC2626;">{high_c:,} customers</div>
</div>
""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    fig_r, ax_r = plt.subplots(figsize=(7, 2.5))
    risk_order = ['Low Risk', 'Medium Risk', 'High Risk']
    risk_vals = [low_c, med_c, high_c]
    sns.barplot(x=risk_vals, y=risk_order, palette=[MUTED_GREEN, MUTED_AMBER, MUTED_RED], ax=ax_r)
    ax_r.set_xlabel("Customer Count", fontsize=8.5)
    ax_r.set_title("Dataset Risk Level Distribution", fontsize=10, fontweight='bold')
    for bar in ax_r.patches:
        w = bar.get_width()
        ax_r.text(w + 5, bar.get_y() + bar.get_height()/2, f"{int(w)}", ha='left', va='center', fontsize=8.5, fontweight='bold')
    plt.tight_layout()
    st.pyplot(fig_r)


# =========================================================
# 4. CUSTOMER INSIGHTS PAGE
# =========================================================
elif nav_option == "Customer Insights":
    render_header(
        "Customer Insights & Key Churn Factors",
        "Empirical analysis of customer attributes strongly associated with higher observed churn rates."
    )

    st.subheader("Key Churn Factors Analysis")
    st.markdown("Comparison of observed churn rates across key customer attribute groups.")

    factors_list = [
        {'Factor': 'Month-to-Month Contract', 'Churn Rate (%)': df[df['Contract'] == 'Month-to-month']['Churn'].eq('Yes').mean() * 100},
        {'Factor': 'Fiber Optic Internet', 'Churn Rate (%)': df[df['InternetService'] == 'Fiber optic']['Churn'].eq('Yes').mean() * 100},
        {'Factor': 'Electronic Check Payment', 'Churn Rate (%)': df[df['PaymentMethod'] == 'Electronic check']['Churn'].eq('Yes').mean() * 100},
        {'Factor': 'Tenure <= 12 Months', 'Churn Rate (%)': df[df['Tenure'] <= 12]['Churn'].eq('Yes').mean() * 100},
        {'Factor': 'Monthly Charges > $70', 'Churn Rate (%)': df[df['MonthlyCharges'] > 70]['Churn'].eq('Yes').mean() * 100},
        {'Factor': 'No Tech Support', 'Churn Rate (%)': df[df['TechSupport'] == 'No']['Churn'].eq('Yes').mean() * 100},
        {'Factor': 'Senior Citizen Status', 'Churn Rate (%)': df[df['SeniorCitizen'] == 1]['Churn'].eq('Yes').mean() * 100}
    ]

    f_df = pd.DataFrame(factors_list).sort_values(by='Churn Rate (%)', ascending=True)

    fig_f, ax_f = plt.subplots(figsize=(7.5, 3.5))
    bars = ax_f.barh(f_df['Factor'], f_df['Churn Rate (%)'], color=PRIMARY_BLUE, height=0.55)
    ax_f.set_xlabel("Observed Churn Rate (%)", fontsize=8.5)
    ax_f.set_title("Observed Churn Rate by Attribute Subgroups", fontsize=10, fontweight='bold')
    ax_f.set_xlim(0, 100)
    for b in bars:
        w = b.get_width()
        ax_f.text(w + 1.5, b.get_y() + b.get_height()/2, f"{w:.1f}%", ha='left', va='center', fontsize=8.5, fontweight='bold')
    plt.tight_layout()
    st.pyplot(fig_f)

    st.markdown("<br>", unsafe_allow_html=True)
    st.subheader("High-Risk Group Summary")

    ig1, ig2 = st.columns(2)
    with ig1:
        st.markdown(f"""
<div class="content-card">
    <div class="form-section-header">PRIMARY HIGH-RISK GROUPS</div>
    <ul style="font-size: 0.875rem; color: #334155; margin-bottom: 0; padding-left: 20px;">
        <li><strong>Month-to-month contract holders</strong> show a higher observed churn rate ({f_df[f_df['Factor']=='Month-to-Month Contract']['Churn Rate (%)'].values[0]:.1f}%).</li>
        <li><strong>Short tenure subscribers (<=12 months)</strong> show a higher observed churn rate ({f_df[f_df['Factor']=='Tenure <= 12 Months']['Churn Rate (%)'].values[0]:.1f}%).</li>
        <li><strong>Electronic check payment users</strong> show a higher observed churn rate ({f_df[f_df['Factor']=='Electronic Check Payment']['Churn Rate (%)'].values[0]:.1f}%).</li>
    </ul>
</div>
""", unsafe_allow_html=True)

    with ig2:
        st.markdown(f"""
<div class="content-card">
    <div class="form-section-header">SECONDARY HIGH-RISK GROUPS</div>
    <ul style="font-size: 0.875rem; color: #334155; margin-bottom: 0; padding-left: 20px;">
        <li><strong>Fiber optic internet subscribers</strong> show a higher observed churn rate ({f_df[f_df['Factor']=='Fiber Optic Internet']['Churn Rate (%)'].values[0]:.1f}%).</li>
        <li><strong>Customers with monthly charges > $70</strong> show a higher observed churn rate ({f_df[f_df['Factor']=='Monthly Charges > $70']['Churn Rate (%)'].values[0]:.1f}%).</li>
        <li><strong>Subscribers without Tech Support</strong> show a higher observed churn rate ({f_df[f_df['Factor']=='No Tech Support']['Churn Rate (%)'].values[0]:.1f}%).</li>
    </ul>
</div>
""", unsafe_allow_html=True)


# =========================================================
# 5. PREDICTION PAGE
# =========================================================
elif nav_option == "Prediction":
    render_header(
        "Customer Churn Prediction",
        "Enter customer demographic and subscription features to estimate churn probability using Logistic Regression."
    )

    st.subheader("Customer Details Input Form")

    with st.form("churn_prediction_form"):
        pf1, pf2, pf3 = st.columns(3)

        with pf1:
            st.markdown("<div class='form-section-header'>SECTION 1: CUSTOMER DETAILS</div>", unsafe_allow_html=True)
            p_gender = st.selectbox("Gender", ["Male", "Female"])
            p_age = st.number_input("Age (Years)", 18, 100, 42)
            p_senior = st.selectbox("Senior Citizen", [0, 1], format_func=lambda x: "Yes (1)" if x == 1 else "No (0)")

        with pf2:
            st.markdown("<div class='form-section-header'>SECTION 2: SUBSCRIPTION</div>", unsafe_allow_html=True)
            p_tenure = st.number_input("Tenure (Months)", 1, 72, 12)
            p_contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
            p_internet = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])

        with pf3:
            st.markdown("<div class='form-section-header'>SECTION 3: SERVICES & BILLING</div>", unsafe_allow_html=True)
            p_monthly = st.number_input("Monthly Charges ($)", 18.0, 150.0, 65.0, step=1.0)
            p_total = st.number_input("Total Charges ($)", 18.0, 10000.0, round(float(p_tenure * p_monthly), 2), step=10.0)
            
            srv_opts = ["Yes", "No"] if p_internet != "No" else ["No internet service"]
            p_tech = st.selectbox("Tech Support", srv_opts)
            p_sec = st.selectbox("Online Security", srv_opts)
            p_pm = st.selectbox("Payment Method", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])
            p_paperless = st.selectbox("Paperless Billing", ["Yes", "No"])

        submit_btn = st.form_submit_button("PREDICT CUSTOMER CHURN", use_container_width=True)

    if submit_btn:
        in_dict = {
            'Gender': p_gender,
            'Age': p_age,
            'Tenure': p_tenure,
            'MonthlyCharges': p_monthly,
            'TotalCharges': p_total,
            'Contract': p_contract,
            'PaymentMethod': p_pm,
            'InternetService': p_internet,
            'TechSupport': p_tech,
            'OnlineSecurity': p_sec,
            'PaperlessBilling': p_paperless,
            'SeniorCitizen': p_senior
        }

        pred_lbl, prob_val, risk_lvl = predict_single_customer(pipeline, in_dict)
        prob_pct = round(prob_val * 100, 1)

        b_class = "risk-banner-low" if risk_lvl == "Low Risk" else ("risk-banner-medium" if risk_lvl == "Medium Risk" else "risk-banner-high")
        st_text = "Likely to Churn" if pred_lbl == "Yes" else "Likely to Stay"
        st_color = MUTED_RED if pred_lbl == "Yes" else MUTED_GREEN

        st.markdown("<br>", unsafe_allow_html=True)
        st.subheader("Prediction Result Panel")

        st.markdown(f"""
<div class="{b_class}">
    <div style="font-size: 0.8rem; font-weight: 700; color: #64748B; text-transform: uppercase;">Prediction Status</div>
    <div style="font-size: 1.45rem; font-weight: 800; color: {st_color}; margin-bottom: 10px;">{st_text}</div>
    <div style="font-size: 0.8rem; font-weight: 700; color: #64748B; text-transform: uppercase;">Churn Probability</div>
    <div style="font-size: 1.55rem; font-weight: 800; color: #0F172A; margin-bottom: 10px;">{prob_pct}%</div>
    <div style="font-size: 0.8rem; font-weight: 700; color: #64748B; text-transform: uppercase;">Assigned Risk Tier</div>
    <div style="font-size: 1.15rem; font-weight: 700; color: #0F172A;">{risk_lvl}</div>
</div>
""", unsafe_allow_html=True)

    st.markdown("<br><hr><br>", unsafe_allow_html=True)

    # Model Performance Section below prediction result
    st.subheader("Model Performance")

    mp1, mp2, mp3, mp4 = st.columns(4)
    with mp1:
        st.metric("Algorithm", "Logistic Regression")
    with mp2:
        st.metric("Train / Test Split", "80% / 20%")
    with mp3:
        st.metric("Test Accuracy", f"{model_metrics['accuracy'] * 100:.2f}%")
    with mp4:
        st.metric("Dataset Size", f"{len(df):,} Records")

    with st.expander("View Confusion Matrix & Classification Report"):
        m_left, m_right = st.columns(2)
        with m_left:
            st.write("**Confusion Matrix:**")
            cm_df = pd.DataFrame(
                model_metrics['confusion_matrix'],
                index=['Actual: Stay (No)', 'Actual: Churn (Yes)'],
                columns=['Predicted: Stay (No)', 'Predicted: Churn (Yes)']
            )
            st.table(cm_df)

        with m_right:
            st.write("**Classification Report:**")
            st.table(pd.DataFrame(model_metrics['classification_report_dict']).transpose().round(2))


# =========================================================
# UPLOAD & PREDICT PAGE
# =========================================================
elif nav_option == "Upload & Predict":
    render_header(
        "Upload Customer Data",
        "Upload a CSV file containing customer information to predict churn for multiple customers."
    )

    st.subheader("Upload CSV File")
    st.markdown("Upload a CSV containing the required customer attributes.")

    uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

    if uploaded_file is None:
        st.info("Please upload a CSV file to continue.")
    else:
        try:
            up_df = pd.read_csv(uploaded_file)
            
            if up_df.empty:
                st.error("The uploaded file does not contain any records.")
            else:
                req_cols = [
                    'Gender', 'Age', 'SeniorCitizen', 'Tenure', 'MonthlyCharges', 
                    'TotalCharges', 'Contract', 'PaymentMethod', 'InternetService', 
                    'TechSupport', 'OnlineSecurity', 'PaperlessBilling'
                ]
                missing_cols = [col for col in req_cols if col not in up_df.columns]

                if missing_cols:
                    st.error("Some required columns are missing.")
                    st.write("**Missing columns:**")
                    for m_col in missing_cols:
                        st.write(f"- `{m_col}`")
                else:
                    # Validate numeric columns gracefully
                    num_cols = ['Age', 'Tenure', 'MonthlyCharges', 'TotalCharges', 'SeniorCitizen']
                    invalid_num_cols = []
                    for nc in num_cols:
                        try:
                            up_df[nc] = pd.to_numeric(up_df[nc])
                        except Exception:
                            invalid_num_cols.append(nc)

                    if invalid_num_cols:
                        st.error(f"Invalid numeric values found in columns: {', '.join(invalid_num_cols)}. Please check your data.")
                    else:
                        st.success("File validated successfully.")
                        st.markdown(f"**Uploaded Records:** `{len(up_df):,}`")

                        with st.expander("Preview uploaded data", expanded=True):
                            st.dataframe(up_df.head(10), use_container_width=True)

                        if st.button("Predict Churn", key="btn_batch_predict_page"):
                            try:
                                scored_up_df = batch_predict_churn(up_df, pipeline)
                                
                                st.markdown("<br>", unsafe_allow_html=True)
                                st.subheader("Batch Prediction Results")

                                u_tot = len(scored_up_df)
                                u_churn = int((scored_up_df['Predicted_Churn'] == 'Yes').sum())
                                u_stay = int((scored_up_df['Predicted_Churn'] == 'No').sum())
                                u_high_risk = int((scored_up_df['Risk_Level'] == 'High Risk').sum())

                                sm1, sm2, sm3, sm4 = st.columns(4)
                                with sm1:
                                    st.markdown(f"""
<div class="kpi-card">
    <div class="kpi-label">Total Uploaded</div>
    <div class="kpi-value">{u_tot:,}</div>
</div>
""", unsafe_allow_html=True)
                                with sm2:
                                    st.markdown(f"""
<div class="kpi-card">
    <div class="kpi-label">Predicted Churn</div>
    <div class="kpi-value" style="color: #DC2626;">{u_churn:,}</div>
</div>
""", unsafe_allow_html=True)
                                with sm3:
                                    st.markdown(f"""
<div class="kpi-card">
    <div class="kpi-label">Predicted Stay</div>
    <div class="kpi-value" style="color: #16534A;">{u_stay:,}</div>
</div>
""", unsafe_allow_html=True)
                                with sm4:
                                    st.markdown(f"""
<div class="kpi-card">
    <div class="kpi-label">High Risk</div>
    <div class="kpi-value" style="color: #DC2626;">{u_high_risk:,}</div>
</div>
""", unsafe_allow_html=True)

                                st.markdown("<br>", unsafe_allow_html=True)
                                st.subheader("Predicted Churn Distribution")
                                fig_u, ax_u = plt.subplots(figsize=(6, 2.8))
                                u_counts = scored_up_df['Predicted_Churn'].value_counts()
                                u_labels = [f"Stay ({u_counts.get('No', 0)})", f"Churn ({u_counts.get('Yes', 0)})"]
                                sns.barplot(x=u_labels, y=[u_counts.get('No', 0), u_counts.get('Yes', 0)], palette=['#2563EB', '#DC2626'], ax=ax_u)
                                ax_u.set_ylabel("Customer Count", fontsize=8.5)
                                ax_u.set_title("Predicted Churn vs Stay Count", fontsize=10, fontweight='bold')
                                for p in ax_u.patches:
                                    ax_u.annotate(f"{int(p.get_height())}", (p.get_x() + p.get_width() / 2., p.get_height()),
                                                 ha='center', va='bottom', fontsize=8.5, xytext=(0, 2), textcoords='offset points')
                                plt.tight_layout()
                                st.pyplot(fig_u)

                                st.markdown("<br>", unsafe_allow_html=True)
                                st.subheader("Results Table")
                                st.dataframe(scored_up_df, use_container_width=True, height=380)

                                pred_download_csv = scored_up_df.to_csv(index=False).encode('utf-8')
                                st.download_button(
                                    label="Download Predictions CSV",
                                    data=pred_download_csv,
                                    file_name="customer_churn_predictions.csv",
                                    mime="text/csv"
                                )
                            except Exception as pred_err:
                                st.error(f"Prediction Error: Unable to process uploaded file. Details: {str(pred_err)}")
        except Exception as file_err:
            st.error(f"Error reading CSV file: {str(file_err)}")


# =========================================================
# 6. WHAT-IF SIMULATOR PAGE
# =========================================================
elif nav_option == "What-If Simulator":
    render_header(
        "What-If Churn Simulator",
        "Modify customer subscription parameters to evaluate real-time probability changes."
    )

    sim_left, sim_right = st.columns([1.1, 0.9])

    with sim_left:
        st.subheader("Customer Scenario Inputs")
        s_contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"], index=0, key="sim_c")
        s_tenure = st.slider("Tenure (Months)", 1, 72, 6, key="sim_t")
        s_monthly = st.slider("Monthly Charges ($)", 18.0, 120.0, 85.0, key="sim_m")
        s_internet = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"], index=1, key="sim_i")
        
        s_srv = ["Yes", "No"] if s_internet != "No" else ["No internet service"]
        s_tech = st.selectbox("Tech Support", s_srv, key="sim_ts")
        s_sec = st.selectbox("Online Security", s_srv, key="sim_os")
        s_pm = st.selectbox("Payment Method", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"], key="sim_pm")
        s_paper = st.selectbox("Paperless Billing", ["Yes", "No"], key="sim_pb")

    with sim_right:
        st.subheader("Simulated Prediction Output")

        # Baseline default dict for comparison (Month-to-month, 6m, 85$)
        baseline_dict = {
            'Gender': 'Female', 'Age': 40, 'Tenure': 6, 'MonthlyCharges': 85.0, 'TotalCharges': 510.0,
            'Contract': 'Month-to-month', 'PaymentMethod': 'Electronic check', 'InternetService': 'Fiber optic',
            'TechSupport': 'No', 'OnlineSecurity': 'No', 'PaperlessBilling': 'Yes', 'SeniorCitizen': 0
        }
        _, base_prob, _ = predict_single_customer(pipeline, baseline_dict)
        base_prob_pct = round(base_prob * 100, 1)

        # Active scenario dict
        scen_dict = {
            'Gender': 'Female', 'Age': 40, 'Tenure': s_tenure, 'MonthlyCharges': s_monthly,
            'TotalCharges': round(float(s_tenure * s_monthly), 2), 'Contract': s_contract,
            'PaymentMethod': s_pm, 'InternetService': s_internet, 'TechSupport': s_tech,
            'OnlineSecurity': s_sec, 'PaperlessBilling': s_paper, 'SeniorCitizen': 0
        }
        s_lbl, s_prob, s_risk = predict_single_customer(pipeline, scen_dict)
        new_prob_pct = round(s_prob * 100, 1)

        diff_pct = new_prob_pct - base_prob_pct
        diff_str = f"Increased by {abs(diff_pct):.1f}%" if diff_pct > 0 else (f"Decreased by {abs(diff_pct):.1f}%" if diff_pct < 0 else "Unchanged")

        b_c = "risk-banner-low" if s_risk == "Low Risk" else ("risk-banner-medium" if s_risk == "Medium Risk" else "risk-banner-high")

        st.markdown(f"""
<div class="{b_c}" style="margin-top: 15px;">
    <div style="font-size: 0.8rem; font-weight: 700; color: #64748B; text-transform: uppercase;">Baseline Churn Probability</div>
    <div style="font-size: 1.1rem; font-weight: 700; color: #475569;">{base_prob_pct}%</div>
    <div style="font-size: 0.8rem; font-weight: 700; color: #64748B; text-transform: uppercase; margin-top: 8px;">Simulated Churn Probability</div>
    <div style="font-size: 2rem; font-weight: 800; color: #0F172A;">{new_prob_pct}%</div>
    <div style="font-size: 0.8rem; font-weight: 700; color: #64748B; text-transform: uppercase; margin-top: 8px;">Assigned Risk Tier</div>
    <div style="font-size: 1.2rem; font-weight: 700;">{s_risk}</div>
</div>
""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(f"""
<div class="observation-box">
    <strong>What Changed?</strong><br>
    Churn probability changed from <strong>{base_prob_pct}%</strong> to <strong>{new_prob_pct}%</strong> ({diff_str}).
</div>
""", unsafe_allow_html=True)

        st.write("**Probability Meter:**")
        st.progress(float(s_prob))


# =========================================================
# 7. CUSTOMER SEGMENTATION PAGE
# =========================================================
elif nav_option == "Customer Segmentation":
    render_header(
        "Customer Behavioral Segmentation",
        "Analyze customer behavior segments based on tenure length, monthly charges, and churn risk."
    )

    seg_df = compute_customer_segmentation(df_scored)

    st.subheader("Customer Segment Overview")
    st.table(seg_df)

    st.markdown("<br>", unsafe_allow_html=True)
    st.subheader("Observed Churn Rate by Segment")

    fig_seg, ax_seg = plt.subplots(figsize=(7.5, 3.2))
    sns.barplot(data=seg_df, x='Segment Name', y='Observed Churn Rate (%)', palette='Blues_r', ax=ax_seg)
    ax_seg.set_ylabel("Churn Rate (%)", fontsize=8.5)
    ax_seg.set_xlabel("Customer Segment", fontsize=8.5)
    plt.xticks(rotation=20, ha='right', fontsize=8)
    ax_seg.set_ylim(0, 100)
    for p in ax_seg.patches:
        ax_seg.annotate(f"{p.get_height():.1f}%", (p.get_x() + p.get_width() / 2., p.get_height()),
                     ha='center', va='bottom', fontsize=8.5, xytext=(0, 2), textcoords='offset points')
    plt.tight_layout()
    st.pyplot(fig_seg)


# =========================================================
# 8. REPORTS PAGE
# =========================================================
elif nav_option == "Reports":
    render_header(
        "Executive Reports & Data Exports",
        "Export dataset records, ML prediction classifications, and risk summary statistics."
    )

    st.subheader("Summary Report Metrics")

    rep1, rep2, rep3 = st.columns(3)
    with rep1:
        st.markdown(f"""
<div class="kpi-card">
    <div class="kpi-label">Dataset Records</div>
    <div class="kpi-value">{len(df_scored):,}</div>
</div>
""", unsafe_allow_html=True)
    with rep2:
        st.markdown(f"""
<div class="kpi-card">
    <div class="kpi-label">Predicted High Risk Count</div>
    <div class="kpi-value" style="color: #DC2626;">{(df_scored['Risk_Level']=='High Risk').sum():,}</div>
</div>
""", unsafe_allow_html=True)
    with rep3:
        st.markdown(f"""
<div class="kpi-card">
    <div class="kpi-label">Logistic Regression Accuracy</div>
    <div class="kpi-value" style="color: #16A34A;">{model_metrics['accuracy']*100:.2f}%</div>
</div>
""", unsafe_allow_html=True)

    st.markdown("<br><hr><br>", unsafe_allow_html=True)
    st.subheader("Download Export Files")

    ex1, ex2, ex3 = st.columns(3)

    with ex1:
        st.markdown("##### 1. Raw Customer Dataset")
        st.markdown("Contains all 750 raw customer attributes.")
        raw_csv = df.to_csv(index=False).encode('utf-8')
        st.download_button("Download Raw CSV", raw_csv, "customer_churn_raw.csv", "text/csv")

    with ex2:
        st.markdown("##### 2. Scored Churn Predictions")
        st.markdown("Contains predictions and risk levels.")
        pred_csv = df_scored.to_csv(index=False).encode('utf-8')
        st.download_button("Download Predictions CSV", pred_csv, "customer_churn_predictions.csv", "text/csv")

    with ex3:
        st.markdown("##### 3. Executive Risk Summary")
        st.markdown("Plain text risk & model evaluation summary.")
        summary_txt = f"""CUSTOMER CHURN ANALYSIS REPORT
=================================
Total Customers: {len(df_scored):,}
Overall Churn Rate: {(df_scored['Churn']=='Yes').mean()*100:.1f}%

MODEL RISK SEGMENTATION:
- Low Risk (0-30%): {(df_scored['Risk_Level']=='Low Risk').sum():,} customers
- Medium Risk (31-60%): {(df_scored['Risk_Level']=='Medium Risk').sum():,} customers
- High Risk (61-100%): {(df_scored['Risk_Level']=='High Risk').sum():,} customers

LOGISTIC REGRESSION TEST ACCURACY: {model_metrics['accuracy']*100:.2f}%
"""
        st.download_button("Download Summary TXT", summary_txt.encode('utf-8'), "churn_summary_report.txt", "text/plain")
