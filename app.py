import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="Transaction Analysis", layout="wide")
st.title("Transaction Analysis & Fraud Detection App")

# ------------------- Upload CSV or Excel -------------------
uploaded_file = st.file_uploader(
    "Upload your transaction CSV or Excel file", type=["csv", "xlsx"]
)

# Sample file download
st.subheader("Try with a sample file")
with open("sample_transactions.xlsx", "rb") as f:
    st.download_button(
        label="Download Sample Excel File",
        data=f,
        file_name="sample_transactions.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

if uploaded_file:
    # ------------------- Read file -------------------
    try:
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
    except ImportError:
        st.error("Missing dependency 'openpyxl'. Install it: pip install openpyxl")
        st.stop()

    st.subheader("Raw Data (First 10 rows)")
    st.dataframe(df.head(10))

    # ------------------- Column Mapping -------------------
    st.subheader("Map your columns to standard fields")
    amount_col = st.selectbox("Transaction amount column:", [""] + list(df.columns))
    account_col = st.selectbox("Account ID column:", [""] + list(df.columns))
    type_col = st.selectbox("Transaction type column:", [""] + list(df.columns))
    merchant_col = st.selectbox("Merchant column (optional):", ["None"] + list(df.columns))
    date_col = st.selectbox("Transaction date column (optional):", ["None"] + list(df.columns))

    if not amount_col or not account_col or not type_col:
        st.error("Please select amount, account ID, and type columns.")
        st.stop()

    # Rename columns
    df = df.rename(columns={amount_col: "amount", account_col: "account_id", type_col: "type"})
    if merchant_col != "None":
        df = df.rename(columns={merchant_col: "merchant"})
    else:
        df["merchant"] = "unknown"

    # ------------------- Data Cleaning -------------------
    initial_rows = len(df)
    df.dropna(how='all', inplace=True)
    df.dropna(axis=1, how='all', inplace=True)
    df['amount'] = pd.to_numeric(df['amount'], errors='coerce')
    df.dropna(subset=['amount'], inplace=True)
    df['type'] = df['type'].fillna('unknown')
    df['account_id'] = df['account_id'].fillna('unknown')
    if date_col != "None":
        df['date'] = pd.to_datetime(df[date_col], errors='coerce')
    else:
        df['date'] = pd.NaT

    rows_dropped = initial_rows - len(df)
    st.info(f"Data cleaned: {rows_dropped} row(s) dropped.")

    # ------------------- Sidebar Filters -------------------
    st.sidebar.subheader("Filters & Thresholds (optional)")
    start_date = st.sidebar.date_input("Start Date (optional)", value=None)
    end_date = st.sidebar.date_input("End Date (optional)", value=None)
    user_threshold = st.sidebar.number_input("High transaction amount threshold", min_value=0, value=0, step=1)
    rapid_txn_threshold = st.sidebar.number_input("Rapid transactions/hour threshold", min_value=0, value=0, step=1)
    risky_merchants = st.sidebar.multiselect("High-risk merchants (optional)", df['merchant'].unique())

    # Apply date filter only if both dates selected
    if start_date and end_date and 'date' in df.columns and not df['date'].isna().all():
        df = df[(df['date'] >= pd.to_datetime(start_date)) & (df['date'] <= pd.to_datetime(end_date))]
        st.info(f"Filter applied: {start_date} to {end_date}. All analyses reflect this filter.")

    # ------------------- Basic Metrics -------------------
    st.subheader("Summary Metrics")
    total_txn = len(df)
    avg_amount = df['amount'].mean()
    max_amount = df['amount'].max()
    min_amount = df['amount'].min()
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Transactions", total_txn)
    col2.metric("Average Amount", f"${avg_amount:.2f}")
    col3.metric("Max Amount", f"${max_amount:.2f}")
    col4.metric("Min Amount", f"${min_amount:.2f}")

    # ------------------- Fraud Detection -------------------
    st.subheader("Advanced Fraud Detection")
    df['flagged'] = False

    # High-value transactions
    if user_threshold > 0:
        df.loc[df['amount'] > user_threshold, 'flagged'] = True

    # Rapid repeated transactions
    if rapid_txn_threshold > 0 and 'date' in df.columns and not df['date'].isna().all():
        df_sorted = df.sort_values(['account_id','date'])
        df_sorted['time_diff'] = df_sorted.groupby('account_id')['date'].diff().dt.total_seconds().fillna(np.inf)
        df_sorted['rapid_flag'] = df_sorted['time_diff'] < 3600
        rapid_ids = df_sorted.groupby('account_id')['rapid_flag'].sum()[lambda x: x>=rapid_txn_threshold].index
        df.loc[df['account_id'].isin(rapid_ids), 'flagged'] = True

    # Unusually large relative to account mean
    df['acct_mean'] = df.groupby('account_id')['amount'].transform('mean')
    df['acct_std'] = df.groupby('account_id')['amount'].transform('std').fillna(0)
    df.loc[df['amount'] > df['acct_mean'] + 2*df['acct_std'], 'flagged'] = True

    # Weekend transactions
    if 'date' in df.columns and not df['date'].isna().all():
        df['weekend'] = df['date'].dt.weekday >= 5
        df.loc[df['weekend'], 'flagged'] = True

    # Risky merchants
    if risky_merchants:
        df.loc[df['merchant'].isin(risky_merchants), 'flagged'] = True

    flagged_df = df[df['flagged']].copy()
    st.metric("Flagged Transactions", len(flagged_df), f"{len(flagged_df)/total_txn*100:.1f}%")

    # Show flagged transaction table
    if len(flagged_df) > 0:
        st.write("### Flagged Transaction Details")
        st.dataframe(flagged_df)

    # ------------------- Visualizations with toggle -------------------

    # Transaction Amount Distribution
    st.subheader("Transaction Amount Distribution")
    if st.checkbox("Show Amount Distribution Chart"):
        fig, ax = plt.subplots(figsize=(4,3))
        ax.hist(df['amount'], bins=50, color='skyblue', edgecolor='black')
        ax.set_xlabel("Amount")
        ax.set_ylabel("Frequency")
        st.pyplot(fig)
    if st.checkbox("Show Amount Distribution Table"):
        st.dataframe(df[['account_id','amount','merchant','type']].sort_values('amount', ascending=False))

    # Transaction Type Distribution
    st.subheader("Transaction Type Distribution")
    if st.checkbox("Show Transaction Type Chart"):
        type_counts = df['type'].value_counts()
        fig1, ax1 = plt.subplots(figsize=(4,3))
        wedges, _ = ax1.pie(type_counts, labels=None, autopct=None, startangle=90)
        ax1.axis('equal')
        ax1.legend(wedges, type_counts.index, title="Transaction Types", loc="center left", bbox_to_anchor=(1,0,0.5,1))
        st.pyplot(fig1)
    if st.checkbox("Show Transaction Type Table"):
        st.dataframe(df.groupby('type')['amount'].sum().reset_index().sort_values('amount', ascending=False))

    # Top 10 Merchants by Transaction Value
    st.subheader("Top 10 Merchants by Transaction Value")
    if st.checkbox("Show Top Merchants Chart"):
        top_merchants = df.groupby('merchant')['amount'].sum().sort_values(ascending=False).head(10)
        fig2, ax2 = plt.subplots(figsize=(4,3))
        top_merchants.plot(kind='bar', ax=ax2, color='lightgreen')
        ax2.set_ylabel("Total Amount")
        ax2.set_xlabel("Merchant")
        st.pyplot(fig2)
    if st.checkbox("Show Top Merchants Table"):
        st.dataframe(df.groupby('merchant')['amount'].sum().sort_values(ascending=False).head(10).reset_index())

    # Top/Bottom 10 Merchants per Account
    top_merchants_per_account = df.groupby(['account_id','merchant'])['amount'].sum().reset_index()

    st.subheader("Top 10 Highest Spending Merchants per Account")
    if st.checkbox("Show Top Merchants per Account Chart"):
        top_chart = top_merchants_per_account.groupby('merchant')['amount'].sum().sort_values(ascending=False).head(10)
        fig3, ax3 = plt.subplots(figsize=(4,3))
        top_chart.plot(kind='bar', ax=ax3, color='orange')
        ax3.set_ylabel("Total Amount")
        ax3.set_xlabel("Merchant")
        st.pyplot(fig3)
    if st.checkbox("Show Top Merchants per Account Table"):
        idx_top = top_merchants_per_account.groupby('account_id')['amount'].idxmax()
        top_per_account = top_merchants_per_account.loc[idx_top].sort_values('amount', ascending=False).head(10)
        st.dataframe(top_per_account)

    st.subheader("Top 10 Lowest Spending Merchants per Account")
    if st.checkbox("Show Lowest Merchants per Account Chart"):
        low_chart = top_merchants_per_account.groupby('merchant')['amount'].sum().sort_values(ascending=True).head(10)
        fig4, ax4 = plt.subplots(figsize=(4,3))
        low_chart.plot(kind='bar', ax=ax4, color='purple')
        ax4.set_ylabel("Total Amount")
        ax4.set_xlabel("Merchant")
        st.pyplot(fig4)
    if st.checkbox("Show Lowest Merchants per Account Table"):
        idx_low = top_merchants_per_account.groupby('account_id')['amount'].idxmin()
        low_per_account = top_merchants_per_account.loc[idx_low].sort_values('amount', ascending=True).head(10)
        st.dataframe(low_per_account)

    # ------------------- Download Flagged -------------------
    st.subheader("Download Flagged Transactions")
    csv = flagged_df.to_csv(index=False).encode('utf-8')
    st.download_button("Download CSV", csv, "flagged_transactions.csv", "text/csv")
