import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# ==============================
# PAGE CONFIG
# ==============================

st.set_page_config(
    page_title="Minimum Variance Portfolio Optimizer",
    layout="wide"
)

st.markdown("""
<style>

/* -------- Global -------- */
html, body, [class*="css"]  {
    font-family: 'Inter', sans-serif;
}

/* Background */
.stApp {
    background: radial-gradient(circle at top left, #1a1f2b, #0e1117 60%);
    color: #e6e8eb;
}

/* -------- Cards -------- */
.card {
    background: rgba(255, 255, 255, 0.04);
    border-radius: 16px;
    padding: 24px;
    box-shadow: 0 8px 24px rgba(0,0,0,0.35);
    margin-bottom: 24px;
}

/* -------- Buttons -------- */
.stButton>button {
    background: linear-gradient(135deg, #4f46e5, #6366f1);
    color: white;
    border-radius: 12px;
    height: 3em;
    font-weight: 600;
    border: none;
}

.stButton>button:hover {
    background: linear-gradient(135deg, #6366f1, #818cf8);
}

/* -------- Metrics -------- */
[data-testid="stMetric"] {
    background: rgba(255,255,255,0.05);
    padding: 16px;
    border-radius: 14px;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
# 📈 Minimum Variance Portfolio Optimizer
### Professional portfolio construction using Modern Portfolio Theory
""")

# ==============================
# LOAD STOCK LIST
# ==============================

try:
    stocks_df = pd.read_csv("indian_stocks.csv")
except Exception:
    st.error("❌ Error loading indian_stocks.csv")
    st.stop()

stocks_df["display"] = stocks_df["name"] + " (" + stocks_df["ticker"] + ")"

# ==============================
# USER INPUT (CARD)
# ==============================

st.markdown('<div class="card">', unsafe_allow_html=True)

selected_stocks = st.multiselect(
    "🔍 Search and select stocks",
    options=stocks_df["display"],
    placeholder="Start typing: Reliance, Tata, HDFC...",
    help="Search by company name or ticker"
)

col1, col2 = st.columns(2)

with col1:
    start_date = st.date_input(
        "Start Date",
        pd.to_datetime("2020-01-01")
    )

with col2:
    end_date = st.date_input(
        "End Date",
        pd.to_datetime("2024-01-01")
    )

optimize_btn = st.button("🚀 Optimize Portfolio")

st.markdown('</div>', unsafe_allow_html=True)

# ==============================
# HELPER FUNCTIONS
# ==============================

def portfolio_volatility(weights, cov_matrix):
    return np.sqrt(weights.T @ cov_matrix @ weights)

def portfolio_performance(weights, mean_returns, cov_matrix):
    annual_return = np.sum(mean_returns * weights) * 252
    annual_volatility = portfolio_volatility(weights, cov_matrix) * np.sqrt(252)
    return annual_return, annual_volatility, annual_volatility ** 2

def risk_contribution(weights, cov_matrix):
    portfolio_var = weights.T @ cov_matrix @ weights
    marginal_risk = cov_matrix @ weights
    return weights * marginal_risk / portfolio_var

@st.cache_data(show_spinner=False)
def load_stock_data(tickers, start, end):
    return yf.download(
        tickers,
        start=start,
        end=end,
        auto_adjust=True,
        progress=False
    )["Close"]

# ==============================
# MAIN LOGIC
# ==============================

if optimize_btn:

    if len(selected_stocks) < 2:
        st.error("❌ Please select at least two stocks.")
        st.stop()

    if start_date >= end_date:
        st.error("❌ Start date must be before end date.")
        st.stop()

    tickers = [s.split("(")[-1].replace(")", "") for s in selected_stocks]

    with st.spinner("📡 Fetching stock data..."):
        data = load_stock_data(tickers, start_date, end_date)

    data = data.dropna(axis=1, how="all")

    if data.shape[1] < 2:
        st.error("❌ Not enough valid stock data.")
        st.stop()

    returns = data.pct_change().dropna()
    mean_returns = returns.mean()
    cov_matrix = returns.cov()

    n = len(data.columns)
    init_weights = np.ones(n) / n

    constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
    bounds = tuple((0, 1) for _ in range(n))

    result = minimize(
        portfolio_volatility,
        init_weights,
        args=(cov_matrix,),
        method="SLSQP",
        bounds=bounds,
        constraints=constraints
    )

    if not result.success:
        st.error("⚠️ Optimization failed.")
        st.stop()

    weights = result.x

    allocation_df = pd.DataFrame({
        "Stock": data.columns,
        "Weight": weights
    })

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📌 Optimal Portfolio Weights")
        st.dataframe(allocation_df.style.format({"Weight": "{:.2%}"}))

    with col2:
        fig, ax = plt.subplots()
        ax.pie(weights, labels=data.columns, autopct="%1.1f%%", startangle=90)
        ax.set_title("Portfolio Allocation")
        st.pyplot(fig)

    ret, vol, var = portfolio_performance(weights, mean_returns, cov_matrix)

    st.subheader("📊 Portfolio Performance")
    c1, c2, c3 = st.columns(3)
    c1.metric("Expected Annual Return", f"{ret:.2%}")
    c2.metric("Annual Volatility", f"{vol:.2%}")
    c3.metric("Variance", f"{var:.6f}")

    risk_df = allocation_df.copy()
    risk_df["Risk Contribution"] = risk_contribution(weights, cov_matrix)

    st.subheader("⚠️ Risk Contribution by Stock")
    st.bar_chart(risk_df.set_index("Stock")["Risk Contribution"])
