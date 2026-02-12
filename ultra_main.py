import numpy as np
import pandas as pd
import yfinance as yf
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import sys

# ==============================
# USER INPUT & VALIDATION
# ==============================

print("Enter Indian stock tickers separated by commas")
print("Example: RELIANCE.NS, TCS.NS, HDFCBANK.NS\n")

user_input = input("Stock Tickers: ")

stocks = [s.strip().upper() for s in user_input.split(",") if s.strip()]

if len(stocks) < 2:
    print("❌ Please enter at least 2 stock tickers.")
    sys.exit()

print(f"\nFetching data for: {stocks}")


# ==============================
# FETCH HISTORICAL DATA
# ==============================

try:
    data = yf.download(
        stocks,
        start="2020-01-01",
        end="2024-01-01",
        auto_adjust=True,
        progress=False
    )["Close"]
except Exception as e:
    print("❌ Error fetching data:", e)
    sys.exit()

# Drop stocks with no data
data = data.dropna(axis=1, how="all")

valid_stocks = list(data.columns)

if len(valid_stocks) < 2:
    print("❌ Not enough valid stocks after data validation.")
    sys.exit()

print(f"✅ Valid stocks used: {valid_stocks}")


# ==============================
# RETURNS & STATISTICS
# ==============================

returns = data.pct_change().dropna()
mean_returns = returns.mean()
cov_matrix = returns.cov()


# ==============================
# PORTFOLIO FUNCTIONS
# ==============================

def portfolio_volatility(weights, cov_matrix):
    return np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))


def portfolio_performance(weights, mean_returns, cov_matrix):
    annual_return = np.sum(mean_returns * weights) * 252
    annual_volatility = portfolio_volatility(weights, cov_matrix) * np.sqrt(252)
    variance = annual_volatility ** 2
    return annual_return, annual_volatility, variance


def risk_contribution(weights, cov_matrix):
    portfolio_var = np.dot(weights.T, np.dot(cov_matrix, weights))
    marginal_risk = np.dot(cov_matrix, weights)
    return weights * marginal_risk / portfolio_var


# ==============================
# OPTIMIZATION SETUP
# ==============================

num_assets = len(valid_stocks)

constraints = ({
    'type': 'eq',
    'fun': lambda w: np.sum(w) - 1
})

bounds = tuple((0, 1) for _ in range(num_assets))
initial_weights = np.array(num_assets * [1 / num_assets])


# ==============================
# MINIMUM VARIANCE OPTIMIZATION
# ==============================

optimized = minimize(
    portfolio_volatility,
    initial_weights,
    args=(cov_matrix,),
    method='SLSQP',
    bounds=bounds,
    constraints=constraints
)

optimal_weights = optimized.x


# ==============================
# RESULTS
# ==============================

allocation_df = pd.DataFrame({
    "Stock": valid_stocks,
    "Optimal Weight": optimal_weights
})

print("\n📌 Optimal Portfolio Allocation")
print(allocation_df)


opt_return, opt_risk, opt_variance = portfolio_performance(
    optimal_weights, mean_returns, cov_matrix
)

equal_weights = np.array(num_assets * [1 / num_assets])
eq_return, eq_risk, eq_variance = portfolio_performance(
    equal_weights, mean_returns, cov_matrix
)

print("\n📊 Minimum Variance Portfolio")
print(f"Expected Annual Return: {opt_return:.2%}")
print(f"Annual Risk: {opt_risk:.2%}")
print(f"Variance: {opt_variance:.6f}")

print("\n📊 Equal Weight Portfolio")
print(f"Expected Annual Return: {eq_return:.2%}")
print(f"Annual Risk: {eq_risk:.2%}")
print(f"Variance: {eq_variance:.6f}")


# ==============================
# RISK CONTRIBUTION
# ==============================

risk_contrib = risk_contribution(optimal_weights, cov_matrix)

risk_df = pd.DataFrame({
    "Stock": valid_stocks,
    "Weight": optimal_weights,
    "Risk Contribution": risk_contrib
})

print("\n⚠️ Risk Contribution by Asset")
print(risk_df)


# ==============================
# VISUALIZATIONS
# ==============================

plt.figure()
plt.pie(optimal_weights, labels=valid_stocks, autopct='%1.1f%%', startangle=90)
plt.title("Minimum Variance Portfolio Allocation")
plt.show()

plt.figure()
plt.bar(valid_stocks, optimal_weights)
plt.title("Optimal Portfolio Weights")
plt.ylabel("Weight")
plt.xticks(rotation=45)
plt.show()

plt.figure()
plt.bar(valid_stocks, risk_contrib)
plt.title("Risk Contribution by Stock")
plt.ylabel("Contribution")
plt.xticks(rotation=45)
plt.show()


# ==============================
# ROLLING WINDOW OPTIMIZER
# ==============================

def rolling_min_variance(returns, window_size=252):
    weights_history = []
    dates = []

    for i in range(window_size, len(returns)):
        window_returns = returns.iloc[i - window_size:i]
        cov_matrix = window_returns.cov()

        n = window_returns.shape[1]
        init_weights = np.array(n * [1 / n])

        result = minimize(
            portfolio_volatility,
            init_weights,
            args=(cov_matrix,),
            method='SLSQP',
            bounds=tuple((0, 1) for _ in range(n)),
            constraints=({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})
        )

        weights_history.append(result.x)
        dates.append(returns.index[i])

    return pd.DataFrame(weights_history, index=dates, columns=returns.columns)


rolling_weights = rolling_min_variance(returns)

plt.figure()
for stock in rolling_weights.columns:
    plt.plot(rolling_weights.index, rolling_weights[stock], label=stock)

plt.title("Rolling Minimum Variance Portfolio Weights (1-Year Window)")
plt.xlabel("Date")
plt.ylabel("Weight")
plt.legend()
plt.show()