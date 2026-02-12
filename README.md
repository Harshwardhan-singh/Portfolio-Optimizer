# Portfolio Optimizer

This repository provides professional-grade tools for constructing and analyzing investment portfolios using **Modern Portfolio Theory (MPT)**. Specifically, it focuses on identifying the **Minimum Variance Portfolio** to minimize risk for a given set of Indian stocks.

The project includes two primary interfaces:
1.  **Streamlit Web App**: An interactive, styled dashboard for stock selection and visualization.
2.  **CLI Tool**: A command-line script for quick optimization and advanced rolling window analysis.

---

## 🚀 Features

* **Stock Selection**: Fetch real-time historical data for Indian stocks via `yfinance`.
* **Portfolio Optimization**: Uses the **SLSQP** (Sequential Least Squares Programming) algorithm to find the weight distribution that minimizes portfolio volatility.
* **Performance Metrics**: Calculates Expected Annual Return, Annual Volatility, and Variance.
* **Risk Analysis**: Visualizes the marginal risk contribution of each asset within the portfolio.
* **Rolling Window Analysis (CLI only)**: Tracks how optimal weights shift over a 1-year rolling window.
* **Interactive Visualizations**: Includes pie charts for allocation, bar charts for risk, and line graphs for weight history.

---

## 🛠️ Installation

### 1. Clone the Repository
```bash
git clone https://github.com/harshwardhan-singh/portfolio-optimizer.git
cd portfolio-optimizer
```
### 2. Set Up a Virtual Environment (Recommended)
```Bash
python -m venv venv
```
#### On Windows:
```
venv\Scripts\activate
```
#### On Mac/Linux:
```
source venv/bin/activate
```
### 3. Install Dependencies
```Bash
pip install numpy pandas yfinance streamlit matplotlib scipy
```

## 💻 Usage
### Option 1: Streamlit Web Dashboard
The web app provides a user-friendly interface with custom CSS styling.
Note: Ensure indian_stocks.csv is present in the root directory for stock searching.

```Bash
streamlit run app.py
```
How to use: Search and select stocks from the dropdown, choose your date range, and click Optimize Portfolio.

### Option 2: Command Line Interface (CLI)
The CLI tool is optimized for scripts and provides a comparison against an Equal Weight Portfolio.

```Bash
python ultra_main.py
```
How to use: Enter Indian stock tickers (with .NS suffix for NSE) separated by commas when prompted (e.g., RELIANCE.NS, TCS.NS, HDFCBANK.NS).


## 📁 File Structure
app.py: Streamlit application code including custom UI styling and interactive widgets.

ultra_main.py: CLI script featuring optimization, risk contribution analysis, and rolling window simulations.

indian_stocks.csv: (Required for app.py) Contains stock names and tickers for the search interface.

## ⚠️ Disclaimer
This tool is for educational and informational purposes only. It does not constitute financial advice. Always perform your own due diligence before making investment decisions.
