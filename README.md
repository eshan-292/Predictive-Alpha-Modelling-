# QRT Quant Quest: Alpha Research Contest

## Overview

In this contest you will engage in alpha research to develop systematic strategies for trading stocks on a daily basis using price and volume based features. By leveraging data analysis, statistical techniques, and machine learning tools, you will be required to identify patterns and build predictive models for profitable trading. The research involves analyzing over 15 years of historical market data to discover robust factors that consistently generate returns across various market conditions.

---

## Data and Features

### Underlying Price Data

The contest is based on derived features rather than raw price data. However, here are the original price components:
- **open:** The first traded price of a stock at the start of the trading day.
- **close:** The last traded price of a stock at the end of the trading day.
- **high:** The highest traded price during the trading day.
- **low:** The lowest traded price during the trading day.
- **volume:** The total number of shares traded during the trading day.

*Note: You do not have direct access to these raw prices, as the provided variables are derived from them.*

### Provided Data Files

- **`features.parquet`**  
  Contains 25 anonymized features (`f1`, `f2`, …, `f25`) derived from the raw price data (open, high, low, close, volume). Each feature is represented as a matrix with rows indexed by trading dates and columns by stock identifiers (from 1 to 2167).  
  - **Training Period:** 2005-01-01 to 2014-12-31

- **`universe.parquet`**  
  A binary matrix indicating the tradeable stocks on each day. For a given day and stock, a value of 1 means the stock is in the trading universe, while 0 indicates that it is not eligible for trading.

- **`returns.parquet`**  
  Contains the daily close-to-close percentage returns for each stock.  
  - **Important:** These returns are used for backtesting your strategy during the training period. Returns cannot be used as an input feature in your strategy.

---

## Trading Setup and Constraints

### Portfolio Allocation

For each trading day \( t \) and stock \( s \), you will allocate a fraction of unit capital \( w(t, s) \):
- **Long Position:** \( w(t, s) > 0 \)
- **Short Position:** \( w(t, s) < 0 \)

### Constraints

1. **Unit Capital Constraint:**  
   \[
   \sum_{s} |w(t, s)| \leq 1 + 10^{-4}
   \]
2. **Maximum Weight Constraint:**  
   \[
   |w(t, s)| \leq 0.1 \quad \text{for each stock on each day}
   \]
3. **Dollar Neutral Constraint:**  
   \[
   \left| \sum_{s} w(t, s) \right| \leq 10^{-4}
   \]
4. **Universe Constraint:**  
   For stocks that are not in the trading universe (i.e., where `universe.parquet` is 0), \( w(t, s) \) must be 0.

---

## Portfolio Metrics

The performance of your portfolio will be evaluated using several metrics:

### Book Value
The total absolute capital invested on a given day.
\[
\text{BookValue}(t) = \sum_{s} |w(t, s)|
\]

### Traded Capital
The total amount of capital traded each day.
\[
\text{Traded}(t) = \sum_{s} |w(t, s) - w(t-1, s)|
\]

### Turnover
The percentage of total capital traded on an average day.
\[
\text{Turnover} = \left( \frac{\sum_{t} \text{Traded}(t)}{\sum_{t} \text{BookValue}(t)} \right) \times 100
\]

### Gross PnL (Profit and Loss)
The gross returns calculated as:
\[
\text{GrossPnL}(t) = \sum_{s} w(t, s) \times \text{returns}(t, s)
\]

### Net PnL
The net returns after accounting for trading costs, where a cost of 0.01% of the traded amount is applied each day.
\[
\text{NetPnL}(t) = \text{GrossPnL}(t) - 0.01\% \times \text{Traded}(t)
\]

### Sharpe Ratios

- **Gross Sharpe Ratio:**
  \[
  \text{Gross Sharpe Ratio} = \sqrt{252} \times \frac{\text{Mean(GrossPnL}(t))}{\text{StdDev(GrossPnL}(t))}
  \]
- **Net Sharpe Ratio:**
  \[
  \text{Net Sharpe Ratio} = \sqrt{252} \times \frac{\text{Mean(NetPnL}(t))}{\text{StdDev(NetPnL}(t))}
  \]

---

## Evaluation and Contest Phases

The contest is divided into three main phases:

### 1. Learning Phase
- **Duration:** First two hours.
- **Data Provided:** 2005-2014 (training period).
- **Objective:** Build and test your statistical models using the training data. Submit your portfolio positions on Kaggle to evaluate your performance based on the Net Sharpe Ratio.

### 2. Validation Phase
- **Duration:** Next two hours.
- **Data Provided:** 2005-2019.
- **Objective:** Validate and, if needed, fine-tune your models on the extended dataset. Submit your portfolio positions on Kaggle to see your performance.

*Note: After the Validation phase, you are not allowed to change your code. Be sure to push your final code to GitHub Classroom before moving forward.*

### 3. Production Phase
- **Duration:** Final one hour.
- **Data Provided:** 2005-2025.
- **Objective:** Run your final code (from the Validation phase) to generate the submission file containing your portfolio positions. This file will be submitted on Kaggle.

---

## Data Directory Structure

For each phase, the corresponding data will be available under a directory named after the phase:
- **Data/Learning**
- **Data/Validation**
- **Data/Production**

Each directory includes:
- `features.parquet`
- `returns.parquet`
- `universe.parquet`

---

## Final Judgement Metric

The final evaluation of your strategy in the Production phase will be based on the **Net Sharpe Ratio** calculated over the period from 2020 to 2025. This metric reflects risk-adjusted portfolio returns after subtracting trading costs.

---

Happy Trading and Good Luck!
