# 📈 Portfolio & Mortgage Tools

This repository contains two separate but related tools for financial analysis:

1. **`stockanalyzer.py`** — A Python script that analyzes portfolio data from a CSV file.  
2. **`mortgagecalculator.html`** — A modern, interactive mortgage payoff calculator built with HTML, CSS, and vanilla JavaScript.  
3. **`Portfolio.csv`** — A sample input file demonstrating the format used by the stock analyzer.

---

## 🧠 Overview

### **1. Stock Analyzer (`stockanalyzer.py`)**
The `stockanalyzer.py` script processes stock portfolio data from a CSV file (e.g., `Portfolio.csv`).  
It fetches stock prices, calculates performance metrics, and provides a summary of portfolio value and returns.

#### **Key Features**
- Reads portfolio holdings from a CSV file  
- Fetches historical or live stock data (e.g., via `yfinance`)  
- Computes:
  - Portfolio value and individual gains/losses  
  - Daily and cumulative returns  
  - Allocation by ticker or sector  
- Optionally generates plots or CSV summaries

#### **Usage**
```bash
python stockanalyzer.py Portfolio.csv
```

#### **Expected CSV Format (`Portfolio.csv`)**
| Ticker | Shares | PurchasePrice |
|--------|---------|----------------|
| AAPL   | 10      | 160.50         |
| MSFT   | 5       | 320.00         |
| TSLA   | 3       | 250.00         |

*(Your CSV can include additional columns like “Date” or “Sector” depending on the script’s configuration.)*

#### **Output**
- Console summary of portfolio performance  
- Optionally: generated charts or exported result files (e.g., `portfolio_analysis.csv`, `portfolio_summary.png`)

---

### **2. Mortgage Payoff Calculator (`mortgagecalculator.html`)**
An advanced, responsive web calculator for exploring mortgage scenarios — including custom payments, lump-sum principal reductions, and early payoff projections.

#### **Features**
- Compute monthly payments based on loan amount, rate, and term  
- Compare baseline amortization vs. early payoff with extra payments  
- Add **variable lump-sum payments** (by date or payment number)  
- Export amortization tables to CSV  
- View payoff charts with interactive tooltips  

#### **How to Use**
1. Open `mortgagecalculator.html` in your web browser.  
2. Enter your loan details:
   - Loan amount  
   - Interest rate  
   - Term or custom monthly payment  
3. Optionally add:
   - Start date  
   - Extra monthly payments  
   - One-time lump sum payments  
4. Click **“Calculate”** to view results, including:
   - Monthly payment  
   - Total interest paid  
   - Interest savings and months saved with extra payments  
   - Interactive payoff chart and amortization table  

---

## 🧩 File Summary

| File | Description |
|------|--------------|
| **Portfolio.csv** | Example portfolio input data |
| **stockanalyzer.py** | Python script for analyzing portfolio performance |
| **mortgagecalculator.html** | Standalone mortgage payoff calculator web app |

---

## ⚙️ Requirements

For `stockanalyzer.py`, you’ll need:
- **Python 3.8+**
- Required packages (install with `pip install -r requirements.txt` if applicable):
  ```bash
  pip install yfinance pandas matplotlib
  ```

---

## 🧾 License

This project is open-source and available under the **MIT License**.

---

## 💡 Future Improvements
- Integrate live price tracking for `stockanalyzer.py`
- Add GUI or web dashboard interface
- Combine portfolio growth and mortgage payoff simulations for holistic financial planning
