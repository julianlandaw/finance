import yfinance as yf
import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import os
from functools import reduce
import sys

def load_portfolio_from_args(args):
    """
    Determine how to load the portfolio:
    - If args[0] ends with '.csv', load CSV
    - If args contain alternating tickers and shares, build DataFrame
    - If no args, default to 'Portfolio.csv'
    """
    if not args:
        default_csv = "Portfolio.csv"
        if os.path.exists(default_csv):
            print(f"📂 No arguments provided — loading default file: {default_csv}")
            return pd.read_csv(default_csv)
        else:
            raise FileNotFoundError(
                "❌ No input provided and 'Portfolio.csv' not found.\n"
                "Please run either:\n"
                "  python3 stockanalyzer.py my_portfolio.csv\n"
                "or\n"
                "  python3 stockanalyzer.py AAPL 100 MSFT 50 TSLA 25"
            )

    if len(args) == 1 and args[0].lower().endswith(".csv"):
        csv_file = args[0]
        if not os.path.exists(csv_file):
            raise FileNotFoundError(f"❌ File not found: {csv_file}")
        print(f"📄 Loading portfolio from CSV: {csv_file}")
        return pd.read_csv(csv_file)

    if len(args) % 2 != 0:
        raise ValueError("❌ Invalid input format.\n"
                         "Please provide ticker-share pairs, e.g.:\n"
                         "  python3 stockanalyzer.py AAPL 100 MSFT 50")

    tickers = args[::2]
    shares = args[1::2]

    try:
        shares = [int(s) for s in shares]
    except ValueError:
        raise ValueError("❌ Share counts must be integers.\n"
                         "Example: python3 stockanalyzer.py AAPL 100 GOOG 200")

    portfolio = pd.DataFrame({"Ticker": tickers, "Shares": shares})
    print("📊 Portfolio loaded from command-line input:")
    print(portfolio)
    return portfolio

def get_stock_data(tickers, period="1y", interval="1d"):
    data = {}
    for ticker in tickers:
        print(f"Fetching data for {ticker}...")
        df = yf.download(tickers=ticker, period=period, interval=interval)
        data[ticker] = df
    return data

def prepare_prophet_df(df, ticker):
    ds = df['Close'][ticker].index
    y = df['Close'][ticker].astype(float)
    prophet_df = pd.DataFrame({'ds': ds, 'y': y})
    prophet_df = prophet_df.dropna(subset=['ds','y'])
    return prophet_df

def forecast_stock(ticker, df, periods, interval="1d", pdf_pages=None):
    interval_map = {"1d":"D","1h":"H","30m":"30min","15m":"15min","5m":"5min","1m":"1min"}
    freq = interval_map.get(interval, "D")

    prophet_df = prepare_prophet_df(df, ticker)
    model = Prophet(daily_seasonality=True)
    model.fit(prophet_df)

    future = model.make_future_dataframe(periods=periods, freq=freq)
    forecast = model.predict(future)
    forecast = forecast[['ds','yhat','yhat_lower','yhat_upper']]
    forecast = forecast.rename(columns={
        'yhat': f'{ticker}_yhat',
        'yhat_lower': f'{ticker}_yhat_lower',
        'yhat_upper': f'{ticker}_yhat_upper'
    })

    fig1 = model.plot(forecast.rename(columns={
        f'{ticker}_yhat':'yhat', f'{ticker}_yhat_lower':'yhat_lower', f'{ticker}_yhat_upper':'yhat_upper'
    }))
    plt.title(f"{ticker} Forecasted Stock Price")
    fig1.tight_layout()
    if pdf_pages:
        pdf_pages.savefig(fig1)
        plt.close(fig1)
    else:
        plt.show()

    return forecast

if __name__ == "__main__":
    args = sys.argv[1:]
    portfolio_df = load_portfolio_from_args(args)
    # Settings
    # portfolio_csv = "portfolio.csv"  # path to portfolio CSV
    period = "1y"
    interval = "1d"  # daily or intraday
    intraday_forecast_periods = 50   # number of periods for intraday

    # Load portfolio CSV
    # portfolio_df = pd.read_csv(portfolio_csv)
    portfolio = dict(zip(portfolio_df['Ticker'], portfolio_df['Shares']))
    tickers = list(portfolio.keys())

    # Forecast horizon
    if interval=="1d":
        forecast_periods = 252  # ~1 year trading days
    else:
        forecast_periods = intraday_forecast_periods

    os.makedirs("forecasts", exist_ok=True)
    stock_data = get_stock_data(tickers, period, interval)

    # PDF for all plots
    pdf_path = os.path.join("forecasts","all_forecasts_plots.pdf")
    pdf_pages = PdfPages(pdf_path)

    # Forecast tickers
    all_forecasts = []
    for ticker, df in stock_data.items():
        print(f"\nForecasting for {ticker}...")
        forecast = forecast_stock(ticker, df, forecast_periods, interval, pdf_pages)
        all_forecasts.append(forecast)

    # Merge forecasts
    combined_forecasts = reduce(lambda l,r: pd.merge(l,r,on='ds',how='outer'), all_forecasts)
    combined_forecasts = combined_forecasts.sort_values('ds').reset_index(drop=True)
    forecast_cols = [c for c in combined_forecasts.columns if c!="ds"]
    combined_forecasts[forecast_cols] = combined_forecasts[forecast_cols].round(2)

    # Save ticker CSV
    csv_path = os.path.join("forecasts","all_forecasts_wide.csv")
    combined_forecasts.to_csv(csv_path,index=False)
    print(f"\n✅ All ticker forecasts saved to {csv_path}")

    # Portfolio forecast
    portfolio_forecast_df = pd.DataFrame()
    portfolio_forecast_df['ds'] = combined_forecasts['ds']
    portfolio_forecast_df['Portfolio_yhat'] = sum(combined_forecasts[f"{t}_yhat"]*shares for t,shares in portfolio.items())
    portfolio_forecast_df['Portfolio_yhat_lower'] = sum(combined_forecasts[f"{t}_yhat_lower"]*shares for t,shares in portfolio.items())
    portfolio_forecast_df['Portfolio_yhat_upper'] = sum(combined_forecasts[f"{t}_yhat_upper"]*shares for t,shares in portfolio.items())
    portfolio_forecast_df[['Portfolio_yhat','Portfolio_yhat_lower','Portfolio_yhat_upper']] = \
        portfolio_forecast_df[['Portfolio_yhat','Portfolio_yhat_lower','Portfolio_yhat_upper']].round(2)

    # Save portfolio CSV
    portfolio_csv_path = os.path.join("forecasts","portfolio_forecast.csv")
    portfolio_forecast_df.to_csv(portfolio_csv_path,index=False)
    print(f"\n✅ Portfolio forecast saved to {portfolio_csv_path}")

    # Plot portfolio
    fig_portfolio, ax = plt.subplots(figsize=(10,5))
    ax.plot(portfolio_forecast_df['ds'], portfolio_forecast_df['Portfolio_yhat'], label='Portfolio Forecast')
    ax.fill_between(portfolio_forecast_df['ds'],
                    portfolio_forecast_df['Portfolio_yhat_lower'],
                    portfolio_forecast_df['Portfolio_yhat_upper'],
                    color='lightblue', alpha=0.4, label='Confidence Interval')
    ax.set_title("Portfolio Forecasted Value")
    ax.set_xlabel("Date")
    ax.set_ylabel("Portfolio Value")
    ax.legend()
    fig_portfolio.tight_layout()
    pdf_pages.savefig(fig_portfolio)
    plt.close(fig_portfolio)

    # Close PDF
    pdf_pages.close()
    print(f"\n✅ All plots including portfolio saved to {pdf_path}")