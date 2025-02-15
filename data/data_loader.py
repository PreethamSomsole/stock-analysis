import yfinance as yf

def fetch_stock_data(ticker, period='1y', interval='1d'):
    data = yf.download(ticker, period=period, interval=interval)
    data.to_csv(f'data/{ticker}.csv')
    return data