import yfinance as yf
import pandas as pd
import joblib
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def fetch_stock_data(symbol: str = "^NSEI", period: str = "90d", interval: str = "1d") -> pd.DataFrame:
    """
    Fetches historical stock data for the given symbol.
    
    Args:
        symbol (str): Ticker symbol, e.g., "^NSEI" for Nifty 50.
        period (str): Period of data to fetch (e.g., '1d', '5d', '1mo', '3mo', '60d', '1y').
        interval (str): Data interval (e.g., '1d', '1h', '5m').
    
    Returns:
        pd.DataFrame: Historical data or empty DataFrame on error.
    """
    try:
        stock_data = yf.Ticker(symbol)
        hist = stock_data.history(period=period, interval=interval)
        
        if hist.empty:
            print(f"No data found for symbol: {symbol}")
        return hist
    
    except Exception as e:
        print(f"Error fetching data for {symbol}: {e}")
        return pd.DataFrame()


def preprocess_data(df: pd.DataFrame, scaler: MinMaxScaler, sequence_length: int = 60):
    """
    Prepares the latest 60-day stock data for inference.
    """
    try:
        
        if df.empty:
            return None

        df = df[df['Volume'] > 0].copy()
        df.reset_index(drop=True, inplace=True)

        close_prices = df['Close'].values.reshape(-1, 1)
        scaled_data = scaler.transform(close_prices)

        if len(scaled_data) < sequence_length:
            print("Not enough data to generate input sequence.")
            return None

        last_60 = scaled_data[-sequence_length:]
        X = np.array([last_60])  # shape: (1, 60, 1)
        return X

    except Exception as e:
        print(f"Error during preprocessing: {e}")
        return None
