from data.data_loader import fetch_stock_data
from visualization.dashboard import display_dashboard
import streamlit as st

def main():
    st.sidebar.title('Stock Analysis App')
    ticker = st.sidebar.text_input('Enter Stock Ticker', 'AAPL')
    if st.sidebar.button('Analyze Stock'):
        fetch_stock_data(ticker)
        display_dashboard(ticker)

if __name__ == '__main__':
    main()