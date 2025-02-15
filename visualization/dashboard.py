import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from analysis.technical_indicators import calculate_sma, calculate_ema, calculate_rsi
from forecasting.forecast_model import forecast_stock


def display_dashboard(ticker):
    # Load the stock data once
    data = pd.read_csv(f'data/{ticker}.csv')
    data['SMA_20'] = calculate_sma(data, 20)
    data['EMA_20'] = calculate_ema(data, 20)
    data['RSI_14'] = calculate_rsi(data)

    st.title(f'{ticker} Stock Analysis')

    # Price and Moving Averages with Plotly
    st.subheader('Price and Moving Averages')
    fig = go.Figure()

    fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], mode='lines', name='Close Price'))
    fig.add_trace(go.Scatter(x=data['Date'], y=data['SMA_20'], mode='lines', name='SMA 20'))
    fig.add_trace(go.Scatter(x=data['Date'], y=data['EMA_20'], mode='lines', name='EMA 20'))

    fig.update_layout(
        title='Stock Price with Moving Averages',
        xaxis_title='Date',
        yaxis_title='Price (USD)',
        hovermode='x unified',
        xaxis=dict(tickangle=45)
    )

    st.plotly_chart(fig)

    # RSI Indicator with Plotly
    st.subheader('RSI Indicator')
    fig_rsi = go.Figure()

    fig_rsi.add_trace(go.Scatter(x=data['Date'], y=data['RSI_14'], mode='lines', name='RSI 14'))
    fig_rsi.add_trace(go.Scatter(x=data['Date'], y=[70]*len(data), mode='lines', name='Overbought (70)', line=dict(dash='dash', color='red')))
    fig_rsi.add_trace(go.Scatter(x=data['Date'], y=[30]*len(data), mode='lines', name='Oversold (30)', line=dict(dash='dash', color='green')))

    fig_rsi.update_layout(
        title='Relative Strength Index (RSI)',
        xaxis_title='Date',
        yaxis_title='RSI Value',
        hovermode='x unified',
        xaxis=dict(tickangle=45)
    )

    st.plotly_chart(fig_rsi)

    # Initialize session state for forecast duration if not present (set to 1 year by default)
    if 'forecast_duration' not in st.session_state:
        st.session_state.forecast_duration = '1 year'  # Default to 1 year

    # Forecast Selection using clickable buttons (Removed for 1-year default forecast)
    st.subheader('Forecast Selection')
    forecast_duration = st.session_state.forecast_duration

    # Mapping of forecast duration to periods in days (1 year)
    duration_mapping = {
        '1 year': 365
    }

    periods = duration_mapping.get(forecast_duration, 365)  # Default to 1 year if undefined
    forecast, model_params, mae, rmse = forecast_stock(data, n_periods=periods)

    st.subheader(f'Forecast for the Next {forecast_duration}')
    
    # Forecast Chart with Plotly
    fig_forecast = go.Figure()

    # Actual Stock Prices (use last data points for actual values in forecast)
    actual_data = data[['Date', 'Close']].tail(periods)

    fig_forecast.add_trace(go.Scatter(x=actual_data['Date'], y=actual_data['Close'], mode='lines', name='Actual', line=dict(color='black', width=2)))
    fig_forecast.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], mode='lines', name='Forecast', line=dict(color='blue')))
    fig_forecast.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_lower'], fill='tonexty', mode='lines', name='Lower Bound', line=dict(color='lightblue', dash='dash')))
    fig_forecast.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_upper'], fill='tonexty', mode='lines', name='Upper Bound', line=dict(color='lightblue', dash='dash')))

    fig_forecast.update_layout(
        title=f'Stock Price Forecast for 1 Year',
        xaxis_title='Date',
        yaxis_title='Price (USD)',
        hovermode='x unified',
        xaxis=dict(tickangle=45)
    )

    st.plotly_chart(fig_forecast)

    # Display model performance metrics
    st.subheader('Model Performance')
    st.write(f"**Mean Absolute Error (MAE):** {mae:.4f}")
    st.write(f"**Root Mean Squared Error (RMSE):** {rmse:.4f}")

    # Optionally, display model parameters
    st.write("**Model Parameters**:")
    st.write(model_params)