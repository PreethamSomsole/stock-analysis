from prophet import Prophet
from pmdarima import auto_arima
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error

def forecast_stock(data, n_periods, model_type='best'):
    # Prepare the data (Ensure that df has only 'ds' and 'y')
    print("Type of n_periods (inside forecast_stock):", type(n_periods))  # Should be <class 'int'>
    print("Value of n_periods (inside forecast_stock):", n_periods)
    df = data[['Date', 'Close']].rename(columns={'Date': 'ds', 'Close': 'y'})
    df['ds'] = pd.to_datetime(df['ds'])
    print("Type of df['ds']:", df['ds'].dtype) # Should be datetime64[ns]
    

    # Function to calculate error metrics
    def calculate_error(actual, predicted):
        mae = mean_absolute_error(actual, predicted)
        rmse = np.sqrt(mean_squared_error(actual, predicted))
        return mae, rmse

    # Initialize variables for evaluation
    mae_prophet = rmse_prophet = mae_arima = rmse_arima = float('inf')
    prophet_params = arima_params = {}

    if model_type == 'prophet' or model_type == 'best':
        # Prophet model
        model = Prophet(
            yearly_seasonality=True,  
            weekly_seasonality=True,  
            daily_seasonality=False,  
            changepoint_prior_scale=0.1  
        )
        model.fit(df)
        print("Dataframe to fit model:", df.head())

        # Check if periods is being passed correctly
        print(f"Periods passed to make_future_dataframe: {n_periods}")

        # Generate future dataframe for prediction (Ensure periods argument is passed correctly)
        future = model.make_future_dataframe(df, periods=n_periods)  # Only passing 'periods' here

        # Predict using Prophet
        forecast = model.predict(future)

        # Calculate error on historical data (comparing actual vs predicted)
        predicted_prophet = forecast['yhat'][-len(df):]  # Historical forecast
        mae_prophet, rmse_prophet = calculate_error(df['y'], predicted_prophet)

        # Extract model parameters used by Prophet
        prophet_params = {
            'seasonality': model.seasonality,
            'changepoint_prior_scale': model.changepoint_prior_scale,
            'holidays': model.holidays
        }

        if model_type == 'prophet':  # If explicitly requested Prophet
            return forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']], prophet_params, mae_prophet, rmse_prophet

    if model_type == 'arima' or model_type == 'best':
        # ARIMA model (using auto_arima for automatic selection)
        model = auto_arima(df['y'], seasonal=False, trace=True, stepwise=True)
        forecast_arima = model.predict(n_periods=n_periods)
        
        # Create a forecast dataframe
        forecast_dates = pd.date_range(df['ds'].max(), periods=n_periods, freq='D')
        forecast_df = pd.DataFrame({
            'ds': forecast_dates,
            'yhat': forecast_arima,
            'yhat_lower': np.nan,  
            'yhat_upper': np.nan  
        })

        # Calculate error on historical data using the ARIMA model
        mae_arima, rmse_arima = calculate_error(df['y'], model.predict_in_sample())

        # ARIMA parameters (extracted from auto_arima)
        arima_params = {
            'order': model.order,  # AR, I, MA order tuple (p, d, q)
            'aic': model.aic(),
            'bic': model.bic()
        }

        if model_type == 'arima':  # If explicitly requested ARIMA
            return forecast_df, arima_params, mae_arima, rmse_arima

    # Evaluate and choose the best model based on MAE and RMSE
    if model_type == 'best':
        if mae_prophet < mae_arima and rmse_prophet < rmse_arima:
            return forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']], prophet_params, mae_prophet, rmse_prophet
        else:
            return forecast_df, arima_params, mae_arima, rmse_arima