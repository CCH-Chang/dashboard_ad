import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from forecast import SARIMAXForecaster, LSTMForecaster, ForecastEvaluator, prepare_data
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="Forecast Models", layout="wide")

st.title("⚡ Energy Load Forecasting")
st.markdown("---")

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv('processed_energy_weather.csv')
    df['time'] = pd.to_datetime(df['time'])
    return df

df = load_data()

# Sidebar controls
st.sidebar.title("📊 Forecast Settings")

forecast_days = st.sidebar.multiselect(
    "Select forecast horizon (days)",
    options=[1, 3, 7],
    default=[1, 3]
)

if not forecast_days:
    st.warning("Please select at least one forecast horizon")
    st.stop()

train_ratio = st.sidebar.slider(
    "Train-Test Split Ratio",
    min_value=0.6,
    max_value=0.9,
    value=0.8,
    step=0.05
)

run_forecast = st.sidebar.button("🚀 Run Forecast", type="primary")

if run_forecast:
    with st.spinner("Running forecasts... This may take a few minutes..."):
        # Prepare data
        data = df['total load actual'].values.astype(float)
        split_point = int(len(data) * train_ratio)
        train_data = data[:split_point]
        test_data = data[split_point:]
        
        results_dict = {}
        
        for days in sorted(forecast_days):
            steps = days * 24
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader(f"📈 {days}-Day Forecast Results")
                
                with st.spinner(f"Running SARIMAX for {days} days..."):
                    sarimax = SARIMAXForecaster()
                    sarimax_success = sarimax.fit(train_data)
                    
                    if sarimax_success:
                        sarimax_forecast, sarimax_ci = sarimax.forecast(steps)
                        
                        if len(test_data) >= steps:
                            sarimax_metrics = ForecastEvaluator.calculate_metrics(
                                test_data[:steps],
                                sarimax_forecast
                            )
                        else:
                            sarimax_metrics = None
                        
                        results_dict[f'{days}D_sarimax'] = {
                            'forecast': sarimax_forecast,
                            'ci': sarimax_ci,
                            'metrics': sarimax_metrics
                        }
                        
                        st.success("✅ SARIMAX model completed")
                        if sarimax_metrics:
                            st.metric("RMSE", f"{sarimax_metrics['RMSE']:.2f}")
                    else:
                        st.error("❌ SARIMAX fitting failed")
            
            with col2:
                with st.spinner(f"Running LSTM for {days} days..."):
                    lstm = LSTMForecaster(lookback=24, epochs=50, batch_size=32)
                    lstm_success = lstm.fit(train_data)
                    
                    if lstm_success:
                        lstm_forecast = lstm.forecast(steps)
                        
                        if len(test_data) >= steps:
                            lstm_metrics = ForecastEvaluator.calculate_metrics(
                                test_data[:steps],
                                lstm_forecast
                            )
                        else:
                            lstm_metrics = None
                        
                        results_dict[f'{days}D_lstm'] = {
                            'forecast': lstm_forecast,
                            'metrics': lstm_metrics
                        }
                        
                        st.success("✅ LSTM model completed")
                        if lstm_metrics:
                            st.metric("RMSE", f"{lstm_metrics['RMSE']:.2f}")
                    else:
                        st.error("❌ LSTM fitting failed")
            
            # Visualization
            st.markdown("---")
            st.subheader(f"Forecast Comparison ({days} Day)")
            
            fig = go.Figure()
            
            # Add test data
            test_hours = np.arange(len(test_data[:steps]))
            fig.add_trace(go.Scatter(
                x=test_hours,
                y=test_data[:steps],
                mode='lines',
                name='Actual Test Data',
                line=dict(color='black', width=2)
            ))
            
            # Add SARIMAX forecast
            if f'{days}D_sarimax' in results_dict:
                sarimax_data = results_dict[f'{days}D_sarimax']
                fig.add_trace(go.Scatter(
                    x=test_hours,
                    y=sarimax_data['forecast'],
                    mode='lines',
                    name='SARIMAX Forecast',
                    line=dict(color='#FF6B6B', width=2, dash='dash')
                ))
                
                # Add confidence interval
                if sarimax_data['ci'] is not None:
                    ci = sarimax_data['ci'].values
                    fig.add_trace(go.Scatter(
                        x=test_hours,
                        y=ci[:, 0],
                        fill=None,
                        mode='lines',
                        line_color='rgba(0,0,0,0)',
                        showlegend=False
                    ))
                    fig.add_trace(go.Scatter(
                        x=test_hours,
                        y=ci[:, 1],
                        fill='tonexty',
                        mode='lines',
                        line_color='rgba(0,0,0,0)',
                        name='SARIMAX 95% CI',
                        fillcolor='rgba(255, 107, 107, 0.2)'
                    ))
            
            # Add LSTM forecast
            if f'{days}D_lstm' in results_dict:
                lstm_data = results_dict[f'{days}D_lstm']
                fig.add_trace(go.Scatter(
                    x=test_hours,
                    y=lstm_data['forecast'],
                    mode='lines',
                    name='LSTM Forecast',
                    line=dict(color='#4ECDC4', width=2, dash='dash')
                ))
            
            fig.update_layout(
                title='',
                xaxis_title='Hours Ahead',
                yaxis_title='Load (MW)',
                hovermode='x unified',
                template='plotly_white',
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Metrics comparison
            st.subheader(f"Performance Metrics ({days} Day)")
            
            metrics_data = []
            
            if f'{days}D_sarimax' in results_dict and results_dict[f'{days}D_sarimax']['metrics']:
                sarimax_metrics = results_dict[f'{days}D_sarimax']['metrics']
                metrics_data.append({
                    'Model': 'SARIMAX',
                    'RMSE': f"{sarimax_metrics['RMSE']:.2f}",
                    'MAE': f"{sarimax_metrics['MAE']:.2f}",
                    'MAPE': f"{sarimax_metrics['MAPE']:.2f}%"
                })
            
            if f'{days}D_lstm' in results_dict and results_dict[f'{days}D_lstm']['metrics']:
                lstm_metrics = results_dict[f'{days}D_lstm']['metrics']
                metrics_data.append({
                    'Model': 'LSTM',
                    'RMSE': f"{lstm_metrics['RMSE']:.2f}",
                    'MAE': f"{lstm_metrics['MAE']:.2f}",
                    'MAPE': f"{lstm_metrics['MAPE']:.2f}%"
                })
            
            if metrics_data:
                metrics_df = pd.DataFrame(metrics_data)
                st.dataframe(metrics_df, use_container_width=True)
            
            st.markdown("---")

st.sidebar.markdown("---")
st.sidebar.info(
    """
    **Model Information:**
    - **SARIMAX**: Seasonal ARIMA with eXogenous variables
    - **LSTM**: Long Short-Term Memory neural network
    
    **Parameters:**
    - Lookback window: 24 hours
    - Train ratio: {:.1f}%
    """.format(train_ratio * 100)
)
