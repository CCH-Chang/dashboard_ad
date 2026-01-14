import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# For forecasting
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

# Page config
st.set_page_config(page_title="Energy Weather Analytics Dashboard", layout="wide", initial_sidebar_state="expanded")

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv('processed_energy_weather.csv')
    df['time'] = pd.to_datetime(df['time'])
    return df

df = load_data()

# Sidebar title
st.sidebar.title("📊 Control Panel")

# Date range filter
date_min = df['time'].min()
date_max = df['time'].max()
date_range = st.sidebar.date_input(
    "Select date range",
    value=(date_min.date(), date_max.date()),
    min_value=date_min.date(),
    max_value=date_max.date()
)

# Filter data
if len(date_range) == 2:
    start_date, end_date = date_range
    mask = (df['time'].dt.date >= start_date) & (df['time'].dt.date <= end_date)
    filtered_df = df[mask].copy()
else:
    filtered_df = df.copy()

# Navigation tabs
tab1, tab2, tab3 = st.sidebar.tabs(["Dashboard", "City Temps", "Forecast"])

with tab1:
    st.sidebar.markdown("**Dashboard Settings**")
    show_raw_data = st.sidebar.checkbox("Show raw data", value=False)

with tab2:
    st.sidebar.markdown("**City Temperature Selection**")
    cities_dict = {
        'Barcelona': 'temp_ Barcelona',
        'Bilbao': 'temp_Bilbao',
        'Madrid': 'temp_Madrid',
        'Seville': 'temp_Seville',
        'Valencia': 'temp_Valencia'
    }
    selected_cities = st.sidebar.multiselect(
        "Select cities",
        options=list(cities_dict.keys()),
        default=['Barcelona', 'Madrid']
    )

with tab3:
    st.sidebar.markdown("**Forecast Settings**")
    forecast_days = st.sidebar.multiselect(
        "Forecast horizon (days)",
        options=[1, 3, 7],
        default=[1]
    )
    train_ratio = st.sidebar.slider(
        "Train-Test Split",
        min_value=0.6,
        max_value=0.9,
        value=0.8,
        step=0.05
    )
    run_forecast_btn = st.sidebar.button("🚀 Run Forecast", type="primary")

# Main title
st.title("⚡ Energy Weather Analytics Dashboard")
st.markdown("---")

# Key metrics
col1, col2, col3, col4 = st.columns(4)

with col1:
    avg_load = filtered_df['total load actual'].mean()
    st.metric(
        label="Average Load (MW)",
        value=f"{avg_load:,.0f}",
        delta=f"±{filtered_df['total load actual'].std():,.0f}"
    )

with col2:
    max_load = filtered_df['total load actual'].max()
    st.metric(
        label="Maximum Load (MW)",
        value=f"{max_load:,.0f}"
    )

with col3:
    avg_temp = filtered_df['temp_avg'].mean()
    st.metric(
        label="Average Temperature (K)",
        value=f"{avg_temp:.2f}",
        delta=f"±{filtered_df['temp_avg'].std():.2f}"
    )

with col4:
    data_points = len(filtered_df)
    st.metric(
        label="Data Points",
        value=f"{data_points:,}"
    )

st.markdown("---")

# Row 1: Load trend and Temperature trend
col1, col2 = st.columns(2)

with col1:
    st.subheader("📈 Energy Load Trend")
    fig_load = go.Figure()
    fig_load.add_trace(go.Scatter(
        x=filtered_df['time'],
        y=filtered_df['total load actual'],
        mode='lines',
        name='Total Load',
        line=dict(color='#FF6B6B', width=2),
        fill='tozeroy',
        fillcolor='rgba(255, 107, 107, 0.2)'
    ))
    fig_load.update_layout(
        title='',
        xaxis_title='Time',
        yaxis_title='Load (MW)',
        hovermode='x unified',
        template='plotly_white',
        height=400
    )
    st.plotly_chart(fig_load, use_container_width=True)

with col2:
    st.subheader("🌡️ Average Temperature Trend")
    fig_temp = go.Figure()
    fig_temp.add_trace(go.Scatter(
        x=filtered_df['time'],
        y=filtered_df['temp_avg'],
        mode='lines',
        name='Average Temperature',
        line=dict(color='#4ECDC4', width=2),
        fill='tozeroy',
        fillcolor='rgba(78, 205, 196, 0.2)'
    ))
    fig_temp.update_layout(
        title='',
        xaxis_title='Time',
        yaxis_title='Temperature (K)',
        hovermode='x unified',
        template='plotly_white',
        height=400
    )
    st.plotly_chart(fig_temp, use_container_width=True)

st.markdown("---")

# Row 2: Temperature analysis by city
st.subheader("🏙️ Temperature Analysis by City")

cities = {
    'Barcelona': 'temp_ Barcelona',
    'Bilbao': 'temp_Bilbao',
    'Madrid': 'temp_Madrid',
    'Seville': 'temp_Seville',
    'Valencia': 'temp_Valencia'
}

fig_cities = go.Figure()
for city, col_name in cities.items():
    fig_cities.add_trace(go.Scatter(
        x=filtered_df['time'],
        y=filtered_df[col_name],
        mode='lines',
        name=city,
        line=dict(width=2)
    ))

fig_cities.update_layout(
    title='',
    xaxis_title='Time',
    yaxis_title='Temperature (K)',
    hovermode='x unified',
    template='plotly_white',
    height=400
)
st.plotly_chart(fig_cities, use_container_width=True)

st.markdown("---")

# SECTION: CITY TEMPERATURE SELECTOR
if 'selected_cities' in dir() and selected_cities:
    st.subheader("🏙️ Selected Cities Temperature Trend")
    
    fig_selected = go.Figure()
    for city in selected_cities:
        col_name = cities_dict[city]
        fig_selected.add_trace(go.Scatter(
            x=filtered_df['time'],
            y=filtered_df[col_name],
            mode='lines',
            name=city,
            line=dict(width=2.5)
        ))
    
    fig_selected.update_layout(
        title='',
        xaxis_title='Time',
        yaxis_title='Temperature (K)',
        hovermode='x unified',
        template='plotly_white',
        height=500
    )
    st.plotly_chart(fig_selected, use_container_width=True)
    
    st.markdown("---")

# Row 3: Load and Temperature correlation
st.subheader("📊 Load vs Average Temperature Correlation")

fig_correlation = go.Figure()
fig_correlation.add_trace(go.Scatter(
    x=filtered_df['temp_avg'],
    y=filtered_df['total load actual'],
    mode='markers',
    marker=dict(
        size=5,
        color=filtered_df['temp_avg'],
        colorscale='Viridis',
        showscale=True,
        colorbar=dict(title="Temp (K)")
    ),
    name='Data Points'
))

fig_correlation.update_layout(
    title='',
    xaxis_title='Average Temperature (K)',
    yaxis_title='Load (MW)',
    hovermode='closest',
    template='plotly_white',
    height=400
)
st.plotly_chart(fig_correlation, use_container_width=True)

st.markdown("---")

# Row 4: Analysis by hour and day
col1, col2 = st.columns(2)

with col1:
    st.subheader("⏰ Average Load by Hour")
    hourly_avg = filtered_df.groupby('hour')['total load actual'].mean().reset_index()
    fig_hourly = px.bar(hourly_avg, x='hour', y='total load actual',
                        labels={'hour': 'Hour', 'total load actual': 'Average Load (MW)'},
                        color='total load actual',
                        color_continuous_scale='Blues')
    fig_hourly.update_layout(template='plotly_white', height=400, showlegend=False)
    st.plotly_chart(fig_hourly, use_container_width=True)

with col2:
    st.subheader("📅 Average Load by Day of Week")
    day_names = {0: 'Monday', 1: 'Tuesday', 2: 'Wednesday', 3: 'Thursday', 4: 'Friday', 5: 'Saturday', 6: 'Sunday'}
    daily_avg = filtered_df.groupby('day_of_week')['total load actual'].mean().reset_index()
    daily_avg['day_name'] = daily_avg['day_of_week'].map(day_names)
    fig_daily = px.bar(daily_avg, x='day_name', y='total load actual',
                       labels={'day_name': 'Day', 'total load actual': 'Average Load (MW)'},
                       color='total load actual',
                       color_continuous_scale='Reds')
    fig_daily.update_layout(template='plotly_white', height=400, showlegend=False)
    st.plotly_chart(fig_daily, use_container_width=True)

st.markdown("---")

# Data table
if st.sidebar.checkbox("Show raw data", value=False):
    st.subheader("📋 Raw Data")
    st.dataframe(filtered_df, use_container_width=True)

# Statistics
with st.sidebar.expander("📈 Statistics Summary"):
    st.write(f"**Data Range:** {date_min.date()} to {date_max.date()}")
    st.write(f"**Selected Range:** {len(filtered_df)} records")
    st.write(f"\n**Energy Load Statistics:**")
    st.write(f"- Minimum: {filtered_df['total load actual'].min():,.0f} MW")
    st.write(f"- Average: {filtered_df['total load actual'].mean():,.0f} MW")
    st.write(f"- Maximum: {filtered_df['total load actual'].max():,.0f} MW")
    st.write(f"\n**Temperature Statistics (K):**")
    st.write(f"- Minimum: {filtered_df['temp_avg'].min():.2f}")
    st.write(f"- Average: {filtered_df['temp_avg'].mean():.2f}")
    st.write(f"- Maximum: {filtered_df['temp_avg'].max():.2f}")

# SECTION: FORECASTING
st.markdown("---")
st.markdown("---")
st.title("🔮 Energy Load Forecasting")

if 'run_forecast_btn' in dir() and run_forecast_btn:
    st.info("Running forecasts... Please wait (this may take a few minutes)")
    
    # Prepare data
    data = df['total load actual'].values.astype(float)
    split_point = int(len(data) * train_ratio)
    train_data = data[:split_point]
    test_data = data[split_point:]
    
    for days in sorted(forecast_days):
        steps = days * 24
        
        st.subheader(f"📊 {days}-Day Forecast ({steps} hours)")
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Exponential Smoothing Model**")
            try:
                # Simple exponential smoothing forecast
                alpha = 0.3
                forecast_es = []
                current_val = train_data[-1]
                
                for _ in range(steps):
                    # Weighted average of last value and current prediction
                    next_val = alpha * current_val + (1 - alpha) * np.mean(train_data[-24:])
                    forecast_es.append(next_val)
                    current_val = next_val
                
                forecast_es = np.array(forecast_es)
                
                if len(test_data) >= steps:
                    rmse_es = np.sqrt(mean_squared_error(test_data[:steps], forecast_es))
                    mae_es = mean_absolute_error(test_data[:steps], forecast_es)
                    st.metric("RMSE", f"{rmse_es:.2f}")
                    st.metric("MAE", f"{mae_es:.2f}")
                st.success("✅ Exponential Smoothing completed")
            except Exception as e:
                st.error(f"❌ Error: {str(e)[:100]}")
                forecast_es = None
        
        with col2:
            st.markdown("**LSTM Model**")
            try:
                # Scale data
                scaler = MinMaxScaler(feature_range=(0, 1))
                train_scaled = scaler.fit_transform(train_data.reshape(-1, 1))
                
                # Create sequences
                lookback = 24
                X_train, y_train = [], []
                for i in range(len(train_scaled) - lookback):
                    X_train.append(train_scaled[i:(i + lookback)])
                    y_train.append(train_scaled[i + lookback])
                X_train, y_train = np.array(X_train), np.array(y_train)
                
                # Build model
                lstm_model = Sequential([
                    LSTM(64, activation='relu', input_shape=(lookback, 1), return_sequences=True),
                    Dropout(0.2),
                    LSTM(32, activation='relu', return_sequences=False),
                    Dropout(0.2),
                    Dense(16, activation='relu'),
                    Dense(1)
                ])
                lstm_model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
                
                early_stop = EarlyStopping(monitor='loss', patience=5, restore_best_weights=True)
                lstm_model.fit(X_train, y_train, epochs=50, batch_size=32, callbacks=[early_stop], verbose=0)
                
                # Forecast
                last_seq = train_scaled[-lookback:].copy()
                lstm_forecast = []
                current_seq = last_seq.copy()
                
                for _ in range(steps):
                    pred = lstm_model.predict(current_seq.reshape(1, lookback, 1), verbose=0)[0, 0]
                    lstm_forecast.append(pred)
                    current_seq = np.append(current_seq[1:], [[pred]], axis=0)
                
                lstm_forecast = scaler.inverse_transform(np.array(lstm_forecast).reshape(-1, 1)).flatten()
                
                if len(test_data) >= steps:
                    rmse_lstm = np.sqrt(mean_squared_error(test_data[:steps], lstm_forecast))
                    mae_lstm = mean_absolute_error(test_data[:steps], lstm_forecast)
                    st.metric("RMSE", f"{rmse_lstm:.2f}")
                    st.metric("MAE", f"{mae_lstm:.2f}")
                st.success("✅ LSTM completed")
            except Exception as e:
                st.error(f"❌ LSTM error: {str(e)[:100]}")
                lstm_forecast = None
        
        # Plot comparison
        st.markdown("---")
        fig_forecast = go.Figure()
        
        hours_range = np.arange(steps)
        fig_forecast.add_trace(go.Scatter(
            x=hours_range,
            y=test_data[:steps],
            mode='lines',
            name='Actual Test Data',
            line=dict(color='black', width=2)
        ))
        
        if 'forecast_es' in dir() and forecast_es is not None:
            fig_forecast.add_trace(go.Scatter(
                x=hours_range,
                y=forecast_es,
                mode='lines',
                name='Exponential Smoothing',
                line=dict(color='#FF6B6B', width=2, dash='dash')
            ))
        
        if 'lstm_forecast' in dir() and lstm_forecast is not None:
            fig_forecast.add_trace(go.Scatter(
                x=hours_range,
                y=lstm_forecast,
                mode='lines',
                name='LSTM Forecast',
                line=dict(color='#4ECDC4', width=2, dash='dash')
            ))
        
        fig_forecast.update_layout(
            title='',
            xaxis_title='Hours Ahead',
            yaxis_title='Load (MW)',
            hovermode='x unified',
            template='plotly_white',
            height=400
        )
        st.plotly_chart(fig_forecast, use_container_width=True)
        st.markdown("---")
    
    st.success("🎉 All forecasts completed!")
else:
    st.info("👈 Configure forecast settings in the sidebar and click 'Run Forecast' to start")

