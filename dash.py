import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import warnings
warnings.filterwarnings('ignore')

# Page config
st.set_page_config(
    page_title="Energy Weather Analytics Dashboard", 
    layout="wide", 
    initial_sidebar_state="expanded"
)

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv('processed_energy_weather.csv')
    df['time'] = pd.to_datetime(df['time'])
    return df

try:
    df = load_data()
except FileNotFoundError:
    st.error("Error: 'processed_energy_weather.csv' not found!")
    st.stop()

# Sidebar
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

# Main title
st.title("⚡ Energy Weather Analytics Dashboard")
st.markdown("---")

# Key metrics
st.subheader("📊 Key Metrics")
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Average Load (MW)", f"{filtered_df['total load actual'].mean():,.0f}")

with col2:
    st.metric("Maximum Load (MW)", f"{filtered_df['total load actual'].max():,.0f}")

with col3:
    st.metric("Minimum Load (MW)", f"{filtered_df['total load actual'].min():,.0f}")

with col4:
    st.metric("Data Points", f"{len(filtered_df):,}")

st.markdown("---")

# Section 1: Time Series Charts
st.subheader("📈 Time Series Analysis")

col1, col2 = st.columns(2)

with col1:
    st.markdown("**Energy Load Trend**")
    fig_load = go.Figure()
    fig_load.add_trace(go.Scatter(
        x=filtered_df['time'],
        y=filtered_df['total load actual'],
        mode='lines',
        name='Total Load',
        line=dict(color='#FF6B6B', width=2),
        fill='tozeroy'
    ))
    fig_load.update_layout(
        xaxis_title='Time',
        yaxis_title='Load (MW)',
        hovermode='x unified',
        template='plotly_white',
        height=400
    )
    st.plotly_chart(fig_load, use_container_width=True)

with col2:
    st.markdown("**Average Temperature Trend**")
    fig_temp = go.Figure()
    fig_temp.add_trace(go.Scatter(
        x=filtered_df['time'],
        y=filtered_df['temp_avg'],
        mode='lines',
        name='Average Temperature',
        line=dict(color='#4ECDC4', width=2),
        fill='tozeroy'
    ))
    fig_temp.update_layout(
        xaxis_title='Time',
        yaxis_title='Temperature (K)',
        hovermode='x unified',
        template='plotly_white',
        height=400
    )
    st.plotly_chart(fig_temp, use_container_width=True)

st.markdown("---")

# Section 2: City Temperature Selection
st.subheader("🏙️ City Temperature Analysis")

cities = {
    'Barcelona': 'temp_ Barcelona',
    'Bilbao': 'temp_Bilbao',
    'Madrid': 'temp_Madrid',
    'Seville': 'temp_Seville',
    'Valencia': 'temp_Valencia'
}

# City selector in sidebar
st.sidebar.markdown("**City Selection**")
selected_cities = st.sidebar.multiselect(
    "Select cities to display",
    options=list(cities.keys()),
    default=['Barcelona', 'Madrid']
)

if selected_cities:
    fig_cities = go.Figure()
    for city in selected_cities:
        col_name = cities[city]
        fig_cities.add_trace(go.Scatter(
            x=filtered_df['time'],
            y=filtered_df[col_name],
            mode='lines',
            name=city,
            line=dict(width=2)
        ))
    
    fig_cities.update_layout(
        xaxis_title='Time',
        yaxis_title='Temperature (K)',
        hovermode='x unified',
        template='plotly_white',
        height=400
    )
    st.plotly_chart(fig_cities, use_container_width=True)

st.markdown("---")

# Section 3: Correlation Analysis
st.subheader("📊 Load vs Temperature Correlation")

fig_correlation = go.Figure()
fig_correlation.add_trace(go.Scatter(
    x=filtered_df['temp_avg'],
    y=filtered_df['total load actual'],
    mode='markers',
    marker=dict(
        size=4,
        color=filtered_df['temp_avg'],
        colorscale='Viridis',
        showscale=True
    ),
    name='Load vs Temp'
))

fig_correlation.update_layout(
    xaxis_title='Average Temperature (K)',
    yaxis_title='Load (MW)',
    hovermode='closest',
    template='plotly_white',
    height=400
)
st.plotly_chart(fig_correlation, use_container_width=True)

st.markdown("---")

# Section 4: Hourly and Daily Analysis
st.subheader("📅 Hourly & Daily Patterns")

col1, col2 = st.columns(2)

with col1:
    st.markdown("**Average Load by Hour**")
    hourly_avg = filtered_df.groupby('hour')['total load actual'].mean().reset_index()
    fig_hourly = px.bar(
        hourly_avg, 
        x='hour', 
        y='total load actual',
        labels={'hour': 'Hour', 'total load actual': 'Average Load (MW)'},
        color='total load actual',
        color_continuous_scale='Blues'
    )
    fig_hourly.update_layout(template='plotly_white', height=400, showlegend=False)
    st.plotly_chart(fig_hourly, use_container_width=True)

with col2:
    st.markdown("**Average Load by Day of Week**")
    day_names = {0: 'Mon', 1: 'Tue', 2: 'Wed', 3: 'Thu', 4: 'Fri', 5: 'Sat', 6: 'Sun'}
    daily_avg = filtered_df.groupby('day_of_week')['total load actual'].mean().reset_index()
    daily_avg['day_name'] = daily_avg['day_of_week'].map(day_names)
    fig_daily = px.bar(
        daily_avg, 
        x='day_name', 
        y='total load actual',
        labels={'day_name': 'Day', 'total load actual': 'Average Load (MW)'},
        color='total load actual',
        color_continuous_scale='Reds'
    )
    fig_daily.update_layout(template='plotly_white', height=400, showlegend=False)
    st.plotly_chart(fig_daily, use_container_width=True)

st.markdown("---")

# Section 5: Data Table
st.subheader("📋 Data Table")

show_data = st.sidebar.checkbox("Show raw data", value=False)

if show_data:
    st.dataframe(filtered_df, use_container_width=True)
    
    # Download filtered data
    csv_data = filtered_df.to_csv(index=False)
    st.download_button(
        label="📥 Download Filtered Data (CSV)",
        data=csv_data,
        file_name=f"energy_weather_data_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv",
        help="Download the filtered dataset as CSV"
    )

# Statistics Summary
st.sidebar.markdown("---")
with st.sidebar.expander("📈 Statistics Summary"):
    st.write(f"**Date Range:** {date_min.date()} to {date_max.date()}")
    st.write(f"**Selected Records:** {len(filtered_df)}")
    st.write(f"\n**Energy Load Statistics:**")
    st.write(f"- Min: {filtered_df['total load actual'].min():,.0f} MW")
    st.write(f"- Avg: {filtered_df['total load actual'].mean():,.0f} MW")
    st.write(f"- Max: {filtered_df['total load actual'].max():,.0f} MW")
    st.write(f"\n**Temperature Statistics:**")
    st.write(f"- Min: {filtered_df['temp_avg'].min():.2f} K")
    st.write(f"- Avg: {filtered_df['temp_avg'].mean():.2f} K")
    st.write(f"- Max: {filtered_df['temp_avg'].max():.2f} K")

st.markdown("---")

st.markdown("---")

# Section 6: Simple Forecasting with User Interaction
st.subheader("🔮 Simple Forecasting")
st.write("Forecast the next 24-48 hours using basic methods")

# Forecast parameters in sidebar
st.sidebar.markdown("---")
st.sidebar.markdown("**⚙️ Forecast Settings**")

forecast_hours = st.sidebar.slider(
    "Hours to forecast",
    min_value=12,
    max_value=168,
    value=24,
    step=12,
    help="Select how many hours ahead to forecast"
)

ma_window = st.sidebar.slider(
    "Moving Average Window (hours)",
    min_value=6,
    max_value=72,
    value=24,
    step=6,
    help="Number of hours to use for moving average calculation"
)

smoothing_alpha = st.sidebar.slider(
    "Exponential Smoothing Factor (α)",
    min_value=0.1,
    max_value=0.9,
    value=0.3,
    step=0.1,
    help="Higher values give more weight to recent data (0.1=smooth, 0.9=reactive)"
)

# Get recent data
recent_data = filtered_df['total load actual'].tail(max(ma_window + 24, 120)).values

# Calculate Moving Average Forecast
ma_forecast = []
for i in range(forecast_hours):
    avg = np.mean(recent_data[-ma_window:])
    ma_forecast.append(avg)
ma_forecast = np.array(ma_forecast)

# Calculate Exponential Smoothing Forecast
es_forecast = []
current_val = recent_data[-1]
baseline = np.mean(recent_data[-24:])

for i in range(forecast_hours):
    next_val = smoothing_alpha * current_val + (1 - smoothing_alpha) * baseline
    es_forecast.append(next_val)
    current_val = next_val
es_forecast = np.array(es_forecast)

# Forecast comparison
col1, col2 = st.columns(2)

with col1:
    st.markdown("**Moving Average Method**")
    
    fig_ma = go.Figure()
    fig_ma.add_trace(go.Scatter(
        x=np.arange(len(recent_data)),
        y=recent_data,
        mode='lines',
        name='Historical',
        line=dict(color='#FF6B6B', width=2)
    ))
    fig_ma.add_trace(go.Scatter(
        x=np.arange(len(recent_data), len(recent_data) + forecast_hours),
        y=ma_forecast,
        mode='lines+markers',
        name='Forecast',
        line=dict(color='#4ECDC4', width=2, dash='dash'),
        marker=dict(size=5)
    ))
    fig_ma.update_layout(
        xaxis_title='Hours',
        yaxis_title='Load (MW)',
        template='plotly_white',
        height=350,
        hovermode='x unified'
    )
    st.plotly_chart(fig_ma, use_container_width=True)
    
    # Metrics for MA
    col_ma1, col_ma2, col_ma3 = st.columns(3)
    with col_ma1:
        st.metric("Min Forecast", f"{ma_forecast.min():,.0f} MW")
    with col_ma2:
        st.metric("Avg Forecast", f"{ma_forecast.mean():,.0f} MW")
    with col_ma3:
        st.metric("Max Forecast", f"{ma_forecast.max():,.0f} MW")

with col2:
    st.markdown("**Exponential Smoothing Method**")
    
    fig_es = go.Figure()
    fig_es.add_trace(go.Scatter(
        x=np.arange(len(recent_data)),
        y=recent_data,
        mode='lines',
        name='Historical',
        line=dict(color='#FF6B6B', width=2)
    ))
    fig_es.add_trace(go.Scatter(
        x=np.arange(len(recent_data), len(recent_data) + forecast_hours),
        y=es_forecast,
        mode='lines+markers',
        name='Forecast',
        line=dict(color='#95E1D3', width=2, dash='dash'),
        marker=dict(size=5)
    ))
    fig_es.update_layout(
        xaxis_title='Hours',
        yaxis_title='Load (MW)',
        template='plotly_white',
        height=350,
        hovermode='x unified'
    )
    st.plotly_chart(fig_es, use_container_width=True)
    
    # Metrics for ES
    col_es1, col_es2, col_es3 = st.columns(3)
    with col_es1:
        st.metric("Min Forecast", f"{es_forecast.min():,.0f} MW")
    with col_es2:
        st.metric("Avg Forecast", f"{es_forecast.mean():,.0f} MW")
    with col_es3:
        st.metric("Max Forecast", f"{es_forecast.max():,.0f} MW")

# Forecast Comparison
st.markdown("---")
st.subheader("📊 Forecast Methods Comparison")

fig_compare = go.Figure()
fig_compare.add_trace(go.Scatter(
    x=np.arange(len(recent_data)),
    y=recent_data,
    mode='lines',
    name='Historical Data',
    line=dict(color='#333333', width=2.5)
))
fig_compare.add_trace(go.Scatter(
    x=np.arange(len(recent_data), len(recent_data) + forecast_hours),
    y=ma_forecast,
    mode='lines+markers',
    name='Moving Average',
    line=dict(color='#4ECDC4', width=2, dash='dash'),
    marker=dict(size=4)
))
fig_compare.add_trace(go.Scatter(
    x=np.arange(len(recent_data), len(recent_data) + forecast_hours),
    y=es_forecast,
    mode='lines+markers',
    name='Exponential Smoothing',
    line=dict(color='#95E1D3', width=2, dash='dot'),
    marker=dict(size=4)
))

fig_compare.update_layout(
    xaxis_title='Hours',
    yaxis_title='Load (MW)',
    template='plotly_white',
    height=400,
    hovermode='x unified'
)
st.plotly_chart(fig_compare, use_container_width=True)

# Forecast difference analysis
st.subheader("📈 Forecast Difference Analysis")

difference = np.abs(ma_forecast - es_forecast)
col_diff1, col_diff2, col_diff3, col_diff4 = st.columns(4)

with col_diff1:
    st.metric("Avg Difference", f"{difference.mean():,.0f} MW")
with col_diff2:
    st.metric("Max Difference", f"{difference.max():,.0f} MW")
with col_diff3:
    st.metric("Min Difference", f"{difference.min():,.0f} MW")
with col_diff4:
    st.metric("Std Difference", f"{difference.std():,.0f} MW")

# Download forecast data
st.markdown("---")
st.subheader("💾 Download Forecast Results")

# Create forecast dataframe
forecast_df = pd.DataFrame({
    'Hour_Ahead': np.arange(1, forecast_hours + 1),
    'Moving_Average_MW': ma_forecast,
    'Exponential_Smoothing_MW': es_forecast,
    'Difference_MW': np.abs(ma_forecast - es_forecast)
})

# Display table
st.dataframe(forecast_df, use_container_width=True)

# Download buttons
col_csv, col_info = st.columns([3, 1])

with col_csv:
    csv = forecast_df.to_csv(index=False)
    st.download_button(
        label="📥 Download Forecast (CSV)",
        data=csv,
        file_name=f"forecast_{forecast_hours}h_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv",
        help="Download forecast data as CSV file"
    )

with col_info:
    st.info(f"📊 {len(forecast_df)} rows")

# Historical comparison
st.markdown("---")
st.subheader("🔄 Forecast Accuracy Check")
st.write("Compare with actual recent data to see model performance")

if len(filtered_df) >= forecast_hours:
    # Get actual next hours if available
    actual_future = filtered_df['total load actual'].tail(forecast_hours).values
    
    accuracy_df = pd.DataFrame({
        'Hour': np.arange(1, forecast_hours + 1),
        'Actual': actual_future,
        'MA_Forecast': ma_forecast,
        'ES_Forecast': es_forecast,
        'MA_Error': np.abs(actual_future - ma_forecast),
        'ES_Error': np.abs(actual_future - es_forecast)
    })
    
    st.dataframe(accuracy_df, use_container_width=True)
    
    col_acc1, col_acc2 = st.columns(2)
    with col_acc1:
        st.metric("MA Mean Absolute Error", f"{accuracy_df['MA_Error'].mean():,.0f} MW")
    with col_acc2:
        st.metric("ES Mean Absolute Error", f"{accuracy_df['ES_Error'].mean():,.0f} MW")
else:
    st.info("✋ Not enough historical data for accuracy comparison")


