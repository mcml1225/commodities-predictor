import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import joblib
import os

from data_loader import load_and_clean_data
from model import train_model, predict_with_existing_model

st.set_page_config(page_title="Commodities Price Predictor", layout="wide")

st.title("📈 Commodities Price Predictor")
st.markdown("Predictive model based on historical Bloomberg data")

# Load data
@st.cache_data
def load_data():
    return load_and_clean_data("Bloomberg_commodities_prices.xlsx")

try:
    df, target_col = load_data()
    
    st.success(f"✅ Data loaded successfully")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Records", len(df))
    with col2:
        st.metric("Features", len(df.columns) - 1)
    with col3:
        st.metric("Date range", f"{df.index.min().strftime('%Y-%m')} to {df.index.max().strftime('%Y-%m')}")
    
    # Show data
    with st.expander("📋 View processed data"):
        st.dataframe(df.tail(20))
    
    # Historical price chart
    st.subheader("📊 Historical price evolution")
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df[target_col],
        mode='lines+markers',
        name=target_col,
        line=dict(color='blue', width=2),
        marker=dict(size=4)
    ))
    fig.update_layout(
        title=f"Historical price - {target_col}",
        xaxis_title="Date",
        yaxis_title="Price (USD)",
        hovermode='x unified'
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Train or load model
    if st.button("🚀 Train model and predict"):
        with st.spinner("Training models for 1, 2, and 3 month predictions..."):
            models, scalers, imputers, metrics, feature_importance, df_multi = train_model(df, target_col, steps=3)
            
        st.success("✅ Models trained successfully!")
        
        # Show metrics for each horizon
        st.subheader("🎯 Model Performance by Horizon")
        
        metrics_data = []
        for step in [1, 2, 3]:
            metrics_data.append({
                'Horizon': f'{step} month{"s" if step > 1 else ""}',
                'Test R²': metrics[step]['test']['r2'],
                'Test MAE': metrics[step]['test']['mae'],
                'Test RMSE': metrics[step]['test']['rmse']
            })
        
        metrics_df = pd.DataFrame(metrics_data)
        st.dataframe(metrics_df, use_container_width=True)
        
        # Show test metrics chart
        fig_test = px.bar(metrics_df, x='Horizon', y='Test R²', 
                         title='Test R² by Horizon',
                         color='Test R²', range_y=[0, 1])
        st.plotly_chart(fig_test, use_container_width=True)
        
        # Load feature names for prediction
        feature_names = joblib.load('features.pkl')
        
        # Make predictions
        from model import predict_future
        predictions, future_dates, last_date = predict_future(
            df, models, scalers, imputers, feature_names, target_col, steps=3
        )
        
        st.subheader("🔮 Price Predictions")
        
        pred_cols = st.columns(3)
        for i, step in enumerate([1, 2, 3]):
            delta = predictions[step] - df[target_col].iloc[-1]
            color = "inverse" if delta < 0 else "normal"
            with pred_cols[i]:
                st.metric(
                    f"Prediction for {future_dates[i].strftime('%B %Y')}", 
                    f"{predictions[step]:.2f} USD",
                    delta=f"{delta:.2f}",
                    delta_color=color
                )
        
        # Show chart with predictions
        st.subheader("📈 Historical Data + Future Predictions")
        
        fig_pred = go.Figure()
        
        # Historical data (last 24 months)
        fig_pred.add_trace(go.Scatter(
            x=df.index[-24:],
            y=df[target_col].iloc[-24:],
            mode='lines+markers',
            name='Historical',
            line=dict(color='blue', width=2),
            marker=dict(size=6)
        ))
        
        # Predictions
        pred_dates = [last_date] + future_dates
        pred_values = [df[target_col].iloc[-1]] + [predictions[step] for step in [1, 2, 3]]
        
        fig_pred.add_trace(go.Scatter(
            x=pred_dates,
            y=pred_values,
            mode='lines+markers',
            name='Prediction',
            line=dict(color='red', width=2, dash='dash'),
            marker=dict(color='red', size=8)
        ))
        
        fig_pred.update_layout(
            title="Price Projection - Next 3 Months",
            xaxis_title="Date",
            yaxis_title="Price (USD)",
            hovermode='x unified'
        )
        st.plotly_chart(fig_pred, use_container_width=True)
        
        # Feature importance
        st.subheader("📈 Most Important Features (1-month model)")
        fig_imp = px.bar(feature_importance.head(15), 
                         x='importance', y='feature',
                         orientation='h',
                         title="Top 15 features influencing the prediction")
        fig_imp.update_layout(yaxis={'categoryorder': 'total ascending'}, height=500)
        st.plotly_chart(fig_imp, use_container_width=True)
        
        # Store model state
        st.session_state['model_trained'] = True
        st.session_state['metrics'] = metrics
        
    # If model already exists, show quick prediction
    elif os.path.exists('models.pkl'):
        if st.button("🔮 Predict next 3 months (existing model)"):
            predictions, future_dates, last_date, metrics = predict_with_existing_model(df, target_col, steps=3)
            
            st.subheader("🔮 Price Predictions")
            
            pred_cols = st.columns(3)
            for i, step in enumerate([1, 2, 3]):
                delta = predictions[step] - df[target_col].iloc[-1]
                color = "inverse" if delta < 0 else "normal"
                with pred_cols[i]:
                    st.metric(
                        f"Prediction for {future_dates[i].strftime('%B %Y')}", 
                        f"{predictions[step]:.2f} USD",
                        delta=f"{delta:.2f}",
                        delta_color=color
                    )
            
            # Show chart with predictions
            fig_pred = go.Figure()
            
            # Historical data (last 24 months)
            fig_pred.add_trace(go.Scatter(
                x=df.index[-24:],
                y=df[target_col].iloc[-24:],
                mode='lines+markers',
                name='Historical',
                line=dict(color='blue', width=2)
            ))
            
            # Predictions
            pred_dates = [last_date] + future_dates
            pred_values = [df[target_col].iloc[-1]] + [predictions[step] for step in [1, 2, 3]]
            
            fig_pred.add_trace(go.Scatter(
                x=pred_dates,
                y=pred_values,
                mode='lines+markers',
                name='Prediction',
                line=dict(color='red', width=2, dash='dash'),
                marker=dict(color='red', size=8)
            ))
            
            fig_pred.update_layout(title="Price Projection - Next 3 Months", xaxis_title="Date", yaxis_title="Price")
            st.plotly_chart(fig_pred, use_container_width=True)
            
            # Show metrics
            with st.expander("📊 View Model Metrics"):
                metrics_data = []
                for step in [1, 2, 3]:
                    metrics_data.append({
                        'Horizon': f'{step} month(s)',
                        'Test R²': metrics[step]['test']['r2'],
                        'Test MAE': metrics[step]['test']['mae'],
                        'Test RMSE': metrics[step]['test']['rmse']
                    })
                st.dataframe(pd.DataFrame(metrics_data))
            
except Exception as e:
    st.error(f"Error processing data: {e}")
    import traceback
    st.code(traceback.format_exc())
    st.info("Make sure 'Bloomberg_commodities_prices.xlsx' is in the same folder")