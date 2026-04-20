import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from data_loader import load_and_clean_data
from model import train_model, predict_next
import joblib
import os

st.set_page_config(page_title="Commodities Predictor", layout="wide")

st.title("📈 Predicción de Precios de Commodities")
st.markdown("Modelo predictivo basado en datos históricos de Bloomberg")

# Cargar datos
@st.cache_data
def load_data():
    return load_and_clean_data("Ejericio Bloomberg Commodities_revisar.xlsx")

try:
    df, target_col = load_data()
    st.success(f"Datos cargados correctamente: {df.shape[0]} registros, {df.shape[1]-1} features")
    
    # Mostrar muestra
    with st.expander("Ver datos de ejemplo"):
        st.dataframe(df.head(10))
    
    # Entrenar o cargar modelo
    if not os.path.exists("model.pkl"):
        st.info("Entrenando modelo por primera vez...")
        model, scaler, imputer, metrics = train_model(df, 'target')
        st.success(f"Modelo entrenado - R²: {metrics[2]:.4f}")
    else:
        model = joblib.load("model.pkl")
        scaler = joblib.load("scaler.pkl")
        imputer = joblib.load("imputer.pkl")
        features = joblib.load("features.pkl")
        st.info("Modelo cargado desde disco")
    
    # Predicción siguiente mes
    if st.button("🔮 Predecir próximo mes"):
        pred = predict_next(df, model, scaler, imputer, features)
        st.metric("Predicción de precio", f"{pred:.2f} USD")
    
    # Gráfico histórico
    st.subheader("📊 Evolución histórica del precio")
    fig = px.line(df.reset_index(), x='Dates', y=target_col, 
                  title=f"Precio histórico - {target_col}",
                  labels={'Dates': 'Fecha', target_col: 'Precio'})
    st.plotly_chart(fig, use_container_width=True)
    
    # Distribución de errores (si modelo está entrenado)
    st.subheader("🎯 Calidad del modelo")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("MAE", f"{metrics[0]:.2f}" if 'metrics' in locals() else "N/A")
    with col2:
        st.metric("RMSE", f"{metrics[1]:.2f}" if 'metrics' in locals() else "N/A")
    with col3:
        st.metric("R²", f"{metrics[2]:.3f}" if 'metrics' in locals() else "N/A")
    
except Exception as e:
    st.error(f"Error al procesar los datos: {e}")
    st.info("Verifica que el archivo Excel esté en la misma carpeta que app.py")