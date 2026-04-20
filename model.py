import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import joblib

def train_model(df, target_col):
    # Separar features y target
    X = df.drop(columns=['target'])
    y = df['target']
    
    # Eliminar columnas con muchos NaN (>30%)
    nan_thresh = 0.3
    X = X.loc[:, X.isnull().mean() < nan_thresh]
    
    # Imputar NaN con la mediana
    from sklearn.impute import SimpleImputer
    imputer = SimpleImputer(strategy='median')
    X_imputed = imputer.fit_transform(X)
    
    # Escalar
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imputed)
    
    # Dividir
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    
    # Entrenar modelo
    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    
    # Evaluar
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    print(f"📊 Evaluación del modelo:")
    print(f"MAE: {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R²: {r2:.4f}")
    
    # Guardar modelo y preprocesadores
    joblib.dump(model, 'model.pkl')
    joblib.dump(scaler, 'scaler.pkl')
    joblib.dump(imputer, 'imputer.pkl')
    joblib.dump(X.columns.tolist(), 'features.pkl')
    
    return model, scaler, imputer, (mae, rmse, r2)

def predict_next(df, model, scaler, imputer, feature_names):
    # Usar última fila para predecir siguiente mes
    last_row = df.drop(columns=['target']).iloc[-1:].copy()
    
    # Alinear columnas
    for col in feature_names:
        if col not in last_row.columns:
            last_row[col] = np.nan
    last_row = last_row[feature_names]
    
    # Imputar y escalar
    last_imputed = imputer.transform(last_row)
    last_scaled = scaler.transform(last_imputed)
    
    pred = model.predict(last_scaled)[0]
    return pred