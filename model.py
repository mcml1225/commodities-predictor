import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import joblib
import warnings
warnings.filterwarnings('ignore')

def create_features(df, target_col):
    """Create derived features for better prediction"""
    df = df.copy()
    
    # Price lags
    for lag in [1, 3, 6]:
        df[f'lag_{lag}m'] = df[target_col].shift(lag)
    
    # Rolling means
    for window in [3, 6]:
        df[f'rolling_mean_{window}m'] = df[target_col].rolling(window=window).mean()
        df[f'rolling_std_{window}m'] = df[target_col].rolling(window=window).std()
    
    # Price changes
    df['price_change_1m'] = df[target_col].pct_change(1)
    df['price_change_3m'] = df[target_col].pct_change(3)
    
    # Drop NaN rows
    df = df.dropna()
    
    return df

def train_test_split_chronological(df, train_ratio=0.7):
    """Split data chronologically"""
    n = len(df)
    train_end = int(n * train_ratio)
    
    train_df = df.iloc[:train_end]
    test_df = df.iloc[train_end:]
    
    return train_df, test_df

def train_model(df, target_col, steps=3):
    """Train models for multi-step prediction"""
    print("Creating features...")
    df_enhanced = create_features(df, target_col)
    
    # Create targets for each step
    for step in range(1, steps + 1):
        df_enhanced[f'target_{step}m'] = df_enhanced[target_col].shift(-step)
    
    # Drop rows with NaN targets
    target_cols = [f'target_{step}m' for step in range(1, steps + 1)]
    df_enhanced = df_enhanced.dropna(subset=target_cols)
    
    # Split data
    train_df, test_df = train_test_split_chronological(df_enhanced)
    
    print(f"Train samples: {len(train_df)}, Test samples: {len(test_df)}")
    
    # Features (exclude target columns)
    feature_cols = [col for col in train_df.columns if col not in target_cols and col != target_col]
    
    models = {}
    scalers = {}
    imputers = {}
    metrics = {}
    
    for step in range(1, steps + 1):
        target = f'target_{step}m'
        print(f"\nTraining model for {step} month(s) ahead...")
        
        X_train = train_df[feature_cols]
        y_train = train_df[target]
        X_test = test_df[feature_cols]
        y_test = test_df[target]
        
        # Remove columns with too many NaN
        valid_cols = X_train.columns[X_train.isnull().mean() < 0.4].tolist()
        X_train = X_train[valid_cols]
        X_test = X_test[valid_cols]
        
        # Impute NaN
        imputer = SimpleImputer(strategy='median')
        X_train_imp = imputer.fit_transform(X_train)
        X_test_imp = imputer.transform(X_test)
        
        # Scale
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_imp)
        X_test_scaled = scaler.transform(X_test_imp)
        
        # Train model
        model = RandomForestRegressor(n_estimators=100, max_depth=6, random_state=42)
        model.fit(X_train_scaled, y_train)
        
        # Evaluate
        y_pred = model.predict(X_test_scaled)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        
        print(f"  Test MAE: {mae:.4f}, R²: {r2:.4f}")
        
        models[step] = model
        scalers[step] = scaler
        imputers[step] = imputer
        metrics[step] = {'test': {'mae': mae, 'rmse': rmse, 'r2': r2}}
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': valid_cols,
        'importance': models[1].feature_importances_
    }).sort_values('importance', ascending=False)
    
    # Save models
    joblib.dump(models, 'models.pkl')
    joblib.dump(scalers, 'scalers.pkl')
    joblib.dump(imputers, 'imputers.pkl')
    joblib.dump(valid_cols, 'features.pkl')
    joblib.dump(feature_importance, 'feature_importance.pkl')
    joblib.dump(metrics, 'metrics.pkl')
    
    return models, scalers, imputers, metrics, feature_importance, df_enhanced

def predict_future(df, models, scalers, imputers, feature_names, target_col, steps=3):
    """Predict next months"""
    df_enhanced = create_features(df, target_col)
    last_row = df_enhanced.iloc[-1:].copy()
    last_date = df.index[-1]
    
    predictions = {}
    future_dates = []
    
    for step in range(1, steps + 1):
        next_date = last_date + pd.DateOffset(months=step)
        future_dates.append(next_date)
        
        # Prepare features
        pred_row = last_row[feature_names].copy()
        
        # Impute and scale
        pred_imputed = imputers[step].transform(pred_row)
        pred_scaled = scalers[step].transform(pred_imputed)
        
        # Predict
        pred = models[step].predict(pred_scaled)[0]
        predictions[step] = pred
    
    return predictions, future_dates, last_date

def predict_with_existing_model(df, target_col, steps=3):
    """Load existing models and predict"""
    models = joblib.load('models.pkl')
    scalers = joblib.load('scalers.pkl')
    imputers = joblib.load('imputers.pkl')
    feature_names = joblib.load('features.pkl')
    metrics = joblib.load('metrics.pkl')
    
    predictions, future_dates, last_date = predict_future(
        df, models, scalers, imputers, feature_names, target_col, steps
    )
    
    return predictions, future_dates, last_date, metrics