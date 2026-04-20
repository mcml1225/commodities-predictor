import pandas as pd
import numpy as np

def load_and_clean_data(file_path):
    # Cargar el archivo
    xl = pd.ExcelFile(file_path)
    df_raw = xl.parse('Mensual', header=None)
    
    # Identificar fila de tickers (fila 3 en 0-index)
    tickers_row = df_raw.iloc[3, 1:].dropna().values
    # Identificar fila de campos (fila 4)
    fields_row = df_raw.iloc[4, 1:].dropna().values
    
    # Crear estructura de columnas: ticker_campo
    columns = []
    for ticker, field in zip(tickers_row, fields_row):
        if isinstance(field, str):
            columns.append(f"{ticker}_{field}")
        else:
            columns.append(str(ticker))
    
    # Datos empiezan en fila 6
    data_rows = df_raw.iloc[6:, :].copy()
    data_rows.columns = ['Dates'] + columns[:len(data_rows.columns)-1]
    
    # Convertir fechas
    data_rows['Dates'] = pd.to_datetime(data_rows['Dates'], errors='coerce')
    data_rows = data_rows.dropna(subset=['Dates'])
    data_rows = data_rows.set_index('Dates')
    
    # Reemplazar '#N/A N/A' y cadenas similares por NaN
    data_rows = data_rows.replace(['#N/A N/A', '#N/A', 'N/A'], np.nan)
    
    # Convertir a numérico
    data_rows = data_rows.apply(pd.to_numeric, errors='coerce')
    
    # Eliminar columnas completamente vacías
    data_rows = data_rows.dropna(axis=1, how='all')
    
    # Crear target (precio del mes siguiente)
    # Buscamos columnas que contengan 'PX_LAST'
    px_last_cols = [col for col in data_rows.columns if 'PX_LAST' in col]
    
    if not px_last_cols:
        raise ValueError("No se encontraron columnas con PX_LAST")
    
    # Usar la primera columna PX_LAST como ejemplo (puedes ajustar)
    target_col = px_last_cols[0]
    
    # Crear target: precio del mes siguiente
    data_rows['target'] = data_rows[target_col].shift(-1)
    
    # Eliminar últimos NaN del target
    data_rows = data_rows.dropna(subset=['target'])
    
    # Eliminar columnas que no sirven (ej. MATURITY, CUR_MKT_CAP si son constantes)
    useless_patterns = ['MATURITY', 'AMT_OUTSTANDING', 'CUR_MKT_CAP']
    for pattern in useless_patterns:
        cols_to_drop = [col for col in data_rows.columns if pattern in col]
        data_rows = data_rows.drop(columns=cols_to_drop, errors='ignore')
    
    return data_rows, target_col

if __name__ == "__main__":
    df, target = load_and_clean_data("Ejericio Bloomberg Commodities_revisar.xlsx")
    print(df.head())
    print(f"\nForma final: {df.shape}")
    print(f"Columnas: {df.columns.tolist()}")