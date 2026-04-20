import pandas as pd
import numpy as np

def load_and_clean_data(file_path):
    """
    Loads and cleans Bloomberg Commodities Excel file
    Handles structure with multiple horizontal ticker blocks
    """
    # Load the file
    xl = pd.ExcelFile(file_path)
    df_raw = xl.parse('Monthly', header=None)
    
    print(f"🔍 File dimensions: {df_raw.shape}")
    
    # ===== 1. FIND TICKER AND FIELDS ROW =====
    tickers_row_idx = 3
    fields_row_idx = 5
    
    print(f"📍 Ticker row: {tickers_row_idx}")
    print(f"📍 Fields row: {fields_row_idx}")
    
    # ===== 2. EXTRACT ALL TICKERS AND THEIR FIELDS =====
    all_tickers = []
    all_fields = []
    col_groups = []
    
    current_group = []
    current_ticker = None
    current_fields = []
    
    for col in range(1, df_raw.shape[1]):
        ticker_val = df_raw.iloc[tickers_row_idx, col]
        field_val = df_raw.iloc[fields_row_idx, col]
        
        if pd.notna(ticker_val) and str(ticker_val).strip() != '' and ticker_val != 'nan':
            if current_ticker is not None:
                col_groups.append((current_ticker, current_fields, current_group))
            current_ticker = str(ticker_val).strip()
            current_fields = []
            current_group = []
        
        if pd.notna(field_val) and str(field_val).strip() != '' and field_val != 'nan':
            current_fields.append(str(field_val).strip())
        
        current_group.append(col)
    
    if current_ticker is not None:
        col_groups.append((current_ticker, current_fields, current_group))
    
    print(f"📊 Ticker groups found: {len(col_groups)}")
    for ticker, fields, cols in col_groups:
        print(f"   - {ticker}: {len(fields)} fields in columns {cols[0]}-{cols[-1]}")
    
    # ===== 3. EXTRACT DATA BY GROUP =====
    date_col = 0
    
    # Find where data starts
    data_start_idx = None
    for idx in range(7, min(50, df_raw.shape[0])):
        val = df_raw.iloc[idx, date_col]
        try:
            if pd.notna(val):
                test_date = pd.to_datetime(val, errors='coerce')
                if pd.notna(test_date):
                    data_start_idx = idx
                    break
        except:
            continue
    
    if data_start_idx is None:
        data_start_idx = 7
    
    print(f"📍 Data start row: {data_start_idx}")
    
    # ===== 4. BUILD DATAFRAME FOR EACH TICKER =====
    all_dfs = []
    
    for ticker, fields, col_indices in col_groups:
        ticker_data = {}
        
        # Extract dates
        dates = []
        for idx in range(data_start_idx, df_raw.shape[0]):
            date_val = df_raw.iloc[idx, date_col]
            if pd.notna(date_val):
                try:
                    date = pd.to_datetime(date_val, errors='coerce')
                    if pd.notna(date):
                        dates.append(date)
                except:
                    pass
        
        if not dates:
            continue
        
        # For each field, extract data
        for i, field in enumerate(fields):
            col_idx = col_indices[i] if i < len(col_indices) else col_indices[-1] + i - len(col_indices) + 1
            
            if col_idx >= df_raw.shape[1]:
                continue
            
            values = []
            for idx in range(data_start_idx, df_raw.shape[0]):
                val = df_raw.iloc[idx, col_idx]
                if pd.isna(val) or str(val) in ['#NAME?', '#N/A', 'N/A', '#NUM!', '']:
                    values.append(np.nan)
                else:
                    try:
                        values.append(float(val))
                    except:
                        values.append(np.nan)
            
            min_len = min(len(dates), len(values))
            col_name = f"{ticker}_{field}"
            ticker_data[col_name] = values[:min_len]
        
        if ticker_data:
            df_ticker = pd.DataFrame(ticker_data)
            df_ticker.index = dates[:len(df_ticker)]
            all_dfs.append(df_ticker)
    
    # ===== 5. COMBINE ALL TICKERS =====
    if not all_dfs:
        raise ValueError("Could not extract data from any ticker")
    
    df_combined = pd.concat(all_dfs, axis=1)
    
    print(f"📊 Combined data: {df_combined.shape}")
    
    # ===== 6. FINAL CLEANING =====
    df_combined = df_combined.replace([np.inf, -np.inf], np.nan)
    
    before_cols = len(df_combined.columns)
    df_combined = df_combined.dropna(axis=1, how='all')
    print(f"🗑️ Removed {before_cols - len(df_combined.columns)} empty columns")
    
    before_rows = len(df_combined)
    df_combined = df_combined.dropna(axis=0, how='all')
    print(f"🗑️ Removed {before_rows - len(df_combined)} empty rows")
    
    # ===== 7. SELECT TARGET =====
    px_last_cols = [col for col in df_combined.columns if 'PX_LAST' in col]
    
    if not px_last_cols:
        numeric_cols = df_combined.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) == 0:
            raise ValueError("No numeric columns found")
        target_col = numeric_cols[0]
    else:
        target_col = px_last_cols[0]
    
    print(f"🎯 Using target: {target_col}")
    
    # Create target (next month's price)
    df_combined['target'] = df_combined[target_col].shift(-1)
    
    before_rows = len(df_combined)
    df_combined = df_combined.dropna(subset=['target'])
    print(f"🗑️ Removed {before_rows - len(df_combined)} rows without target")
    
    # Remove columns with >50% NaN
    nan_thresh = 0.5
    before_cols = len(df_combined.columns)
    df_combined = df_combined.loc[:, df_combined.isnull().mean() < nan_thresh]
    print(f"🗑️ Removed {before_cols - len(df_combined.columns)} columns with >50% NaN")
    
    # Keep only numeric columns
    numeric_cols = df_combined.select_dtypes(include=[np.number]).columns
    df_combined = df_combined[numeric_cols]
    
    print(f"📊 Final dataset shape: {df_combined.shape}")
    print(f"📅 Date range: {df_combined.index.min()} to {df_combined.index.max()}")
    
    return df_combined, target_col

if __name__ == "__main__":
    try:
        df, target = load_and_clean_data("Bloomberg_commodities_prices.xlsx")
        print("\n✅ Data loaded successfully!")
        print(f"\n📋 First 5 rows:")
        print(df.head())
        print(f"\n📊 Available columns:")
        for col in df.columns:
            print(f"   - {col}")
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()