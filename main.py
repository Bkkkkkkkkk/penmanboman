import os
import json
import gspread
import pandas as pd
import numpy as np
from datetime import datetime
import pytz
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
import scipy.stats as stats
from statsmodels.tsa.seasonal import STL # 🌟 新增 STL 模組

def main():
    print("🚀 啟動戰術雷達：深度分析模組連線中...")
    
    # ==========================================
    # 1. 系統連線與資料讀取
    # ==========================================
    sheet_id = os.environ.get("SPREADSHEET_ID")
    sa_json = os.environ.get("GCP_SERVICE_ACCOUNT")
    
    credentials = json.loads(sa_json)
    gc = gspread.service_account_from_dict(credentials)
    sh = gc.open_by_key(sheet_id)
    
    ws_raw = sh.worksheet("Raw_Data")
    raw_records = ws_raw.get_all_values()
    df = pd.DataFrame(raw_records[1:], columns=raw_records[0])
    
    # ==========================================
    # 2. 資料清洗與聚合
    # ==========================================
    df['Amount'] = df['Amount'].astype(str).str.replace(r'[NT\$,\s]', '', regex=True)
    df['Amount'] = pd.to_numeric(df['Amount'], errors='coerce').fillna(0)
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    
    df_clean = df[(df['Type'] == '支出') & (~df['Category'].isin(['股票', '固定帳單']))].copy()
    
    daily_df = df_clean.groupby(df_clean['Date'].dt.date)['Amount'].sum().reset_index()
    daily_df['Date'] = pd.to_datetime(daily_df['Date'])
    
    if not daily_df.empty:
        min_date = daily_df['Date'].min()
        max_date = daily_df['Date'].max()
        full_dates = pd.date_range(start=min_date, end=max_date)
        daily_df = daily_df.set_index('Date').reindex(full_dates, fill_value=0).reset_index()
        daily_df.rename(columns={'index': 'Date'}, inplace=True)

    # ==========================================
    # 3. Phase 4.1: 模型 PK (Walk-Forward Validation)
    # ==========================================
    print("⚔️ 執行 Phase 4.1: 模型 PK 回測...")
    daily_df['DayOfWeek'] = daily_df['Date'].dt.dayofweek
    daily_df['IsWeekend'] = daily_df['DayOfWeek'].isin([5, 6]).astype(int)
    daily_df['Lag_1'] = daily_df['Amount'].shift(1).fillna(0)
    daily_df['Rolling_Mean_3'] = daily_df['Amount'].shift(1).rolling(window=3, min_periods=1).mean().fillna(0)
    daily_df['Rolling_Mean_7'] = daily_df['Amount'].shift(1).rolling(window=7, min_periods=1).mean().fillna(0)
    
    model_df = daily_df.dropna().reset_index(drop=True)
    
    train_window, test_window = 90, 30
    mae_ewma_list, mae_median_list, mae_ridge_list = [], [], []
    features = ['DayOfWeek', 'IsWeekend', 'Lag_1', 'Rolling_Mean_3', 'Rolling_Mean_7']
    
    for start_idx in range(0, len(model_df) - train_window - test_window + 1, test_window):
        train_end = start_idx + train_window
        test_end = train_end + test_window
        
        train_data, test_data = model_df.iloc[start_idx:train_end], model_df.iloc[train_end:test_end]
        y_true = test_data['Amount'].values
        
        ewma_pred = np.full(len(y_true), train_data['Amount'].ewm(span=7, adjust=False).mean().iloc[-1])
        mae_ewma_list.append(mean_absolute_error(y_true, ewma_pred))
        
        median_pred = np.full(len(y_true), train_data['Amount'].median())
        mae_median_list.append(mean_absolute_error(y_true, median_pred))
        
        ridge_model = Ridge(alpha=1.0)
        ridge_model.fit(train_data[features], train_data['Amount'])
        mae_ridge_list.append(mean_absolute_error(y_true, np.clip(ridge_model.predict(test_data[features]), 0, None)))

    final_ewma = round(np.mean(mae_ewma_list), 1) if mae_ewma_list else 0
    final_median = round(np.mean(mae_median_list), 1) if mae_median_list else 0
    final_ridge = round(np.mean(mae_ridge_list), 1) if mae_ridge_list else 0
    scores = {"EWMA均線": final_ewma, "中位數模型": final_median, "Ridge迴歸": final_ridge}
    best_model = min(scores, key=scores.get) if scores else "資料量不足"

    # ==========================================
    # 4. Phase 4.2: 大額事件分析 (Extreme Value Theory)
    # ==========================================
    print("🌪️ 執行 Phase 4.2: 洪峰極端值分析...")
    event_threshold = np.percentile(df_clean['Amount'], 95)
    large_events = df_clean[df_clean['Amount'] > event_threshold]['Amount']
    
    total_days = (df_clean['Date'].max() - df_clean['Date'].min()).days
    total_months = max(total_days / 30.0, 1)
    lambda_poisson = len(large_events) / total_months
    excesses = large_events - event_threshold
    
    if len(excesses) > 0:
        c, loc, scale = stats.genpareto.fit(excesses, floc=0)
        if c < 1:
            expected_excess = stats.genpareto.mean(c, loc, scale)
            expected_magnitude = event_threshold + expected_excess
        else:
            expected_magnitude = event_threshold + excesses.mean()
    else:
        c, expected_magnitude = 0, 0

    # ==========================================
    # 5. Phase 4.3: 趨勢分解 (STL Decomposition)
    # ==========================================
    print("📈 執行 Phase 4.3: STL 趨勢分解...")
    try:
        # 將資料轉為每日頻率的時間序列
        ts_data = daily_df.set_index('Date')['Amount'].asfreq('D').fillna(0)
        
        # 執行 STL (設定週期為 7 天，開啟 robust 抵抗極端值)
        stl = STL(ts_data, period=7, robust=True)
        res = stl.fit()
        
        # 取出趨勢線
        trend = res.trend
        seasonal = res.seasonal
        
        # 1. 抓取最新基礎水位 (最近一天的真實趨勢值)
        current_trend = trend.iloc[-1]
        
        # 2. 評估水位漂移 (與 30 天前相比，基礎開銷是變高還是變低？)
        if len(trend) >= 30:
            trend_30d_drift = current_trend - trend.iloc[-30]
        else:
            trend_30d_drift = current_trend - trend.iloc[0]
            
        # 3. 星期震盪幅度 (最花錢的日子跟最省錢的日子，光是「週期」就差多少)
        seasonal_amplitude = seasonal.max() - seasonal.min()
        
    except Exception as e:
        print(f"❌ STL 分解失敗: {e}")
        current_trend, trend_30d_drift, seasonal_amplitude = 0, 0, 0

    # ==========================================
    # 6. 統一匯出資料庫 (寫回 py_output)
    # ==========================================
    ws_out = sh.worksheet("py_output")
    taipei_tz = pytz.timezone('Asia/Taipei')
    update_time = datetime.now(taipei_tz).strftime("%Y-%m-%d %H:%M:%S")
    
    export_data = [
        ["model_mae_ewma", final_ewma, update_time],
        ["model_mae_median", final_median, update_time],
        ["model_mae_ridge", final_ridge, update_time],
        ["model_best_winner", best_model, update_time],
        ["event_threshold_p95", round(event_threshold, 0), update_time],
        ["event_lambda_monthly", round(lambda_poisson, 2), update_time],
        ["event_tail_shape_c", round(c, 3), update_time],
        ["event_expected_magnitude", round(expected_magnitude, 0), update_time],
        ["stl_current_trend", round(current_trend, 0), update_time],
        ["stl_30d_drift", round(trend_30d_drift, 0), update_time],
        ["stl_seasonal_amplitude", round(seasonal_amplitude, 0), update_time]
    ]
    
    # 擴增寫入範圍至 C13，把所有高階指標一網打盡
    ws_out.update('A3:C13', export_data)
    print("✅ 全模組戰果已成功寫入 py_output！")

if __name__ == "__main__":
    main()
