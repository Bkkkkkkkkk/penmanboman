import os
import json
import gspread
import pandas as pd
import numpy as np
from datetime import datetime
import pytz
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error

def main():
    print("🚀 啟動戰術雷達：深度歷史回測模組 (Walk-Forward Validation)...")
    
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
    
    # 將資料轉為 DataFrame (假設第一列為標題)
    df = pd.DataFrame(raw_records[1:], columns=raw_records[0])
    
    # ==========================================
    # 2. 資料清洗與聚合 (過濾剛性支出)
    # ==========================================
    # 清理金額欄位 (去除 NT$ 與逗號)
    df['Amount'] = df['Amount'].astype(str).str.replace(r'[NT\$,\s]', '', regex=True)
    df['Amount'] = pd.to_numeric(df['Amount'], errors='coerce').fillna(0)
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    
    # 嚴格過濾：僅限「支出」，且排除「股票」與「固定帳單」
    df_clean = df[(df['Type'] == '支出') & (~df['Category'].isin(['股票', '固定帳單']))].copy()
    
    # 按日加總
    daily_df = df_clean.groupby(df_clean['Date'].dt.date)['Amount'].sum().reset_index()
    daily_df['Date'] = pd.to_datetime(daily_df['Date'])
    
    # 補齊時間序列破洞 (沒花錢的日子補 0)
    if not daily_df.empty:
        min_date = daily_df['Date'].min()
        max_date = daily_df['Date'].max()
        full_dates = pd.date_range(start=min_date, end=max_date)
        daily_df = daily_df.set_index('Date').reindex(full_dates, fill_value=0).reset_index()
        daily_df.rename(columns={'index': 'Date'}, inplace=True)
        
    print(f"📊 乾淨日資料準備完成，共 {len(daily_df)} 天的有效時序數據。")

    # ==========================================
    # 3. 特徵工程 (Feature Engineering)
    # ==========================================
    daily_df['DayOfWeek'] = daily_df['Date'].dt.dayofweek
    daily_df['IsWeekend'] = daily_df['DayOfWeek'].isin([5, 6]).astype(int)
    
    # 時間滯後特徵 (避免洩漏未來資訊，特徵必須 shift 1 天)
    daily_df['Lag_1'] = daily_df['Amount'].shift(1).fillna(0)
    daily_df['Rolling_Mean_3'] = daily_df['Amount'].shift(1).rolling(window=3, min_periods=1).mean().fillna(0)
    daily_df['Rolling_Mean_7'] = daily_df['Amount'].shift(1).rolling(window=7, min_periods=1).mean().fillna(0)
    
    # 刪除因為 shift 產生 NaN 的最初幾筆資料
    model_df = daily_df.dropna().reset_index(drop=True)

    # ==========================================
    # 4. Walk-Forward 回測框架與模型 PK
    # ==========================================
    train_window = 90  # 用過去 90 天訓練
    test_window = 30   # 預測未來 30 天
    
    mae_ewma_list = []
    mae_median_list = []
    mae_ridge_list = []
    
    # 特徵欄位 X，目標欄位 Y
    features = ['DayOfWeek', 'IsWeekend', 'Lag_1', 'Rolling_Mean_3', 'Rolling_Mean_7']
    
    # 滾動回測迴圈
    for start_idx in range(0, len(model_df) - train_window - test_window + 1, test_window):
        train_end = start_idx + train_window
        test_end = train_end + test_window
        
        train_data = model_df.iloc[start_idx:train_end]
        test_data = model_df.iloc[train_end:test_end]
        
        y_true = test_data['Amount'].values
        
        # --- 選手 A：EWMA (指數加權移動平均) ---
        # 模擬實務，以訓練集最後一天的 EWMA 值作為未來預測基準
        ewma_series = train_data['Amount'].ewm(span=7, adjust=False).mean()
        ewma_pred = np.full(len(y_true), ewma_series.iloc[-1])
        mae_ewma_list.append(mean_absolute_error(y_true, ewma_pred))
        
        # --- 選手 B：分位數模型 (中位數 P50) ---
        median_pred = np.full(len(y_true), train_data['Amount'].median())
        mae_median_list.append(mean_absolute_error(y_true, median_pred))
        
        # --- 選手 C：Ridge 迴歸模型 ---
        X_train = train_data[features]
        y_train = train_data['Amount']
        X_test = test_data[features]
        
        ridge_model = Ridge(alpha=1.0)
        ridge_model.fit(X_train, y_train)
        ridge_pred = ridge_model.predict(X_test)
        # 防止預測出負的消費金額
        ridge_pred = np.clip(ridge_pred, a_min=0, a_max=None)
        mae_ridge_list.append(mean_absolute_error(y_true, ridge_pred))

    # ==========================================
    # 5. 結算戰果與寫回 py_output
    # ==========================================
    # 如果資料量太少不夠滾動回測，給予預設值
    if not mae_ridge_list:
        final_ewma, final_median, final_ridge = 0, 0, 0
        best_model = "資料量不足"
    else:
        final_ewma = round(np.mean(mae_ewma_list), 1)
        final_median = round(np.mean(mae_median_list), 1)
        final_ridge = round(np.mean(mae_ridge_list), 1)
        
        scores = {"EWMA均線": final_ewma, "中位數模型": final_median, "Ridge迴歸": final_ridge}
        best_model = min(scores, key=scores.get) # MAE 越小越好

    print(f"🏆 模型 PK 結算 (平均絕對誤差 MAE):")
    print(f"EWMA: ${final_ewma} | 分位數: ${final_median} | Ridge: ${final_ridge}")
    print(f"勝出者: {best_model}")

    # 寫回試算表
    ws_out = sh.worksheet("py_output")
    taipei_tz = pytz.timezone('Asia/Taipei')
    update_time = datetime.now(taipei_tz).strftime("%Y-%m-%d %H:%M:%S")
    
    export_data = [
        ["model_mae_ewma", final_ewma, update_time],
        ["model_mae_median", final_median, update_time],
        ["model_mae_ridge", final_ridge, update_time],
        ["model_best_winner", best_model, update_time]
    ]
    
    # 清除舊的指標並寫入最新戰果 (從 A3 開始寫，保留標題與前次連線測試)
    ws_out.update('A3:C6', export_data)
    print("✅ 深度回測戰果已成功寫回 py_output 分頁！")

if __name__ == "__main__":
    main()
