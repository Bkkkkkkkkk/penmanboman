import os
import json
import gspread
import pandas as pd
import numpy as np
from datetime import datetime
import pytz
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
import scipy.stats as stats
from statsmodels.tsa.seasonal import STL

def main():
    print("🚀 啟動戰術雷達：終極特徵工程與深度分析模組連線中...")

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
    # 3. Phase 4.1: 終極特徵工程與模型 PK
    # ==========================================
    print("🧠 萃取高階消費行為特徵...")

    # [基礎特徵]
    daily_df['DayOfWeek'] = daily_df['Date'].dt.dayofweek
    daily_df['IsWeekend'] = daily_df['DayOfWeek'].isin([5, 6]).astype(int)

    # [滯後特徵]
    daily_df['Lag_1'] = daily_df['Amount'].shift(1).fillna(0)
    daily_df['Lag_7'] = daily_df['Amount'].shift(7).fillna(0)  # 上次同星期花費

    # [均線特徵]
    daily_df['Rolling_Mean_3'] = daily_df['Amount'].shift(1).rolling(window=3, min_periods=1).mean().fillna(0)
    daily_df['Rolling_Mean_7'] = daily_df['Amount'].shift(1).rolling(window=7, min_periods=1).mean().fillna(0)

    # [日曆特徵]
    daily_df['DaysToMonthEnd'] = daily_df['Date'].apply(lambda d: d.days_in_month - d.day)

    # ⚠️ 你可以修改這裡的 5，換成你真實的發薪日 (例如 10 號就改成 10)
    payday_date = 5
    def get_days_to_payday(d, payday):
        if d.day <= payday:
            return payday - d.day
        else:
            return d.days_in_month - d.day + payday
    daily_df['DaysToPayday'] = daily_df['Date'].apply(lambda d: get_days_to_payday(d, payday_date))

    # [忍耐爆發指數 - 修正版 v2]
    # v1問題：用全體近30天中位數(P50)當門檻，沒有分星期，星期效應與真正的
    #   壓抑消費訊號混在一起。
    # v2問題（本次修正）：分星期後，用「今天所屬星期的P25」去比較「昨天的花費」，
    #   但昨天可能是別的星期，星期對不齊，導致觸發率被系統性灌高（實測36.5% vs 理論25%）。
    # 修正：先在「星期對齊」的前提下，判斷當天是否相對於自己星期的歷史P25偏低（is_low_raw），
    #   再整體平移一天才當作特徵使用（避免用當天結果預測當天）。
    dow_p25 = daily_df.groupby('DayOfWeek')['Amount'].transform(
        lambda s: s.shift(1).rolling(window=12, min_periods=4).quantile(0.25)
    )
    # 資料不足4次同星期樣本時的 fallback - 修正版
    # 舊版問題：用「全體中位數」頂著，遠高於大部分星期的真實P25，讓 fallback 期間
    #   （約前4-5週，佔197天資料約14%）的門檻異常寬鬆，稀釋整體觸發率往上偏移。
    #   而且用的是全體197天算出的中位數，等於在早期就用到了「未來」資料，邏輯不乾淨。
    # 修正：fallback 改用「截至前一天為止、全體資料」的展開P25（expanding + shift(1)），
    #   量綱跟主邏輯一致（都是P25），且不使用未來資料。
    global_p25_to_date = daily_df['Amount'].expanding(min_periods=4).quantile(0.25).shift(1)
    daily_df['DowP25'] = dow_p25.fillna(global_p25_to_date)

    # 第一步：星期對齊下，判斷「當天自己」是否相對於「自己星期」偏低
    is_low_raw = (daily_df['Amount'] < daily_df['DowP25']).astype(int)

    # 第二步：整體往後移一天，變成可用於「預測當天」的特徵，避免用當天結果預測當天
    is_low = is_low_raw.shift(1).fillna(0).astype(int)
    daily_df['Low_Spend_Streak'] = is_low.groupby((is_low == 0).cumsum()).cumsum()

    # 健檢診斷：確認觸發率是否符合預期，只印出來看，不寫入 py_output，不影響下游邏輯
    # is_low_raw 是乾淨的基準線（星期已對齊），理論上應貼近25%
    marginal_rate_raw = is_low_raw.mean()
    marginal_rate_shifted = is_low.mean()
    streak_rate_2plus = (daily_df['Low_Spend_Streak'] >= 2).mean()
    streak_rate_3plus = (daily_df['Low_Spend_Streak'] >= 3).mean()
    print(f"📊 [健檢] is_low_raw 邊際觸發率(星期對齊基準線): {marginal_rate_raw:.1%}（理論應接近25%）")
    print(f"📊 [健檢] is_low 邊際觸發率(平移後，餵給模型的版本): {marginal_rate_shifted:.1%}（理論應接近25%）")
    print(f"📊 [健檢] Low_Spend_Streak≥2 佔比: {streak_rate_2plus:.1%}（理論預期約6%左右）")
    print(f"📊 [健檢] Low_Spend_Streak≥3 佔比: {streak_rate_3plus:.1%}（理論預期約1.5%左右）")

    # 清除 shift(7) 產生的空值
    model_df = daily_df.dropna().reset_index(drop=True)

    train_window, test_window = 90, 30
    mae_ewma_list, mae_median_list, mae_ridge_list = [], [], []

    # 所有特徵一齊上陣
    features = ['DayOfWeek', 'IsWeekend', 'Lag_1', 'Lag_7', 'Rolling_Mean_3',
                'Rolling_Mean_7', 'DaysToMonthEnd', 'DaysToPayday', 'Low_Spend_Streak']

    ridge_model = None
    scaler = None

    for start_idx in range(0, len(model_df) - train_window - test_window + 1, test_window):
        train_end = start_idx + train_window
        test_end = train_end + test_window

        train_data, test_data = model_df.iloc[start_idx:train_end], model_df.iloc[train_end:test_end]
        y_true = test_data['Amount'].values

        # EWMA（不受此次改動影響）
        ewma_pred = np.full(len(y_true), train_data['Amount'].ewm(span=7, adjust=False).mean().iloc[-1])
        mae_ewma_list.append(mean_absolute_error(y_true, ewma_pred))

        # 中位數（不受此次改動影響）
        median_pred = np.full(len(y_true), train_data['Amount'].median())
        mae_median_list.append(mean_absolute_error(y_true, median_pred))

        # Ridge 迴歸 - 修正版：訓練前標準化特徵
        # 標準化不改變 Ridge 對線性資料的預測能力，MAE 應與舊版接近；
        # 主要差異在係數尺度變得可比，top_feature_name 的結果才有意義。
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(train_data[features])
        X_test_scaled = scaler.transform(test_data[features])

        ridge_model = Ridge(alpha=1.0)
        ridge_model.fit(X_train_scaled, train_data['Amount'])
        mae_ridge_list.append(
            mean_absolute_error(y_true, np.clip(ridge_model.predict(X_test_scaled), 0, None))
        )

    final_ewma = round(np.mean(mae_ewma_list), 1) if mae_ewma_list else 0
    final_median = round(np.mean(mae_median_list), 1) if mae_median_list else 0
    final_ridge = round(np.mean(mae_ridge_list), 1) if mae_ridge_list else 0
    scores = {"EWMA均線": final_ewma, "中位數模型": final_median, "Ridge迴歸": final_ridge}
    best_model = min(scores, key=scores.get) if scores else "資料量不足"

    # 印出影響預測最深的特徵（現在用標準化後的係數，比較才公平）
    top_feature_name = "N/A"
    if ridge_model is not None:
        coefs = pd.Series(ridge_model.coef_, index=features)
        top_feature_name = coefs.abs().idxmax()
        print(f"🎯 [行為解析] 影響你預測模型最深的特徵是: {top_feature_name}")

    # ==========================================
    # 4. Phase 4.2: 大額事件分析（未改動）
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
            expected_magnitude = event_threshold + stats.genpareto.mean(c, loc, scale)
        else:
            expected_magnitude = event_threshold + excesses.mean()
    else:
        c, expected_magnitude = 0, 0

    # ==========================================
    # 5. Phase 4.3: 趨勢分解 (STL Decomposition)（未改動）
    # ==========================================
    print("📈 執行 Phase 4.3: STL 趨勢分解...")
    try:
        ts_data = daily_df.set_index('Date')['Amount'].asfreq('D').fillna(0)
        stl = STL(ts_data, period=7, robust=True)
        res = stl.fit()
        trend = res.trend
        seasonal = res.seasonal

        current_trend = trend.iloc[-1]
        trend_30d_drift = current_trend - trend.iloc[-30] if len(trend) >= 30 else current_trend - trend.iloc[0]
        seasonal_amplitude = seasonal.max() - seasonal.min()
    except Exception as e:
        print(f"❌ STL 分解失敗: {e}")
        current_trend, trend_30d_drift, seasonal_amplitude = 0, 0, 0

    # ==========================================
    # 6. 統一匯出資料庫 (寫回 py_output)（結構未改動）
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
        ["stl_seasonal_amplitude", round(seasonal_amplitude, 0), update_time],
        ["model_top_feature", top_feature_name, update_time]
    ]

    ws_out.update('A3:C14', export_data)
    print("✅ 全模組戰果與特徵工程已成功寫入 py_output！")

if __name__ == "__main__":
    main()
