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
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import scipy.stats as stats
from statsmodels.tsa.seasonal import STL

def sync_params(ws_params, updates, alpha=0.3):
    """
    把校準參數寫回params分頁，數值欄位用平滑機制更新（新值佔alpha比例、舊值佔1-alpha），
    避免單月雜訊讓參數暴衝。沿用既有的「參數名稱｜數值｜說明」三欄格式，
    只更新我們關心的幾列，完全不動既有的孤兒參數（回測窗口天數/IQR_倍率/趨勢修正係數）。
    找不到舊值時（第一次寫入、或格式讀取失敗），保守地直接採用新值，不阻斷流程。
    """
    existing = ws_params.get_all_values()
    name_to_row = {row[0]: idx + 1 for idx, row in enumerate(existing) if row}  # 1-indexed，含表頭

    taipei_tz = pytz.timezone('Asia/Taipei')
    today_str = datetime.now(taipei_tz).strftime("%Y-%m-%d")

    for name, new_value in updates.items():
        new_value = float(new_value)  # 確保是原生 Python float，避免 numpy 型別造成 JSON 序列化失敗（上次撞名bug的同類風險）
        if name in name_to_row:
            row_idx = name_to_row[name]
            try:
                old_value = float(existing[row_idx - 1][1])
                smoothed = round(alpha * new_value + (1 - alpha) * old_value, 1)
            except (ValueError, IndexError):
                smoothed = round(new_value, 1)
            ws_params.update_cell(row_idx, 2, smoothed)
        else:
            ws_params.append_row([name, round(new_value, 1), "Python自動新增，說明待補充"])

    if "上次校準日期" in name_to_row:
        ws_params.update_cell(name_to_row["上次校準日期"], 2, today_str)
    else:
        ws_params.append_row(["上次校準日期", today_str, "Python 最後一次更新這些參數的時間"])


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

    ws_params = sh.worksheet("params")
    ws_centroids = sh.worksheet("cluster_centroids")
    ws_diag = sh.worksheet("model_diagnostics")

    # ==========================================
    # 2. 資料清洗與聚合
    # ==========================================
    df['Amount'] = df['Amount'].astype(str).str.replace(r'[NT\$,\s]', '', regex=True)
    df['Amount'] = pd.to_numeric(df['Amount'], errors='coerce').fillna(0)
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')

    df_clean = df[(df['Type'] == '支出') & (~df['Category'].isin(['股票', '固定帳單']))].copy()

    # ------------------------------------------
    # 2.1 日常 vs 大額事件分離（本輪新增，架構級修正）
    # ------------------------------------------
    # 問題：舊版只用「類別名稱」排除股票/固定帳單，但實際資料顯示「其他」類別是
    #   大雜燴——裡面同時混了瑣碎小額（香油錢$100）跟不規律大額（玉山卡費$10,882、
    #   公會入會費$10,500、老婆旅費$23,784）。這些大額一次性支出的類別標籤逃過了
    #   排除邏輯，直接被加進每日總支出，污染了Phase 4.1(星期分桶/Low_Spend_Streak/
    #   Ridge訓練)跟Phase 4.3(STL)的日常消費基準線。
    # 修正：不再用類別名稱切分，改用P95門檻（跟Phase 4.2同一套標準）在源頭把
    #   「日常小額(habitual_df)」跟「大額事件(large_events_df)」拆開，
    #   Phase 4.1/4.3 只用 habitual_df，Phase 4.2 只用 large_events_df，
    #   兩條管線不再互相污染。
    event_threshold = np.percentile(df_clean['Amount'], 95)
    large_events_df = df_clean[df_clean['Amount'] > event_threshold].copy()
    habitual_df = df_clean[df_clean['Amount'] <= event_threshold].copy()
    print(f"📊 [資料分離] 日常小額: {len(habitual_df)}筆 / 大額事件: {len(large_events_df)}筆（P95門檻: {event_threshold:.0f}）")

    daily_df = habitual_df.groupby(habitual_df['Date'].dt.date)['Amount'].sum().reset_index()
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
    # v2問題：分星期後，用「今天所屬星期的P25」去比較「昨天的花費」，
    #   但昨天可能是別的星期，星期對不齊，導致觸發率被系統性灌高（實測36.5% vs 理論25%）。
    # 修正：先在「星期對齊」的前提下，判斷當天是否相對於自己星期的歷史P25偏低（is_low_raw），
    #   再整體平移一天才當作特徵使用（避免用當天結果預測當天）。
    dow_p25 = daily_df.groupby('DayOfWeek')['Amount'].transform(
        lambda s: s.shift(1).rolling(window=12, min_periods=4).quantile(0.25)
    )

    # 資料不足4次同星期樣本時的 fallback - 修正版
    # 舊版問題：用「全體中位數」頂著，遠高於大部分星期的真實P25，讓 fallback 期間
    #   （約前4-5週）的門檻異常寬鬆，稀釋整體觸發率往上偏移；而且用的是全體資料
    #   算出的中位數，等於在早期就用到了「未來」資料，邏輯不乾淨。
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

        # EWMA（不受本輪改動影響）
        ewma_pred = np.full(len(y_true), train_data['Amount'].ewm(span=7, adjust=False).mean().iloc[-1])
        mae_ewma_list.append(mean_absolute_error(y_true, ewma_pred))

        # 中位數（不受本輪改動影響）
        median_pred = np.full(len(y_true), train_data['Amount'].median())
        mae_median_list.append(mean_absolute_error(y_true, median_pred))

        # Ridge 迴歸 - 標準化修正版：訓練前對特徵做標準化
        # 標準化不改變 Ridge 對線性資料的預測能力，MAE 應與未標準化版本接近；
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

    # 印出影響預測最深的特徵（用標準化後的係數，比較才公平）
    top_feature_name = "N/A"
    if ridge_model is not None:
        coefs = pd.Series(ridge_model.coef_, index=features)
        top_feature_name = coefs.abs().idxmax()
        print(f"🎯 [行為解析] 影響你預測模型最深的特徵是: {top_feature_name}")

        # 完整係數排序（只印出來看，不寫入py_output）：標準化後的係數，正負號代表方向
        # （正=該特徵越大，預測花費越高；負=該特徵越大，預測花費越低），
        # 數值大小代表影響力強弱，彼此可以公平比較（因為都已經標準化過）
        coef_ranked = coefs.reindex(coefs.abs().sort_values(ascending=False).index)
        print("📋 [完整係數排序]")
        for feat_name, val in coef_ranked.items():
            direction = "正相關(越高花費越多)" if val > 0 else "負相關(越高花費越少)"
            print(f"    {feat_name}: {val:+.1f}（{direction}）")

    # ==========================================
    # 4. Phase 4.2: 大額事件分析
    # ==========================================
    # 本輪改動：large_events 改用第2.1節已經拆分好的 large_events_df，
    # 不再從 df_clean 重新篩一次，跟 Phase 4.1/4.3 用的 habitual_df 徹底分流，
    # event_threshold 也沿用第2.1節算好的同一個門檻，避免重複計算或標準不一致。
    print("🌪️ 執行 Phase 4.2: 洪峰極端值分析...")
    large_events = large_events_df['Amount']

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
    # 5. Phase 4.3: 趨勢分解 (STL Decomposition)
    # ==========================================
    print("📈 執行 Phase 4.3: STL 趨勢分解...")
    try:
        # log1p 轉換修正版
        # 舊版問題：daily_df 是零膨脹資料，STL即使開了robust=True，也容易把離群尖峰的
        #   能量錯誤分配進「季節性」分量，導致seasonal_amplitude遠大於current_trend
        #   這種不合理結果（本輪daily_df已經是habitual_df聚合出來的，尖峰問題已大幅
        #   緩解，但log1p仍保留作為雙重保險，讓STL對剩餘的日常波動更穩健）。
        # 修正：對Amount先做log1p壓縮尺度差異，trend/seasonal都在log空間，
        #   換算回原始尺度時語意要跟著調整：
        #   - trend：可以合理換回金額（expm1），代表「當前基礎消費水位」
        #   - seasonal：在log空間是「倍數效應」而非「金額」，換算成「最花錢星期是
        #     最省錢星期的幾倍」（fold change）才符合log分解的數學意義
        ts_data = daily_df.set_index('Date')['Amount'].asfreq('D').fillna(0)
        ts_data_log = np.log1p(ts_data)

        stl = STL(ts_data_log, period=7, robust=True)
        res = stl.fit()
        trend_log = res.trend
        seasonal_log = res.seasonal

        # trend 換回金額尺度，代表當前/30天前的基礎消費水位
        current_trend = np.expm1(trend_log.iloc[-1])
        trend_30d_ago = np.expm1(trend_log.iloc[-30]) if len(trend_log) >= 30 else np.expm1(trend_log.iloc[0])
        trend_30d_drift = current_trend - trend_30d_ago

        # seasonal 換算成「最花錢星期是最省錢星期的幾倍」，單位是倍率不是金額
        # 修正：不能直接用 seasonal.max()-seasonal.min()！因為只有約30個完整週期，
        #   同星期不同週的seasonal值仍會抖動（robust=True會依每週殘差動態加權），
        #   直接抓極值等於抓到「單一天的雜訊」而非「星期幾的代表值」（實測曾抓到
        #   兩個不相干的單日，算出23.5倍這種不合理數字）。正確做法是先依星期幾
        #   平均，取得每個星期的「典型」季節效應，再用平均值的極差算倍率。
        seasonal_df = seasonal_log.reset_index()
        seasonal_df.columns = ['Date', 'seasonal']
        seasonal_df['dow'] = seasonal_df['Date'].dt.dayofweek
        seasonal_by_dow = seasonal_df.groupby('dow')['seasonal'].mean()
        seasonal_fold_change = np.exp(seasonal_by_dow.max() - seasonal_by_dow.min())
    except Exception as e:
        print(f"❌ STL 分解失敗: {e}")
        current_trend, trend_30d_drift, seasonal_fold_change = 0, 0, 0

    # ==========================================
    # 6. Phase 4.4: 類別關聯挖掘
    # ==========================================
    print("🔗 執行 Phase 4.4: 類別關聯挖掘...")

    # 沿用 habitual_df（已排除大額事件），跟日常分析同一套資料基礎，
    # 避免像「老婆旅費」這種一次性大額製造出不具代表性的假相關
    MIN_CATEGORY_COUNT = 20  # 樣本數太少的類別（如3C/家電只有5筆），相關係數不可靠，先排除
    category_counts = habitual_df['Category'].value_counts()
    valid_categories = category_counts[category_counts >= MIN_CATEGORY_COUNT].index.tolist()
    print(f"📊 [4.4] 納入分析的類別（樣本數≥{MIN_CATEGORY_COUNT}）: {valid_categories}")

    cat_habitual = habitual_df[habitual_df['Category'].isin(valid_categories)].copy()
    cat_pivot = cat_habitual.pivot_table(
        index=cat_habitual['Date'].dt.date, columns='Category', values='Amount',
        aggfunc='sum', fill_value=0
    )
    cat_pivot.index = pd.to_datetime(cat_pivot.index)
    full_dates_cat = pd.date_range(cat_pivot.index.min(), cat_pivot.index.max())
    cat_pivot = cat_pivot.reindex(full_dates_cat, fill_value=0)
    cols = cat_pivot.columns.tolist()

    # --- 同日相關係數矩陣 ---
    corr_matrix = cat_pivot.corr()
    corr_pairs = []
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            corr_pairs.append((cols[i], cols[j], corr_matrix.iloc[i, j]))
    corr_pairs_sorted = sorted(corr_pairs, key=lambda x: abs(x[2]), reverse=True)

    print("📋 [4.4-同日關聯] Top 5 相關係數配對：")
    for a, b, v in corr_pairs_sorted[:5]:
        print(f"    {a} × {b}: {v:+.3f}")

    # --- 滯後關聯：類別A今天 vs 類別B未來1~3天 ---
    # 樣本量本來就不大（單一類別日常筆數幾十到幾百筆），配對數又多（多重比較問題），
    # 這邊p值只當「探索方向」參考，不做嚴謹統計推論，之後資料量增加可以再收斂
    lag_results = []
    for lag in [1, 2, 3]:
        for a in cols:
            for b in cols:
                if a == b:
                    continue
                shifted_b = cat_pivot[b].shift(-lag)
                valid_mask = shifted_b.notna()
                if valid_mask.sum() < 30:  # 樣本太少不列入
                    continue
                r, p = stats.pearsonr(cat_pivot[a][valid_mask], shifted_b[valid_mask])
                lag_results.append((a, b, lag, r, p))

    lag_results_sorted = sorted(lag_results, key=lambda x: abs(x[3]), reverse=True)
    print("📋 [4.4-滯後關聯] Top 5 滯後相關（探索性參考，非嚴謹統計推論）：")
    for a, b, lag, r, p in lag_results_sorted[:5]:
        sig_mark = "✓p<0.05" if p < 0.05 else "不顯著"
        print(f"    {a}(t) → {b}(t+{lag}): r={r:+.3f}, p={p:.3f} {sig_mark}")

    # 摘要成可寫回 py_output 的欄位（同日關聯前2名 + 滯後關聯第1名）
    if len(corr_pairs_sorted) >= 1:
        corr_top1_pair = f"{corr_pairs_sorted[0][0]}×{corr_pairs_sorted[0][1]}"
        corr_top1_value = round(corr_pairs_sorted[0][2], 3)
    else:
        corr_top1_pair, corr_top1_value = "資料不足", 0

    if len(corr_pairs_sorted) >= 2:
        corr_top2_pair = f"{corr_pairs_sorted[1][0]}×{corr_pairs_sorted[1][1]}"
        corr_top2_value = round(corr_pairs_sorted[1][2], 3)
    else:
        corr_top2_pair, corr_top2_value = "資料不足", 0

    if len(lag_results_sorted) >= 1:
        la, lb, llag, lr, lp = lag_results_sorted[0]
        lag_top1_relation = f"{la}(t)→{lb}(t+{llag})"
        lag_top1_value = round(lr, 3)
    else:
        lag_top1_relation, lag_top1_value = "資料不足", 0

    # ==========================================
    # 7. Phase 4.5: 消費模式聚類 (KMeans)
    # ==========================================
    print("🧩 執行 Phase 4.5: 消費模式聚類...")

    # 沿用 4.4 整理好的「日期×類別」攤平資料當特徵向量，每一天是一筆樣本，
    # 每個類別的當日金額是一個維度，讓 KMeans 找出「常一起出現的消費組合」
    scaler_cluster = StandardScaler()
    X_cluster = scaler_cluster.fit_transform(cat_pivot)

    # 依輪廓係數(silhouette score)自動挑選群數k(2~6)，不手動猜k，
    # 樣本量約200天、6個維度，k超過6容易切得太碎、不好解讀，先限制上限
    # ⚠️ 變數命名特別加上 cluster_ 前綴，避免跟 Phase 4.1 的 best_model（模型PK贏家字串）撞名，
    #   上次事故：撞名導致 best_model 從字串被覆蓋成 KMeans 物件本身，寫回py_output時
    #   JSON序列化失敗（TypeError: Object of type KMeans is not JSON serializable）
    best_cluster_k, best_cluster_score, best_cluster_model, best_cluster_labels = None, -1, None, None
    for k in range(2, 7):
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X_cluster)
        if len(set(labels)) < 2:
            continue
        score = silhouette_score(X_cluster, labels)
        print(f"    k={k}: 輪廓係數={score:.3f}")
        if score > best_cluster_score:
            best_cluster_k, best_cluster_score, best_cluster_model, best_cluster_labels = k, score, km, labels

    cluster_features = cat_pivot.copy()
    cluster_features['Cluster'] = best_cluster_labels
    print(f"📊 [4.5] 自動選出最佳群數 k={best_cluster_k}（輪廓係數={best_cluster_score:.3f}，越接近1代表分群品質越好）")

    # 各群輪廓：天數、平均總花費、主要類別，用來幫每個群取一個好理解的名字
    cluster_summary = cluster_features.groupby('Cluster')[cols].mean()
    cluster_summary['TotalMean'] = cluster_summary[cols].sum(axis=1)
    cluster_summary['DayCount'] = cluster_features.groupby('Cluster').size()
    cluster_summary = cluster_summary.sort_values('TotalMean')

    print("📋 [4.5-群集輪廓]（依平均總花費由低到高排序）")
    for cluster_id, row in cluster_summary.iterrows():
        pct = row['DayCount'] / len(cluster_features) * 100
        top_cat = row[cols].idxmax()
        print(f"    群{cluster_id}: {row['DayCount']:.0f}天({pct:.1f}%), 平均花費${row['TotalMean']:.0f}, 主要類別:{top_cat}")

    # 依「平均總花費」高低幫每個群取一個直覺的名字（純粹方便閱讀，非模型輸出）
    sorted_cluster_ids = cluster_summary.index.tolist()
    name_pool = ['省錢日', '一般日', '花費日', '爆買日', '極端日', '其他']
    cluster_name_map = {cid: (name_pool[rank] if rank < len(name_pool) else f"群{cid}")
                         for rank, cid in enumerate(sorted_cluster_ids)}

    # 最近一天（資料最後一天）屬於哪一群
    today_cluster_id = cluster_features['Cluster'].iloc[-1]
    today_cluster_name = cluster_name_map[today_cluster_id]
    today_dominant_category = cluster_summary.loc[today_cluster_id, cols].idxmax()
    print(f"🎯 [4.5] 最近一天的消費模式：{today_cluster_name}（主要花在：{today_dominant_category}）")

    # ==========================================
    # 8. 統一匯出資料庫（拆成四個分頁，各司其職）
    # ==========================================
    taipei_tz = pytz.timezone('Asia/Taipei')
    update_time = datetime.now(taipei_tz).strftime("%Y-%m-%d %H:%M:%S")

    # --- 8.1 py_output：不需平滑、不驅動GAS公式的單一摘要指標 ---
    ws_out = sh.worksheet("py_output")
    export_data = [
        ["model_mae_ewma", final_ewma, update_time],
        ["model_mae_median", final_median, update_time],
        ["model_mae_ridge", final_ridge, update_time],
        ["model_best_winner", best_model, update_time],
        ["model_top_feature", top_feature_name, update_time],
        ["event_lambda_monthly", round(lambda_poisson, 2), update_time],
        ["event_tail_shape_c", round(c, 3), update_time],
        ["event_expected_magnitude", round(expected_magnitude, 0), update_time],
        ["pattern_cluster_k", best_cluster_k, update_time],
        ["pattern_silhouette_score", round(best_cluster_score, 3), update_time],
    ]
    ws_out.update(range_name='A3:C12', values=export_data)
    print("✅ py_output 摘要指標已寫入")

    # --- 8.2 params：驅動GAS每日公式的校準參數，用平滑機制寫入，不覆蓋既有孤兒參數 ---
    param_updates = {
        "大額事件門檻(P95)": event_threshold,
        "基礎消費水位(STL)": current_trend,
        "30天趨勢漂移": trend_30d_drift,
        "週末效應倍率(STL)": seasonal_fold_change,
    }
    sync_params(ws_params, param_updates, alpha=0.3)
    print("✅ params 校準參數已平滑更新")

    # --- 8.3 cluster_centroids：三個群的重心，供GAS每日算距離分類用，直接覆蓋不平滑 ---
    centroid_rows = [["群集", "標籤", "天數", "佔比(%)", "平均總花費"] + cols]
    for cluster_id, row in cluster_summary.iterrows():
        pct = round(row['DayCount'] / len(cluster_features) * 100, 1)
        centroid_rows.append(
            [int(cluster_id), cluster_name_map[cluster_id], int(row['DayCount']), pct, round(row['TotalMean'], 0)]
            + [round(row[c], 0) for c in cols]
        )
    ws_centroids.clear()
    ws_centroids.update(range_name='A1', values=centroid_rows)
    print("✅ cluster_centroids 群心明細已寫入")

    # --- 8.4 model_diagnostics：月報表敘事素材，純資料庫，不驅動任何公式 ---
    diag_rows = [["類別", "項目", "數值/內容", "備註", "更新時間"]]

    diag_rows.append(["模型健檢", "is_low_raw邊際觸發率", f"{marginal_rate_raw:.1%}", "理論應接近25%", update_time])
    diag_rows.append(["模型健檢", "Low_Spend_Streak≥2佔比", f"{streak_rate_2plus:.1%}", "理論預期約6%", update_time])
    diag_rows.append(["模型健檢", "Low_Spend_Streak≥3佔比", f"{streak_rate_3plus:.1%}", "理論預期約1.5%", update_time])

    for feat_name, val in coef_ranked.items():
        direction = "正相關" if val > 0 else "負相關"
        diag_rows.append(["Ridge係數排序", feat_name, round(val, 1), direction, update_time])

    for a, b, v in corr_pairs_sorted[:10]:
        diag_rows.append(["同日類別關聯", f"{a}×{b}", round(v, 3), "", update_time])

    for a, b, lag, r, p in lag_results_sorted[:10]:
        sig = "p<0.05" if p < 0.05 else "不顯著"
        diag_rows.append(["滯後類別關聯", f"{a}(t)→{b}(t+{lag})", round(r, 3), f"p={p:.3f}, {sig}", update_time])

    ws_diag.clear()
    ws_diag.update(range_name='A1', values=diag_rows)
    print("✅ model_diagnostics 完整診斷資料已寫入")

    print("🎉 全模組戰果已成功分流寫入四個分頁！")

if __name__ == "__main__":
    main()
