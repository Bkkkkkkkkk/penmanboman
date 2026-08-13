import os
import json
import gspread
import requests
import pandas as pd
import numpy as np
import traceback
from datetime import datetime, timedelta
import random

# ==========================================
# ⚙️ 1. 環境變數與連線設定
# ==========================================
SPREADSHEET_ID = os.environ.get("SPREADSHEET_ID")
LINE_TOKEN = os.environ.get("LINE_ACCESS_TOKEN")
LINE_USER_ID = os.environ.get("LINE_USER_ID")
GCP_SA_JSON = os.environ.get("GCP_SERVICE_ACCOUNT")

def init_gspread():
    credentials_dict = json.loads(GCP_SA_JSON)
    gc = gspread.service_account_from_dict(credentials_dict)
    return gc.open_by_key(SPREADSHEET_ID)

def send_line_message(message):
    url = "https://api.line.me/v2/bot/message/push"
    headers = {"Authorization": f"Bearer {LINE_TOKEN}", "Content-Type": "application/json"}
    payload = {"to": LINE_USER_ID, "messages": [{"type": "text", "text": message}]}
    requests.post(url, headers=headers, json=payload)

# ==========================================
# 🚀 2. 時間序列特徵工程與異常偵測
# ==========================================
if __name__ == "__main__":
    try:
        sh = init_gspread()
        ws = sh.worksheet("Raw_Data")
        
        # 1. 資料預處理
        raw_data = ws.get_all_values()
        df = pd.DataFrame(raw_data[1:], columns=raw_data[0])
        df['Amount'] = pd.to_numeric(df['Amount'], errors='coerce').fillna(0)
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        expense_df = df[df['Type'] == '支出'].copy()
        
        # 2. 按日聚合 (Daily Aggregation)
        daily_df = expense_df.groupby(expense_df['Date'].dt.date)['Amount'].sum().reset_index()
        daily_df['Date'] = pd.to_datetime(daily_df['Date'])
        daily_df = daily_df.sort_values('Date').reset_index(drop=True)
        
        # 3. 計算 EWMA (指數加權移動平均) 基準線 (Span=7)
        daily_df['EWMA_7'] = daily_df['Amount'].ewm(span=7, adjust=False).mean()
        
        # 4. 異常偵測 (Rolling Z-score, 觀察過去 30 天)
        daily_df['Rolling_Mean'] = daily_df['Amount'].rolling(window=30, min_periods=3).mean()
        daily_df['Rolling_Std'] = daily_df['Amount'].rolling(window=30, min_periods=3).std().fillna(1)
        daily_df['Z_Score'] = (daily_df['Amount'] - daily_df['Rolling_Mean']) / daily_df['Rolling_Std']
        
        # 5. 抓取「昨日」特徵 (因為日報通常結算已發生的昨日)
        taipei_now = datetime.utcnow() + timedelta(hours=8)
        yesterday = taipei_now.date() - timedelta(days=1)
        
        # 確保有昨日資料，若無則補 0
        if yesterday in daily_df['Date'].dt.date.values:
            target_row = daily_df[daily_df['Date'].dt.date == yesterday].iloc[0]
            y_amount = int(target_row['Amount'])
            y_ewma = int(target_row['EWMA_7'])
            y_zscore = target_row['Z_Score']
        else:
            y_amount, y_ewma, y_zscore = 0, 0, 0
            
        # 6. 星期分桶與嚴格百分位數 (Exclusive Percentile Logic)
        weekday_num = yesterday.weekday()
        weekday_names = ["一", "二", "三", "四", "五", "六", "日"]
        weekday_df = daily_df[daily_df['Date'].dt.weekday == weekday_num]
        
        # 使用 method='lower' 逼近排除式安全水位設計
        if len(weekday_df) > 3:
            q3_limit = int(np.percentile(weekday_df['Amount'], 75, method='lower'))
        else:
            q3_limit = y_ewma # 資料不足時退回 EWMA

        # ==========================================
        # 🛡️ 3. f-string 模板引擎 (完全控制字數與語氣)
        # ==========================================
        # 判斷異常狀態
        is_anomaly = y_zscore > 2.0  # Z-score 大於 2 視為魔力暴走
        deviation = y_amount - y_ewma
        
        if is_anomaly:
            templates = [
                f"[系統警告] 偵測到魔力異常波動！昨日耗散 ${y_amount}，嚴重偏離均線 (Z-score: {y_zscore:.1f})。已突破星期{weekday_names[weekday_num]}安全水位 (${q3_limit})，建議立即啟動隱匿防禦陣型。",
                f"[系統警告] 慾望侵蝕加劇！昨日支出 ${y_amount}，較動態基準 (EWMA) 高出 ${deviation}。異常指數飆高，請收斂非必要開銷，避免防禦崩潰。"
            ]
        elif deviation > 0:
            templates = [
                f"[系統通知] 昨日耗散 ${y_amount}，微幅高於短期基準 ${y_ewma}。星期{weekday_names[weekday_num]}頂標水位為 ${q3_limit}，目前尚在掌控中，請保持警戒。",
                f"[系統通知] 戰略結算：昨日支出 ${y_amount}。動態均線上升中，但未觸發 Z-score 異常警報。繼續維持常態推進。"
            ]
        else:
            templates = [
                f"[系統通知] 完美防禦！昨日僅耗散 ${y_amount}，低於 EWMA 基準 ${y_ewma}。魔力穩定積累，落於星期{weekday_names[weekday_num]}安全區間內。",
                f"[系統通知] 潛行狀態維持良好。昨日支出 ${y_amount}，Z-score 呈現負值穩定。持續擴大資源優勢。"
            ]
            
        # 隨機抽取對應狀態的句型
        final_report = random.choice(templates)
        
        # 發送至 LINE
        send_line_message(f"\n{final_report}")
        print("✅ 深度數據日報推播完畢！")
        
    except Exception as e:
        print(f"❌ 系統發生異常: {e}")
        traceback.print_exc()
