import os
import json
import gspread
import requests
import pandas as pd

# ==========================================
# ⚙️ 1. 讀取 GitHub Secrets 機密環境變數
# ==========================================
SPREADSHEET_ID = os.environ.get("SPREADSHEET_ID")
LINE_TOKEN = os.environ.get("LINE_ACCESS_TOKEN")
LINE_USER_ID = os.environ.get("LINE_USER_ID")
GCP_SA_JSON = os.environ.get("GCP_SERVICE_ACCOUNT")

def init_gspread():
    """使用 JSON 字串初始化 Google Sheets 連線，並透過絕對 ID 開啟試算表"""
    credentials_dict = json.loads(GCP_SA_JSON)
    gc = gspread.service_account_from_dict(credentials_dict)
    return gc.open_by_key(SPREADSHEET_ID)

def send_line_message(message):
    """發送 LINE Messaging API 訊息"""
    url = "https://api.line.me/v2/bot/message/push"
    headers = {
        "Authorization": f"Bearer {LINE_TOKEN}",
        "Content-Type": "application/json"
    }
    payload = {
        "to": LINE_USER_ID,
        "messages": [{"type": "text", "text": message}]
    }
    response = requests.post(url, headers=headers, json=payload)
    if response.status_code == 200:
        print("✅ LINE 推播成功")
    else:
        print(f"❌ LINE 推播失敗: {response.text}")

# ==========================================
# 🚀 2. 主程式：資料庫連線與推播測試
# ==========================================
if __name__ == "__main__":
    try:
        print("🛸 系統啟動：嘗試連線 Google 試算表...")
        sh = init_gspread()
        ws = sh.worksheet("Raw_Data")
        
        # 抓取資料並轉為 Pandas DataFrame
        raw_data = ws.get_all_values()
        df = pd.DataFrame(raw_data[1:], columns=raw_data[0])
        
        # 計算一下總筆數作為驗證
        total_rows = len(df)
        print(f"📊 成功讀取 Raw_Data，共 {total_rows} 筆流水帳。")
        
        # 組合測試訊息
        msg = f"[系統通知]\n後勤數據中心連線成功！\n目前已成功讀取 {total_rows} 筆基礎資料，通訊管線運作正常。"
        send_line_message(msg)
        
    except Exception as e:
        print(f"❌ 系統發生異常: {e}")
