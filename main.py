import os
import json
import gspread
from datetime import datetime
import pytz

def main():
    print("🚀 啟動系統連線測試...")
    
    # 1. 讀取環境變數中的金鑰
    sheet_id = os.environ.get("SPREADSHEET_ID")
    sa_json = os.environ.get("GCP_SERVICE_ACCOUNT")
    
    if not sheet_id or not sa_json:
        raise ValueError("❌ 找不到環境變數 SPREADSHEET_ID 或 GCP_SERVICE_ACCOUNT")

    # 2. 登入 Google Sheets
    credentials = json.loads(sa_json)
    gc = gspread.service_account_from_dict(credentials)
    sh = gc.open_by_key(sheet_id)
    
    # 3. 讀取 Raw_Data (確保讀取權限正常)
    ws_raw = sh.worksheet("Raw_Data")
    raw_records = ws_raw.get_all_values()
    data_length = len(raw_records) - 1 # 扣除標題列
    print(f"📊 成功讀取 Raw_Data，目前共有 {data_length} 筆資料。")

    # 4. 寫入 py_output (確保寫入權限與格式正常)
    ws_out = sh.worksheet("py_output")
    taipei_tz = pytz.timezone('Asia/Taipei')
    update_time = datetime.now(taipei_tz).strftime("%Y-%m-%d %H:%M:%S")
    
    # 依照 Phase 0 規格：metric_key | value | 更新時間
    test_row = ["system_status", f"Python 管線連線成功 (資料數: {data_length})", update_time]
    
    # 將測試結果寫回
    ws_out.insert_row(test_row, index=2)
    print("✅ 測試數據已成功寫回 py_output 分頁！")

if __name__ == "__main__":
    main()
