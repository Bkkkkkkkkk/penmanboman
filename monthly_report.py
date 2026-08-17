import os
import json
import gspread
import pandas as pd
import numpy as np
from datetime import datetime
import pytz
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email.mime.text import MIMEText
from email import encoders

import matplotlib
matplotlib.use('Agg')  # 無頭環境（GitHub Actions）不能開視窗，一定要在import pyplot前設定
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

from statsmodels.tsa.seasonal import STL

# ==========================================
# 中文字型設定
# ==========================================
# ⚠️ GitHub Actions的ubuntu runner預設沒有中文字型，matplotlib畫出來的中文
#   會變成一堆空白方框。這裡會嘗試找系統上第一個可用的CJK字型；
#   如果完全找不到（代表workflow裡沒有安裝字型），會印出警告但不中斷執行——
#   寧可圖表中文顯示不出來，也不要讓整個月報表任務失敗、寄不出信。
#   對應的workflow yaml需要加這一步（跑在pip install之前）：
#     - name: 安裝中文字型
#       run: sudo apt-get update && sudo apt-get install -y fonts-noto-cjk fonts-wqy-zenhei
def setup_chinese_font():
    cjk_candidates = ['Noto Sans CJK JP', 'Noto Sans CJK TC', 'Noto Sans CJK SC',
                       'WenQuanYi Zen Hei', 'Microsoft JhengHei', 'PingFang TC']
    available = {f.name for f in fm.fontManager.ttflist}
    for name in cjk_candidates:
        if name in available:
            plt.rcParams['font.family'] = name
            plt.rcParams['axes.unicode_minus'] = False
            print(f"✅ 中文字型設定成功: {name}")
            return
    print("⚠️ 找不到中文字型，圖表中的中文可能無法正常顯示。"
          "請確認GitHub Actions workflow有安裝 fonts-noto-cjk。")


# ==========================================
# 資料讀取輔助函式
# ==========================================

def read_key_value_sheet(ws, key_col=0, val_col=1):
    """把 py_output / params 這種 key-value 格式的分頁讀成字典"""
    rows = ws.get_all_values()
    result = {}
    for row in rows:
        if len(row) > max(key_col, val_col) and row[key_col]:
            result[row[key_col]] = row[val_col]
    return result


def to_float(value, default=0.0):
    try:
        return float(value)
    except (ValueError, TypeError):
        return default


def read_cluster_centroids(ws):
    """讀 cluster_centroids，回傳 (categories, clusters_df)，clusters_df只含真正的群集列"""
    data = ws.get_all_values()
    if not data:
        return [], pd.DataFrame()
    headers = data[0]
    categories = headers[5:]
    rows = []
    for row in data[1:]:
        if not row or not row[0] or not row[0].lstrip('-').isdigit():
            continue  # 跳過空白列 / SCALER_MEAN / SCALER_STD 這些非群集列
        rows.append(row)
    if not rows:
        return categories, pd.DataFrame()
    df = pd.DataFrame(rows, columns=headers[:len(rows[0])])
    for col in ['天數', '佔比(%)', '平均總花費'] + categories:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    return categories, df


def read_model_diagnostics(ws):
    """讀 model_diagnostics，回傳依「類別」分組的 DataFrame 字典"""
    data = ws.get_all_values()
    if not data:
        return {}
    df = pd.DataFrame(data[1:], columns=data[0])
    return {name: group for name, group in df.groupby('類別')}


# ==========================================
# 資料準備：日常/大額分離 + 每日序列 + STL
# ==========================================
# ⚠️ 技術債提醒：這段跟 main.py（日報深度分析腳本）的 Phase 2.1/4.3 邏輯是重複的
#   （日常/大額分離、STL趨勢分解），因為月報表需要「完整的每日序列」畫趨勢圖，
#   但 main.py 目前只把「最終數字」（如stl_current_trend）寫回py_output，沒有
#   把整條序列存下來。如果未來main.py的分離邏輯有調整，這裡要記得同步修改，
#   不然兩邊的「日常消費」定義會兜不起來。長期可以考慮讓main.py額外把每日序列
#   寫進一個新分頁，這裡直接讀就好，不用重算，但這次先用重算版本快速交付。
def prepare_daily_series(sh):
    ws_raw = sh.worksheet("Raw_Data")
    raw_records = ws_raw.get_all_values()
    df = pd.DataFrame(raw_records[1:], columns=raw_records[0])

    df['Amount'] = df['Amount'].astype(str).str.replace(r'[NT\$,\s]', '', regex=True)
    df['Amount'] = pd.to_numeric(df['Amount'], errors='coerce').fillna(0)
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')

    df_clean = df[(df['Type'] == '支出') & (~df['Category'].isin(['股票', '固定帳單']))].copy()
    event_threshold = np.percentile(df_clean['Amount'], 95)
    habitual_df = df_clean[df_clean['Amount'] <= event_threshold].copy()

    daily_df = habitual_df.groupby(habitual_df['Date'].dt.date)['Amount'].sum().reset_index()
    daily_df['Date'] = pd.to_datetime(daily_df['Date'])
    full_dates = pd.date_range(daily_df['Date'].min(), daily_df['Date'].max())
    daily_df = daily_df.set_index('Date').reindex(full_dates, fill_value=0).reset_index()
    daily_df.rename(columns={'index': 'Date'}, inplace=True)

    ts = daily_df.set_index('Date')['Amount'].asfreq('D').fillna(0)
    stl_result = STL(np.log1p(ts), period=7, robust=True).fit()
    trend = np.expm1(stl_result.trend)

    MIN_CATEGORY_COUNT = 20
    category_counts = habitual_df['Category'].value_counts()
    valid_categories = category_counts[category_counts >= MIN_CATEGORY_COUNT].index.tolist()
    cat_habitual = habitual_df[habitual_df['Category'].isin(valid_categories)].copy()
    cat_pivot = cat_habitual.pivot_table(
        index=cat_habitual['Date'].dt.date, columns='Category', values='Amount',
        aggfunc='sum', fill_value=0
    )
    cat_pivot.index = pd.to_datetime(cat_pivot.index)
    cat_pivot = cat_pivot.reindex(full_dates, fill_value=0)

    return ts, trend, cat_pivot


# ==========================================
# 圖表產出（每個函式回傳存檔路徑）
# ==========================================

def chart_trend(ts, trend, out_dir):
    fig, ax = plt.subplots(figsize=(7, 3.2))
    ax.plot(ts.index, ts.values, color='#c7c5be', linewidth=1, label='每日實際花費')
    ax.plot(trend.index, trend.values, color='#2a78d6', linewidth=2, label='STL基礎趨勢')
    ax.set_title('每日花費與基礎趨勢')
    ax.set_ylabel('金額($)')
    ax.legend(loc='upper left', fontsize=9)
    fig.tight_layout()
    path = os.path.join(out_dir, 'chart_trend.png')
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def chart_model_mae(py_output, out_dir):
    labels = ['EWMA均線', '中位數模型', 'Ridge迴歸']
    values = [to_float(py_output.get('model_mae_ewma')),
              to_float(py_output.get('model_mae_median')),
              to_float(py_output.get('model_mae_ridge'))]
    colors = ['#c7c5be'] * 3
    winner = py_output.get('model_best_winner', '')
    for i, name in enumerate(labels):
        if name == winner:
            colors[i] = '#2a78d6'

    fig, ax = plt.subplots(figsize=(5, 3))
    ax.bar(labels, values, color=colors)
    ax.set_title(f'模型預測誤差比較（贏家：{winner}）')
    ax.set_ylabel('平均絕對誤差 MAE')
    fig.tight_layout()
    path = os.path.join(out_dir, 'chart_mae.png')
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def chart_coefficients(diagnostics, out_dir):
    coef_df = diagnostics.get('Ridge係數排序')
    if coef_df is None or coef_df.empty:
        return None
    coef_df = coef_df.copy()
    coef_df['數值/內容'] = pd.to_numeric(coef_df['數值/內容'], errors='coerce')
    coef_df = coef_df.sort_values('數值/內容')

    colors = ['#d9534f' if v < 0 else '#2a78d6' for v in coef_df['數值/內容']]
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.barh(coef_df['項目'], coef_df['數值/內容'], color=colors)
    ax.axvline(0, color='#898781', linewidth=0.8)
    ax.set_title('預測模型係數排序（標準化後）')
    fig.tight_layout()
    path = os.path.join(out_dir, 'chart_coef.png')
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def chart_correlation_heatmap(cat_pivot, out_dir):
    if cat_pivot.shape[1] < 2:
        return None
    corr = cat_pivot.corr()
    fig, ax = plt.subplots(figsize=(5, 4.2))
    im = ax.imshow(corr.values, cmap='RdBu_r', vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=45, ha='right', fontsize=8)
    ax.set_yticks(range(len(corr.columns)))
    ax.set_yticklabels(corr.columns, fontsize=8)
    for i in range(len(corr.columns)):
        for j in range(len(corr.columns)):
            ax.text(j, i, f"{corr.values[i, j]:.2f}", ha='center', va='center', fontsize=7)
    ax.set_title('類別同日花費關聯係數')
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    path = os.path.join(out_dir, 'chart_corr.png')
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def chart_cluster_pie(categories, clusters_df, out_dir):
    if clusters_df.empty:
        return None
    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    colors = plt.cm.Set2.colors
    ax.pie(
        clusters_df['天數'], labels=clusters_df['標籤'],
        autopct=lambda p: f'{p:.1f}%' if p > 0 else '',
        colors=colors[:len(clusters_df)], startangle=90
    )
    ax.set_title('本月消費模式分布')
    fig.tight_layout()
    path = os.path.join(out_dir, 'chart_pie.png')
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


# ==========================================
# PDF 組裝
# ==========================================

def build_pdf(out_path, report_month, py_output, params, diagnostics, chart_paths):
    from reportlab.lib.pagesizes import A4
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import cm
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont

    # 註冊中文字型給reportlab用（跟matplotlib分開設定，reportlab不會自動抓系統字型）
    # ⚠️ 踩過兩個坑，記錄一下：
    #   1. Noto Sans CJK是OpenType/CFF格式，reportlab的TTFont不支援，會直接報錯
    #      "postscript outlines are not supported"。
    #   2. 改用reportlab內建CID字型（如MSung-Light）雖然不報錯，但實測會產生
    #      亂碼（PDF閱讀器/poppler渲染CID字型時的字符映射跟預期不符，抽取出來的
    #      文字是完全錯誤的中文字，肉眼看甚至像是空白，因為字符對應錯亂）。
    #   最終方案：改用 WenQuanYi Zen Hei（真正的TrueType/glyf格式字型，Ubuntu套件
    #   fonts-wqy-zenhei），直接embed進PDF，實測文字擷取完全正確，不會亂碼。
    #   對應workflow yaml的字型安裝步驟要記得也裝這個：
    #     sudo apt-get install -y fonts-noto-cjk fonts-wqy-zenhei
    cjk_font_paths = [
        '/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc',
        '/usr/share/fonts/truetype/wqy/wqy-microhei.ttc',
    ]
    cjk_font_name = 'Helvetica'  # fallback，中文會顯示不出來但至少不會整份報表崩潰
    for fp in cjk_font_paths:
        if os.path.exists(fp):
            try:
                pdfmetrics.registerFont(TTFont('CJK', fp, subfontIndex=0))
                cjk_font_name = 'CJK'
                break
            except Exception as e:
                print(f"⚠️ 字型註冊失敗 {fp}: {e}")
    if cjk_font_name == 'Helvetica':
        print("⚠️ 找不到可用的中文TrueType字型，PDF內文中文可能無法正常顯示。")

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle('CJKTitle', parent=styles['Title'], fontName=cjk_font_name)
    heading_style = ParagraphStyle('CJKHeading', parent=styles['Heading2'], fontName=cjk_font_name)
    body_style = ParagraphStyle('CJKBody', parent=styles['Normal'], fontName=cjk_font_name, fontSize=10, leading=15)

    doc = SimpleDocTemplate(out_path, pagesize=A4,
                             topMargin=1.5 * cm, bottomMargin=1.5 * cm,
                             leftMargin=2 * cm, rightMargin=2 * cm)
    story = []

    story.append(Paragraph(f"勇者記帳RPG 月度結算報告 — {report_month}", title_style))
    story.append(Spacer(1, 12))

    # --- 區塊1：本月總覽 ---
    story.append(Paragraph("本月總覽", heading_style))
    stl_trend = params.get('基礎消費水位(STL)', 'N/A')
    trend_drift = params.get('30天趨勢漂移', 'N/A')
    event_lambda = py_output.get('event_lambda_monthly', 'N/A')
    top_feature = py_output.get('model_top_feature', 'N/A')

    summary_lines = [
        f"基礎消費水位（STL趨勢）：${stl_trend}，近30天變化：${trend_drift}",
        f"本月最強預測特徵：{top_feature}",
        f"大額事件月頻率：{event_lambda} 次/月",
        f"模型PK贏家：{py_output.get('model_best_winner', 'N/A')}",
    ]
    for line in summary_lines:
        story.append(Paragraph(line, body_style))
    story.append(Spacer(1, 16))

    # --- 區塊2~5：圖表 ---
    chart_titles = {
        'trend': '每日花費與基礎趨勢',
        'mae': '模型健康度',
        'coef': '預測特徵重要性',
        'corr': '類別關聯熱力圖',
        'pie': '消費模式分布',
    }
    for key, title in chart_titles.items():
        path = chart_paths.get(key)
        if not path or not os.path.exists(path):
            continue
        story.append(Paragraph(title, heading_style))
        # 依圖檔實際寬高比動態算顯示高度，避免所有圖表被強制套用同一個框而變形
        # （踩過的坑：圓餅圖原始檔是正方形，套用固定的15cm×9cm框會被拉伸成橢圓）
        from PIL import Image as PILImage
        with PILImage.open(path) as im:
            img_w, img_h = im.size
        display_width = 15 * cm
        display_height = display_width * (img_h / img_w)
        max_height = 18 * cm  # 避免過高的圖表超出頁面太多
        if display_height > max_height:
            display_height = max_height
            display_width = display_height * (img_w / img_h)
        story.append(Image(path, width=display_width, height=display_height))
        story.append(Spacer(1, 12))

    doc.build(story)
    return out_path


# ==========================================
# 寄信
# ==========================================

def send_report_email(pdf_path, report_month):
    sender = os.environ.get("GMAIL_ADDRESS")
    app_password = os.environ.get("GMAIL_APP_PASSWORD")
    recipient = os.environ.get("REPORT_RECIPIENT", sender)

    if not sender or not app_password:
        raise ValueError("缺少 GMAIL_ADDRESS 或 GMAIL_APP_PASSWORD 環境變數（請確認GitHub Secrets有設定）")

    msg = MIMEMultipart()
    msg['From'] = sender
    msg['To'] = recipient
    msg['Subject'] = f"勇者記帳RPG 月度結算報告 - {report_month}"
    msg.attach(MIMEText(f"{report_month} 的月度結算報告已產出，詳見附件PDF。", 'plain'))

    with open(pdf_path, 'rb') as f:
        part = MIMEBase('application', 'octet-stream')
        part.set_payload(f.read())
    encoders.encode_base64(part)
    part.add_header('Content-Disposition', f'attachment; filename="{os.path.basename(pdf_path)}"')
    msg.attach(part)

    with smtplib.SMTP('smtp.gmail.com', 587) as server:
        server.starttls()
        server.login(sender, app_password)
        server.send_message(msg)
    print(f"✅ 月報表已寄出至 {recipient}")


# ==========================================
# 主流程
# ==========================================

def main():
    print("📜 啟動月度結算報告產出...")
    setup_chinese_font()

    sheet_id = os.environ.get("SPREADSHEET_ID")
    sa_json = os.environ.get("GCP_SERVICE_ACCOUNT")
    credentials = json.loads(sa_json)
    gc = gspread.service_account_from_dict(credentials)
    sh = gc.open_by_key(sheet_id)

    py_output = read_key_value_sheet(sh.worksheet("py_output"))
    params = read_key_value_sheet(sh.worksheet("params"))
    categories, clusters_df = read_cluster_centroids(sh.worksheet("cluster_centroids"))
    diagnostics = read_model_diagnostics(sh.worksheet("model_diagnostics"))

    print("📊 重建每日序列與STL趨勢（供繪圖用）...")
    ts, trend, cat_pivot = prepare_daily_series(sh)

    out_dir = "/tmp/monthly_report_charts"
    os.makedirs(out_dir, exist_ok=True)

    print("🖼️ 產出圖表...")
    chart_paths = {
        'trend': chart_trend(ts, trend, out_dir),
        'mae': chart_model_mae(py_output, out_dir),
        'coef': chart_coefficients(diagnostics, out_dir),
        'corr': chart_correlation_heatmap(cat_pivot, out_dir),
        'pie': chart_cluster_pie(categories, clusters_df, out_dir),
    }

    taipei_tz = pytz.timezone('Asia/Taipei')
    report_month = datetime.now(taipei_tz).strftime("%Y年%m月")

    print("📄 組裝PDF...")
    pdf_path = f"/tmp/monthly_report_{datetime.now(taipei_tz).strftime('%Y%m')}.pdf"
    build_pdf(pdf_path, report_month, py_output, params, diagnostics, chart_paths)

    print("📧 寄送報告...")
    send_report_email(pdf_path, report_month)

    print("🎉 月度結算報告流程完成！")


if __name__ == "__main__":
    main()
