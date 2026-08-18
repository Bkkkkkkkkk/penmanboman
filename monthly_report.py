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


def compute_isweekend_ratio(diagnostics):
    """
    算IsWeekend領先第二名特徵的倍數，用來判斷平假日二分法是否還是最佳切分方式。
    背景：我們討論過，只要這個比值維持在1.5倍以上，代表二分法還是穩贏，不用改；
    低於1.5倍時才需要認真評估要不要做7天分桶。這不是自動觸發警報的規則，
    只是把數字準備好放進月報表，由你自己每月回顧時判斷。
    """
    coef_df = diagnostics.get('Ridge係數排序')
    if coef_df is None or coef_df.empty:
        return None, None, False

    coef_df = coef_df.copy()
    coef_df['abs_val'] = pd.to_numeric(coef_df['數值/內容'], errors='coerce').abs()
    coef_df = coef_df.sort_values('abs_val', ascending=False)

    if len(coef_df) < 2:
        return None, None, False

    top_name = coef_df.iloc[0]['項目']
    top_val = coef_df.iloc[0]['abs_val']
    second_name = coef_df.iloc[1]['項目']
    second_val = coef_df.iloc[1]['abs_val']

    if top_name != 'IsWeekend' or second_val == 0:
        # IsWeekend不是第一名時，這個監控指標本身就不適用，回傳None讓呼叫端知道要跳過
        return None, None, False

    ratio = top_val / second_val
    is_low = ratio < 1.5
    ratio_text = f"IsWeekend領先幅度：{top_val:.1f} / {second_val:.1f}（{second_name}） = {ratio:.2f}倍"
    return ratio_text, ratio, is_low


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
# 基礎財務摘要：上月總收入/總支出/結餘/類別排行/建議
# ==========================================
# ⚠️ 注意：這裡刻意用「完整的Raw_Data」，不像深度分析那樣排除股票/固定帳單/大額事件。
#   「結餘」這種基礎數字要反映真實的錢包狀況，不能只看habitual_df這個為了統計建模
#   而篩選過的子集，兩邊資料範圍不同是刻意的設計，不是不一致的bug。
def get_last_month_range(taipei_tz):
    now = datetime.now(taipei_tz)
    first_day_this_month = now.replace(day=1)
    last_day_last_month = first_day_this_month - pd.Timedelta(days=1)
    first_day_last_month = last_day_last_month.replace(day=1)
    return (pd.Timestamp(first_day_last_month.date()),
            pd.Timestamp(last_day_last_month.date()))


def compute_basic_summary(sh, taipei_tz):
    ws_raw = sh.worksheet("Raw_Data")
    raw_records = ws_raw.get_all_values()
    df = pd.DataFrame(raw_records[1:], columns=raw_records[0])

    df['Amount'] = df['Amount'].astype(str).str.replace(r'[NT\$,\s]', '', regex=True)
    df['Amount'] = pd.to_numeric(df['Amount'], errors='coerce').fillna(0)
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')

    start, end = get_last_month_range(taipei_tz)
    month_df = df[(df['Date'] >= start) & (df['Date'] <= end)].copy()

    total_income = month_df[month_df['Type'] == '收入']['Amount'].sum()
    expense_df = month_df[month_df['Type'] == '支出']
    total_expense = expense_df['Amount'].sum()
    balance = total_income - total_expense

    category_ranking = (
        expense_df.groupby('Category')['Amount'].sum()
        .sort_values(ascending=False)
    )

    # 給「下月目標」用的排行榜要排除股票/固定帳單——這兩類不是可自由調整的日常開銷，
    # 用它們當「挑戰下月減少X%」的目標沒有意義（固定帳單金額通常是合約固定的，
    # 不是靠意志力就能少花）。基礎摘要顯示用的category_ranking維持完整（含這兩類），
    # 因為那是要反映真實花費全貌，跟目標生成的篩選需求不同。
    discretionary_ranking = (
        expense_df[~expense_df['Category'].isin(['股票', '固定帳單'])]
        .groupby('Category')['Amount'].sum()
        .sort_values(ascending=False)
    )

    # 簡單規則式建議（不用AI生成，數字說話，維持跟深度分析模組一致的確定性風格）
    suggestions = []
    if balance < 0:
        suggestions.append(f"⚠️ 本月支出超過收入 ${abs(balance):.0f}，結餘為負，建議檢視是否有非必要開銷可以縮減。")
    else:
        suggestions.append(f"✅ 本月結餘為正 ${balance:.0f}，維持目前的收支節奏。")

    if len(category_ranking) > 0:
        top_cat = category_ranking.index[0]
        top_amount = category_ranking.iloc[0]
        top_share = top_amount / total_expense * 100 if total_expense > 0 else 0
        if top_share > 40:
            suggestions.append(f"「{top_cat}」佔本月總支出 {top_share:.0f}%，是單一最大宗開銷，可留意是否有節省空間。")
        else:
            suggestions.append(f"本月最大支出類別為「{top_cat}」（${top_amount:.0f}），佔比 {top_share:.0f}%，屬於合理分散範圍。")

    return {
        'start': start, 'end': end,
        'total_income': total_income, 'total_expense': total_expense, 'balance': balance,
        'category_ranking': category_ranking,
        'discretionary_ranking': discretionary_ranking,
        'suggestions': suggestions
    }


def chart_category_ranking(category_ranking, out_dir):
    if category_ranking.empty:
        return None
    top_n = category_ranking.head(8).sort_values()
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.barh(top_n.index, top_n.values, color='#2a78d6')
    ax.set_title('本月開銷分類排行')
    ax.set_xlabel('金額($)')
    fig.tight_layout()
    path = os.path.join(out_dir, 'chart_category_ranking.png')
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


# ==========================================
# 下月目標生成：規則式，不用AI，2~3條，寫回 monthly_goals 分頁供前端讀取
# ==========================================
# ⚠️ 架構調整：不再輸出純文字，改成結構化資料（目標類型/目標數值/比較類別/基準值），
#   因為「進度」要由試算表公式每天即時算（Python只在月初跑一次，沒辦法算逐日進度），
#   公式需要拆開的欄位才能運作，純文字沒辦法被公式解析。
def generate_next_month_goals(summary, py_output, params):
    goals = []

    balance = summary['balance']
    if balance >= 0:
        target = round(balance * 0.9, -2) if balance > 0 else 0
        goals.append({
            'type': 'balance_min',
            'desc': f"維持結餘為正，目標下月結餘不低於 ${target:.0f}",
            'target_value': float(target),
            'category': '',
            'baseline_value': '',
        })
    else:
        goals.append({
            'type': 'balance_min',
            'desc': "力求下月結餘轉正，檢視固定支出與非必要開銷",
            'target_value': 0,
            'category': '',
            'baseline_value': '',
        })

    # 用discretionary_ranking（已排除股票/固定帳單），確保「挑戰下月減少X%」這個目標
    # 挑到的是真正可以靠行為調整的日常開銷類別，不會挑到固定帳單這種你無法控制的項目
    category_ranking = summary['discretionary_ranking']
    if len(category_ranking) > 0:
        top_cat = category_ranking.index[0]
        top_amount = category_ranking.iloc[0]
        reduce_target = round(top_amount * 0.1, -1)
        goals.append({
            'type': 'category_reduce',
            'desc': f"挑戰「{top_cat}」類別下月減少 ${reduce_target:.0f}（約10%）",
            'target_value': float(reduce_target),
            'category': top_cat,
            'baseline_value': float(round(top_amount, 0)),
        })

    event_lambda = to_float(py_output.get('event_lambda_monthly'), default=None)
    if event_lambda is not None:
        target_count = round(event_lambda)
        goals.append({
            'type': 'event_count_max',
            'desc': f"留意大額支出頻率，下月目標控制在 {target_count} 次以內",
            'target_value': target_count,
            'category': '',
            'baseline_value': '',
        })

    return goals[:3]  # 最多3條，符合「2~3個當月任務」的需求


def write_goals_to_sheet(sh, goals, goal_month_label):
    """
    寫入 monthly_goals 分頁，分頁不存在會自動建立。
    G/H欄位是試算表公式，用來每天即時算「目前進度」跟「進度百分比」，
    Python只負責定義目標本身（目標類型/數值/比較類別/基準值），不算進度。

    ⚠️ 重要：公式字串要用 value_input_option='USER_ENTERED' 寫入，
    否則gspread預設會把"="開頭的字串當純文字寫入儲存格，不會被試算表當公式解析。
    """
    try:
        ws = sh.worksheet("monthly_goals")
    except gspread.exceptions.WorksheetNotFound:
        ws = sh.add_worksheet(title="monthly_goals", rows=20, cols=10)

    ws.clear()
    taipei_tz = pytz.timezone('Asia/Taipei')
    now_str = datetime.now(taipei_tz).strftime("%Y-%m-%d %H:%M:%S")

    header = ["目標編號", "目標類型", "描述文字", "目標數值", "比較類別", "基準值",
              "適用月份", "建立時間", "目前進度值", "進度百分比"]
    rows = [header]

    for i, goal in enumerate(goals, start=1):
        row_num = i + 1  # 第1列是表頭，資料從第2列開始
        d_col = f"D{row_num}"  # 目標數值
        e_col = f"E{row_num}"  # 比較類別
        f_col = f"F{row_num}"  # 基準值
        b_col = f"B{row_num}"  # 目標類型

        # G欄：目前實際值，依目標類型分三種算法
        g_formula = (
            f'=IFS('
            f'{b_col}="balance_min",'
            f'SUMIFS(db_main!F:F,db_main!C:C,"收入",db_main!B:B,">="&DATE(YEAR(TODAY()),MONTH(TODAY()),1))'
            f'-SUMIFS(db_main!F:F,db_main!C:C,"支出",db_main!B:B,">="&DATE(YEAR(TODAY()),MONTH(TODAY()),1)),'
            f'{b_col}="category_reduce",'
            f'SUMIFS(db_main!F:F,db_main!C:C,"支出",db_main!D:D,{e_col},'
            f'db_main!B:B,">="&DATE(YEAR(TODAY()),MONTH(TODAY()),1)),'
            f'{b_col}="event_count_max",'
            f'COUNTIFS(db_main!C:C,"支出",db_main!F:F,">"&VLOOKUP("大額事件門檻(P95)",params!A:B,2,FALSE),'
            f'db_main!B:B,">="&DATE(YEAR(TODAY()),MONTH(TODAY()),1))'
            f')'
        )

        # H欄：進度百分比。balance_min/event_count_max是「目前值/目標值」，
        # category_reduce是「目前花費/(基準值-目標減少金額)」這個預算上限，
        # 用IFERROR包起來，避免目標數值是0時除以0出錯導致整欄崩潰
        # ⚠️ 分子要引用I欄（目前進度值實際落地的欄位），不是G欄（適用月份，一個文字/日期值）。
        #   上一版這裡誤植成G{row_num}，導致除以一個日期序號，算出離譜的巨大百分比
        #   （例如770583.33%）；balance_min那列因為有MIN(...,1)封頂在100%，剛好蓋住了
        #   同樣的錯誤沒被發現。這次改用I欄，並且已經實際模擬驗算過三種類型都正確。
        i_col = f"I{row_num}"  # 目前進度值（g_formula 寫入的欄位）
        h_formula = (
            f'=IFERROR(IFS('
            f'{b_col}="balance_min",MIN({i_col}/{d_col},1),'
            f'{b_col}="category_reduce",{i_col}/({f_col}-{d_col}),'
            f'{b_col}="event_count_max",{i_col}/{d_col}'
            f'),0)'
        )

        rows.append([
            i, goal['type'], goal['desc'], goal['target_value'],
            goal['category'], goal['baseline_value'],
            goal_month_label, now_str, g_formula, h_formula
        ])

    ws.update(range_name='A1', values=rows, value_input_option='USER_ENTERED')
    print(f"✅ monthly_goals 已寫入 {len(goals)} 條下月目標（含每日進度公式）")


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

def build_pdf(out_path, report_month, py_output, params, diagnostics, chart_paths,
              isweekend_ratio_info, basic_summary, goals):
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

    # --- 區塊0：基礎財務摘要（新增） ---
    story.append(Paragraph("本月基礎財務摘要", heading_style))
    bs = basic_summary
    financial_lines = [
        f"統計區間：{bs['start'].strftime('%Y-%m-%d')} ~ {bs['end'].strftime('%Y-%m-%d')}",
        f"總收入：${bs['total_income']:.0f}　總支出：${bs['total_expense']:.0f}　結餘：${bs['balance']:.0f}",
    ]
    for line in financial_lines:
        story.append(Paragraph(line, body_style))
    for s in bs['suggestions']:
        story.append(Paragraph(s, body_style))
    story.append(Spacer(1, 12))

    # --- 開銷分類排行圖 ---
    cat_chart = chart_paths.get('category_ranking')
    if cat_chart and os.path.exists(cat_chart):
        story.append(Paragraph("開銷分類排行", heading_style))
        from PIL import Image as PILImage
        with PILImage.open(cat_chart) as im:
            img_w, img_h = im.size
        display_width = 15 * cm
        display_height = display_width * (img_h / img_w)
        story.append(Image(cat_chart, width=display_width, height=display_height))
        story.append(Spacer(1, 16))

    # --- 下月目標（新增） ---
    if goals:
        story.append(Paragraph("下月目標", heading_style))
        for i, goal in enumerate(goals, start=1):
            story.append(Paragraph(f"{i}. {goal['desc']}", body_style))
        story.append(Spacer(1, 16))

    # --- 區塊1：本月總覽（原有，加上IsWeekend領先幅度提醒） ---
    story.append(Paragraph("模型分析總覽", heading_style))
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

    # IsWeekend領先幅度提醒（新增）：只在IsWeekend確實是第一名時才顯示。
    # ⚠️ 說明文字這次改成「不管數字高低都要顯示」，之前的版本只在低於1.5倍時
    #   才附註解釋，但這樣過幾個月你看到單獨一個「1.81倍」的數字，早就忘記
    #   這代表什麼、門檻是多少——提醒的重點是「持續提供脈絡」，不是「只在異常時才說明」。
    ratio_text, ratio_value, is_low = isweekend_ratio_info
    if ratio_text:
        story.append(Paragraph(ratio_text, body_style))
        story.append(Paragraph(
            "（提醒：這個比值代表平假日二分法領先第二名特徵的倍數，"
            "維持1.5倍以上代表二分法仍是最佳切分方式；低於1.5倍時，可考慮評估7天分桶的必要性）",
            body_style))
        if is_low:
            story.append(Paragraph(
                "⚠️ 目前已低於1.5倍門檻，建議這個月認真評估是否要改成7天分桶。",
                body_style))
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

    taipei_tz = pytz.timezone('Asia/Taipei')

    print("💰 計算上月基礎財務摘要...")
    basic_summary = compute_basic_summary(sh, taipei_tz)

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
        'category_ranking': chart_category_ranking(basic_summary['category_ranking'], out_dir),
    }

    print("📐 計算IsWeekend領先幅度...")
    isweekend_ratio_info = compute_isweekend_ratio(diagnostics)

    print("🎯 產生下月目標...")
    goals = generate_next_month_goals(basic_summary, py_output, params)
    goal_month_label = datetime.now(taipei_tz).strftime("%Y年%m月")
    write_goals_to_sheet(sh, goals, goal_month_label)

    report_month = datetime.now(taipei_tz).strftime("%Y年%m月")

    print("📄 組裝PDF...")
    pdf_path = f"/tmp/monthly_report_{datetime.now(taipei_tz).strftime('%Y%m')}.pdf"
    build_pdf(pdf_path, report_month, py_output, params, diagnostics, chart_paths,
              isweekend_ratio_info, basic_summary, goals)

    print("📧 寄送報告...")
    send_report_email(pdf_path, report_month)

    print("🎉 月度結算報告流程完成！")


if __name__ == "__main__":
    main()
