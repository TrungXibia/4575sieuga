import streamlit as st
import pandas as pd
import requests
import json
from collections import Counter, defaultdict
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import plotly.graph_objects as go

# ---- CẤU HÌNH TRANG ----
st.set_page_config(
    page_title="Thống Kê XSMB Online",
    page_icon="🎲",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ---- CONSTANTS & SESSION ----
HEADERS = {
    "User-Agent": "Mozilla/5.0",
    "Accept": "application/json, text/plain, */*",
    "Referer": "https://www.kqxs88.live/",
}

API_OPTIONS = {
    "Miền Bắc 45s": "https://www.kqxs88.live/api/front/open/lottery/history/list/game?limitNum=50&gameCode=miba45",
    "Miền Bắc 75s": "https://www.kqxs88.live/api/front/open/lottery/history/list/game?limitNum=50&gameCode=mbmg",
    "Miền Bắc": "https://www.kqxs88.live/api/front/open/lottery/history/list/game?limitNum=50&gameCode=miba"
}

API_200_OPTIONS = {
    "Miền Bắc 75s": "https://www.kqxs88.live/api/front/open/lottery/history/list/game?limitNum=200&gameCode=mbmg",
    "Miền Bắc 45s": "https://www.kqxs88.live/api/front/open/lottery/history/list/game?limitNum=200&gameCode=miba45",
    "Miền Bắc": "https://www.kqxs88.live/api/front/open/lottery/history/list/game?limitNum=200&gameCode=miba"
}

GIAI_LABELS = [
    "ĐB", "G1", "G2-1", "G2-2",
    "G3-1", "G3-2", "G3-3", "G3-4", "G3-5", "G3-6",
    "G4-1", "G4-2", "G4-3", "G4-4",
    "G5-1", "G5-2", "G5-3", "G5-4", "G5-5", "G5-6",
    "G6-1", "G6-2", "G6-3",
    "G7-1", "G7-2", "G7-3", "G7-4"
]

# ---- HÀM HỖ TRỢ (UTILS) ----

@st.cache_resource
def get_session():
    s = requests.Session()
    retry = Retry(
        total=3, connect=3, read=3,
        backoff_factor=0.5,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=frozenset(["GET"]),
    )
    s.mount("https://", HTTPAdapter(max_retries=retry))
    s.mount("http://", HTTPAdapter(max_retries=retry))
    return s

SESSION = get_session()

@st.cache_data(ttl=60)  # Cache dữ liệu trong 60 giây
def fetch_data(url):
    try:
        resp = SESSION.get(url, headers=HEADERS, timeout=10)
        resp.raise_for_status()
        return resp.json().get("t", {}).get("issueList", [])
    except Exception as e:
        st.error(f"Lỗi kết nối: {e}")
        return []

def get_nhi_hop_cham_tong(list_0):
    if isinstance(list_0, str):
        list_0 = [x.strip() for x in list_0.split(',') if x.strip().isdigit()]
    elif isinstance(list_0, list):
        list_0 = [str(x) for x in list_0]
        
    result = []
    for i in range(100):
        so = f"{i:02d}"
        a, b = int(so[0]), int(so[1])
        if str(a) in list_0 or str(b) in list_0 or str((a+b)%10) in list_0:
            result.append(so)
    return ' '.join(result)

def parse_prizes(detail_str):
    detail = json.loads(detail_str)
    prizes = []
    for field in detail:
        prizes += field.split(",")
    return (prizes + [""] * 27)[:27]

# ---- TAB 1: KQ 45s/75s ----

def render_tab_kq_short():
    st.header("Thống kê Kết quả 45s / 75s / MB")
    
    # Controls
    c1, c2, c3, c4 = st.columns([2, 1, 1, 2])
    with c1:
        api_key = st.selectbox("Chọn loại xổ số:", list(API_OPTIONS.keys()), key="tab1_api")
    with c2:
        topn = st.number_input("TopN:", min_value=1, max_value=10, value=2)
    with c3:
        pass # Spacer
    with c4:
        tk_type = st.selectbox("Kiểu thống kê:", 
            ["Tất cả các giải", "Thiếu số ở Giải", "Chỉ dàn lô KQ Lẻ MB", "Chữ số đầu các giải"])

    # Conditional inputs for "Thiếu số ở Giải"
    selected_giais = []
    if tk_type == "Thiếu số ở Giải":
        st.write("Chọn giải để thống kê thiếu số:")
        cols = st.columns(9)
        for idx, label in enumerate(GIAI_LABELS):
            # Mặc định chọn G1, G2-1, G2-2 như code cũ
            default_val = True if idx in [1, 2, 3] else False
            with cols[idx % 9]:
                if st.checkbox(label, value=default_val, key=f"chk_{idx}"):
                    selected_giais.append(idx)

    if st.button("Cập nhật dữ liệu", key="btn_update_1"):
        st.rerun()

    # Fetch data
    data = fetch_data(API_OPTIONS[api_key])
    if not data:
        return

    # Process Data for Main Table
    table_rows = []
    freq_rows = []
    
    # Prepare Top1 calculation containers
    all_details = []

    for item in data:
        detail_json = json.loads(item['detail'])
        all_details.append(detail_json)
        
        prizes = parse_prizes(item['detail'])
        row = {"Kỳ": item['turnNum']}
        for idx, prize in enumerate(prizes):
            row[GIAI_LABELS[idx]] = prize
        table_rows.append(row)

        # Frequency Logic
        counter = Counter()
        if tk_type == "Tất cả các giải":
            all_nums = ''.join(prizes)
            counter = Counter(all_nums)
        elif tk_type == "Thiếu số ở Giải":
            g_numbers = []
            for idx in selected_giais:
                if idx < len(prizes):
                    g_numbers.extend([ch for ch in prizes[idx] if ch.isdigit()])
            counter = Counter(g_numbers)
        elif tk_type == "Chỉ dàn lô KQ Lẻ MB":
            los = [lo.strip()[-2:] for lo in prizes if len(lo.strip()) >= 2]
            counter = Counter(''.join(los))
        elif tk_type == "Chữ số đầu các giải":
            first_digits = [p.strip()[0] for p in prizes if p.strip()]
            counter = Counter(first_digits)

        counts = [counter.get(str(d), 0) for d in range(10)]
        
        # TopN formatting
        if topn >= 10:
            show = [str(v) for v in counts]
        else:
            temp = sorted(set(counts), reverse=True)
            topn_vals = set(temp[:topn])
            show = [str(v) if v in topn_vals and v > 0 else ("0" if v == 0 else "") for v in counts]
        
        summary = ','.join(str(i) for i, v in enumerate(show) if v and v != "0")
        list0 = [str(i) for i, v in enumerate(counts) if v == 0]
        nhi_hop = get_nhi_hop_cham_tong(list0)
        
        f_row = {str(i): show[i] for i in range(10)}
        f_row["Tổng hợp"] = summary
        f_row["List 0"] = ','.join(list0)
        f_row["Nhị hợp chạm-tổng"] = nhi_hop
        freq_rows.append(f_row)

    # Display Tables
    df_results = pd.DataFrame(table_rows)
    df_freq = pd.DataFrame(freq_rows)
    
    # Styling columns
    st.markdown("### Bảng kết quả & Tần suất")
    c_res, c_freq = st.columns([2, 1])
    
    with c_res:
        st.dataframe(df_results, hide_index=True, height=500, use_container_width=True)
    with c_freq:
        st.dataframe(df_freq, hide_index=True, height=500, use_container_width=True)

    # Analysis: Dàn lô 2 số (Đầu/Đuôi)
    st.markdown("---")
    st.subheader("Phân tích Đầu/Đuôi Top 1 (3 kỳ gần nhất)")
    
    col_dau, col_duoi = st.columns(2)
    
    def get_lo_dict(detail_list, mode='dau'):
        lo_dict = {str(i): [] for i in range(10)}
        for field in detail_list:
            for num in field.split(','):
                if len(num) >= 2:
                    val = num[-2] if mode == 'dau' else num[-1]
                    lo = num[-2:]
                    if val in lo_dict:
                        lo_dict[val].append(lo)
        return lo_dict

    def analyze_top1(container, title, mode):
        top1_str = ""
        with container:
            st.markdown(f"**{title}**")
            txt_area = ""
            for i in range(min(3, len(data))):
                turn = data[i]['turnNum']
                detail = all_details[i]
                lo_dict = get_lo_dict(detail, mode)
                
                # Find top 1
                lens = {k: len(v) for k, v in lo_dict.items()}
                max_len = max(lens.values()) if lens else 0
                top1 = [k for k, v in lens.items() if v == max_len and max_len > 0]
                
                if i == 0:
                    top1_str = "".join(top1)

                txt_area += f"Kỳ {turn}:\n"
                for d in range(10):
                    d_str = str(d)
                    line = f"{d_str}: {', '.join(lo_dict[d_str])}" if lo_dict[d_str] else f"{d_str}:"
                    marker = " 🔥" if i == 0 and d_str in top1 else ""
                    txt_area += f"{line}{marker}\n"
                txt_area += "-"*20 + "\n"
            
            st.text_area(f"Chi tiết {mode}", txt_area, height=300)
            st.error(f"Top 1 {mode} kỳ mới nhất: {top1_str}")

    analyze_top1(col_dau, "Dàn lô theo ĐẦU", 'dau')
    analyze_top1(col_duoi, "Dàn lô theo ĐUÔI", 'duoi')

# ---- TAB 2: KQ 200 KỲ ----

def render_tab_kq_200():
    st.header("Thống kê 200 kỳ - Bạc nhớ & Thiếu")
    
    c1, c2 = st.columns([2, 6])
    with c1:
        api_key = st.selectbox("Chọn đài:", list(API_200_OPTIONS.keys()), key="tab2_api")
    with c2:
        st.info("Dữ liệu tự động cập nhật lô ra theo đầu/đuôi thiếu và bạc nhớ.")

    data = fetch_data(API_200_OPTIONS[api_key])
    if not data:
        return

    # Bac nho data logic
    BN_DUOI = {'0': ['02','04','09'], '1': ['10','12','17'], '2': ['21','23'], '3': ['33','37','38'],
               '4': ['41','42','46'], '5': ['50','51','52','56','57'], '6': ['61','66','68','69'],
               '7': ['73','74','75','78','79'], '8': ['82','83','86'], '9': ['90','91','92','94']}
    BN_DAU = {'0': ['10','20','50','90'], '1': ['21','41','51','91'], '2': ['12','42','62','82','92'],
              '3': ['23','33','73','83'], '4': ['04','74','94'], '5': ['25','35','65','75'],
              '6': ['46','66','86','96'], '7': ['17','27','57','87','97'], '8': ['08','68','78'],
              '9': ['09','19','69','79','89']}

    rows = []
    all_los_collection = [] # For frequency table

    # First Pass: Generate basic data
    for item in data:
        detail = json.loads(item['detail'])
        prizes = parse_prizes(item['detail'])
        
        los = [p.strip()[-2:] for p in prizes if len(p.strip()) >= 2 and p.strip()[-2:].isdigit()]
        all_los_collection.extend(los)
        
        all_daus = set(l[0] for l in los)
        dau_thieu = [str(d) for d in range(10) if str(d) not in all_daus]
        
        all_duois = set(l[1] for l in los)
        duoi_thieu = [str(d) for d in range(10) if str(d) not in all_duois]
        
        # Calc Bac nho
        bn_preds = []
        for d in dau_thieu: bn_preds.extend(BN_DAU.get(d, []))
        for d in duoi_thieu: bn_preds.extend(BN_DUOI.get(d, []))
        bn_preds = sorted(list(set(bn_preds)))

        row = {
            "Kỳ": item['turnNum'],
            "Dàn lô": " ".join(los),
            "Đầu thiếu": ",".join(dau_thieu),
            "Đuôi thiếu": ",".join(duoi_thieu),
            "Bạc nhớ": " ".join(bn_preds),
            "LoList": los, # Hidden for calculation
            **{GIAI_LABELS[i]: prizes[i] for i in range(27)}
        }
        rows.append(row)

    # Second Pass: Calculate "Lô ra theo đầu/đuôi thiếu" (Look ahead)
    # row[i] refers to current issue. We need to see if row[i]'s missing numbers appeared in row[i-1] (next issue in time)
    # Note: data is sorted NEWEST first (index 0). So previous issue is index+1.
    # WAIT: Logic in original code:
    # "next_row = all_rows[idx - 1]" (index 0 is newest).
    # If index 0 is today, index 1 is yesterday.
    # Logic: What appeared TODAY based on YESTERDAY's missing?
    # So row[i] (Today) should show what hit based on row[i+1] (Yesterday)'s missing?
    # Original code:
    # for idx... if idx==0: continue
    # next_row = all_rows[idx-1] -> This implies looking at NEWER issue.
    # Let's trace:
    # Row 0 (Today): Missing A.
    # Row 1 (Yesterday): Missing B.
    # Original: `lo_ra_theo_dau_thieu` at Row 1 checks results of Row 0 (Newer) against Row 1's missing.
    
    for i in range(len(rows)):
        if i == 0:
            rows[i]["Lô ra (Đầu thiếu)"] = ""
            rows[i]["Lô ra (Đuôi thiếu)"] = ""
            continue
        
        # Current row is `i` (Older). We want to see result in `i-1` (Newer)
        res_newer = rows[i-1]["LoList"]
        
        # Check dau thieu of current row `i`
        dau_thieu = rows[i]["Đầu thiếu"].split(",") if rows[i]["Đầu thiếu"] else []
        hits_dau = [lo for lo in res_newer if lo[0] in dau_thieu and lo]
        rows[i]["Lô ra (Đầu thiếu)"] = " ".join(sorted(set(hits_dau)))
        
        # Check duoi thieu of current row `i`
        duoi_thieu = rows[i]["Đuôi thiếu"].split(",") if rows[i]["Đuôi thiếu"] else []
        hits_duoi = [lo for lo in res_newer if lo[1] in duoi_thieu and lo]
        rows[i]["Lô ra (Đuôi thiếu)"] = " ".join(sorted(set(hits_duoi)))

    # Clean up hidden column
    for r in rows: del r["LoList"]

    df = pd.DataFrame(rows)
    # Reorder columns
    main_cols = ["Kỳ"] + GIAI_LABELS + ["Dàn lô", "Đầu thiếu", "Đuôi thiếu", "Lô ra (Đầu thiếu)", "Lô ra (Đuôi thiếu)", "Bạc nhớ"]
    df = df[main_cols]

    st.dataframe(df, height=600, use_container_width=True, hide_index=True)

    # Frequency Tables
    st.subheader("Tần suất Lô theo Đầu/Đuôi thiếu (Toàn bộ 200 kỳ)")
    
    freq_dau = defaultdict(list)
    freq_duoi = defaultdict(list)
    ctr = Counter(all_los_collection)
    
    for lo, count in ctr.items():
        freq_dau[lo[0]].append((lo, count))
        freq_duoi[lo[1]].append((lo, count))
        
    c_fdau, c_fduoi = st.columns(2)
    
    with c_fdau:
        st.write("Theo Đuôi Thiếu (nhóm theo đầu lô)")
        data_dau = []
        for d in range(10):
            s_d = str(d)
            items = sorted(freq_dau[s_d], key=lambda x: -x[1])
            txt = "  ".join([x[0] for x in items])
            data_dau.append({"Đầu": s_d, "Lô (giảm dần tần suất)": txt})
        st.table(pd.DataFrame(data_dau))

    with c_fduoi:
        st.write("Theo Đầu Thiếu (nhóm theo đuôi lô)")
        data_duoi = []
        for d in range(10):
            s_d = str(d)
            items = sorted(freq_duoi[s_d], key=lambda x: -x[1])
            txt = "  ".join([x[0] for x in items])
            data_duoi.append({"Đuôi": s_d, "Lô (giảm dần tần suất)": txt})
        st.table(pd.DataFrame(data_duoi))

# ---- TAB 3: BIỂU ĐỒ ĐẦU/ĐUÔI ----

def render_tab_charts():
    st.header("Biểu đồ & Bạc nhớ Đầu/Đuôi Top 1")
    
    c1, c2, c3, c4 = st.columns(4)
    with c1: api_key = st.selectbox("Chọn đài:", list(API_OPTIONS.keys()), key="tab3_api")
    with c2: num_kys = st.selectbox("Số kỳ hiển thị:", [10, 20, 30, 50], index=1)
    with c3: n_bacnho = st.selectbox("Số kỳ bạc nhớ:", [2, 3, 5, 10, 20], index=0)
    with c4: start_idx = st.selectbox("Lùi cửa sổ:", [0, 1, 2, 3, 5], index=0)
    
    data = fetch_data(API_OPTIONS[api_key])
    if not data: return
    
    # Prepare data (Top 1 Head/Tail per issue)
    chart_data = [] # [{"Issue":.., "HeadTop1": [..], "TailTop1": [..]}]
    
    # Logic: We need the top 1 appearing head/tail for each issue
    processed_data = [] # Store for bac nho calc
    
    # Process from newest to oldest, but for chart we want oldest -> newest
    subset = data[:num_kys + start_idx + 50] # Fetch extra for calculation
    subset_rev = subset[::-1] # Oldest first
    
    for item in subset_rev:
        detail = json.loads(item['detail'])
        los = []
        for field in detail:
            for num in field.split(','):
                if len(num.strip()) >= 2: los.append(num.strip()[-2:])
        
        c_dau = Counter([l[0] for l in los])
        c_duoi = Counter([l[1] for l in los])
        
        max_d = max(c_dau.values()) if c_dau else 0
        top_dau = [int(k) for k,v in c_dau.items() if v == max_d and max_d > 0]
        
        max_u = max(c_duoi.values()) if c_duoi else 0
        top_duoi = [int(k) for k,v in c_duoi.items() if v == max_u and max_u > 0]
        
        processed_data.append({
            "turn": item['turnNum'],
            "dau": top_dau,
            "duoi": top_duoi
        })

    # Slice for chart
    chart_slice = processed_data[-(num_kys+start_idx) : ]
    if start_idx > 0:
        chart_slice = chart_slice[:-start_idx]
    
    # --- Draw Charts with Plotly ---
    def draw_plotly(data_list, key_type, title):
        fig = go.Figure()
        
        x_vals = []
        y_vals = []
        text_vals = []
        
        for i, item in enumerate(data_list):
            for val in item[key_type]:
                x_vals.append(item['turn'])
                y_vals.append(val)
        
        fig.add_trace(go.Scatter(
            x=x_vals, y=y_vals,
            mode='markers+lines',
            marker=dict(size=10, color='#127fd6'),
            line=dict(width=1, color='rgba(18, 127, 214, 0.3)'),
            name=title
        ))
        
        fig.update_layout(
            title=title,
            yaxis=dict(tickmode='linear', tick0=0, dtick=1, range=[-0.5, 9.5]),
            xaxis=dict(type='category'),
            height=350,
            margin=dict(l=20, r=20, t=40, b=20)
        )
        st.plotly_chart(fig, use_container_width=True)

    c_chart1, c_chart2 = st.columns(2)
    with c_chart1:
        draw_plotly(chart_slice, 'dau', "Biểu đồ ĐẦU Top 1")
    with c_chart2:
        draw_plotly(chart_slice, 'duoi', "Biểu đồ ĐUÔI Top 1")
        
    # --- Logic Bạc nhớ ---
    # Calculate generic transition matrix from entire history available
    def calc_transition(raw_data, type_key):
        # raw_data is Oldest -> Newest
        matrix = {i: Counter() for i in range(10)}
        for i in range(len(raw_data) - 1):
            curr_tops = raw_data[i][type_key]
            next_tops = raw_data[i+1][type_key]
            for c in curr_tops:
                for n in next_tops:
                    matrix[c][n] += 1
        return matrix

    # Use larger dataset for learning
    learn_data = processed_data[:-(start_idx)] if start_idx > 0 else processed_data
    mat_dau = calc_transition(learn_data, 'dau')
    mat_duoi = calc_transition(learn_data, 'duoi')

    # Analyze Current Issue (The last one in the visible chart)
    last_issue = chart_slice[-1]
    
    def get_predictions(matrix, current_tops, n_top=3):
        preds = set()
        for val in current_tops:
            ctr = matrix[val]
            most_common = [k for k, v in ctr.most_common() if v > 0]
            # Get max occurrences logic similar to original
            if most_common:
                max_freq = ctr[most_common[0]]
                tops = [k for k, v in ctr.items() if v == max_freq]
                preds.update(tops)
        return sorted(list(preds))

    pred_dau = get_predictions(mat_dau, last_issue['dau'])
    pred_duoi = get_predictions(mat_duoi, last_issue['duoi'])

    st.success(f"**Nhận định kỳ tiếp theo (dựa trên kỳ {last_issue['turn']}):**")
    c_txt1, c_txt2 = st.columns(2)
    with c_txt1:
        st.info(f"ĐẦU Top 1 vừa ra: {last_issue['dau']}")
        st.write(f"Dự đoán ĐẦU kế tiếp: **{', '.join(map(str, pred_dau))}**")
        with st.expander("Chi tiết bạc nhớ ĐẦU"):
            st.json({k: dict(v) for k,v in mat_dau.items() if k in last_issue['dau']})
    with c_txt2:
        st.info(f"ĐUÔI Top 1 vừa ra: {last_issue['duoi']}")
        st.write(f"Dự đoán ĐUÔI kế tiếp: **{', '.join(map(str, pred_duoi))}**")
        with st.expander("Chi tiết bạc nhớ ĐUÔI"):
            st.json({k: dict(v) for k,v in mat_duoi.items() if k in last_issue['duoi']})

# ---- TAB 4: VẼ CẦU VỊ TRÍ ----

def render_tab_cau_vitri():
    st.header("Soi Cầu Vị Trí (G0-G7)")
    
    col1, col2, col3, col4 = st.columns([2, 2, 1, 1])
    with col1: api_key = st.selectbox("Chọn đài:", list(API_OPTIONS.keys()), key="tab4_api")
    with col2: mode = st.radio("Chế độ:", ["nhiều nháy", "lô cặp", "nhập tay"], horizontal=True)
    with col3: 
        manual_input = ""
        if mode == "nhập tay":
            manual_input = st.text_input("Nhập lô (vd: 68):", max_chars=2)
    with col4: st.write("")

    # Fetch
    data = fetch_data(API_OPTIONS[api_key])
    if not data or len(data) < 2:
        st.warning("Chưa đủ dữ liệu.")
        return

    # Constants for position mapping
    # Giai structure: DB(1x5), G1(1x5), G2(2x5), G3(6x5), G4(4x4), G5(6x4), G6(3x3), G7(4x2)
    # Total Chars = 5+5 + 10 + 30 + 16 + 24 + 9 + 8 = 107 chars. Indices 0-106.
    # Logic: Flatten detail string into a list of 107 chars.
    
    def flatten_data(item):
        detail = json.loads(item['detail'])
        chars = []
        for field in detail:
            for sub in field.split(','):
                chars.extend(list(sub.strip()))
        return chars

    # Get Today and Yesterday (Newest is index 0)
    # Note: To check "Bridge" (Cầu), we look at Yesterday's positions that created Today's result.
    today = data[0]
    yesterday = data[1]
    
    chars_today = flatten_data(today)
    chars_yesterday = flatten_data(yesterday)
    
    # Find target Lotteries in TODAY's result
    prizes_today = parse_prizes(today['detail'])
    los_today = [p[-2:] for p in prizes_today if len(p)>=2 and p[-2:].isdigit()]
    
    targets = []
    if mode == "nhiều nháy":
        ctr = Counter(los_today)
        targets = [k for k,v in ctr.items() if v >= 2]
    elif mode == "lô cặp":
        s_los = set(los_today)
        targets = []
        seen = set()
        for l in los_today:
            rev = l[::-1]
            if rev in s_los and l != rev and l not in seen:
                targets.append(l)
                seen.add(l)
                seen.add(rev)
    elif mode == "nhập tay":
        if len(manual_input) == 2: targets = [manual_input]

    # Finding Bridge:
    # A "Bridge" at position X, Y means: chars_yesterday[X] + chars_yesterday[Y] == One of the Targets.
    # And that Target appeared Today.
    
    # Actually, the logic in the Python script is:
    # 1. Identify target numbers (Winning numbers today).
    # 2. Find where those numbers appeared in YESTERDAY's result list (as Lo pairs). 
    # 3. Then check what number is at that position TODAY.
    # Wait, let's re-read the original `do_cau` logic.
    # `los_today`: List of pairs.
    # `los_yest`: List of pairs constructed from specific positions in the flattened array.
    # Original script: "Lô {lo} hôm qua ở vị trí giải thứ {vt+1} => hôm nay là {homnay}"
    # It implies: Look for `lo` (target) in Yesterday's LO LIST.
    # If found at index `idx`, look at index `idx` in Today's LO LIST.
    
    # Re-implementing extract_los_from_row logic
    # The original script maps flattened chars to specific Lo indices based on Giai structure.
    # Hardcoding positions (start, end) for 27 prizes is tricky.
    # Easier approach: Just use the standard parsed prizes list (27 prizes).
    
    def get_27_los(item_detail):
        prizes = parse_prizes(item_detail)
        # Ensure we get 27 prizes, filter strictly
        clean_los = []
        for p in prizes:
            p = p.strip()
            if len(p) >= 2: clean_los.append(p[-2:])
            else: clean_los.append("??")
        return clean_los

    vec_today = get_27_los(today['detail'])
    vec_yesterday = get_27_los(yesterday['detail'])

    st.markdown(f"**Kết quả dò cầu (Kỳ {today['turnNum']} so với {yesterday['turnNum']})**")
    
    if not targets:
        st.info("Không có lô nào thỏa mãn điều kiện lọc.")
    else:
        results_txt = []
        for t in targets:
            # Find index in Yesterday
            indices = [i for i, val in enumerate(vec_yesterday) if val == t]
            for idx in indices:
                if idx < len(vec_today):
                    res_now = vec_today[idx]
                    giai_name = GIAI_LABELS[idx] if idx < len(GIAI_LABELS) else str(idx)
                    results_txt.append(f"- Lô **{t}** hôm qua nổ ở giải **{giai_name}** -> Hôm nay giải đó về: **{res_now}**")
        
        if results_txt:
            for line in results_txt: st.markdown(line)
        else:
            st.write("Không tìm thấy vị trí tương ứng (hoặc lô nhập không xuất hiện hôm qua).")
            
    with st.expander("Xem bảng dữ liệu thô"):
        st.write("Hôm nay:", vec_today)
        st.write("Hôm qua:", vec_yesterday)

# ---- MAIN APP LAYOUT ----

tab1, tab2, tab3, tab4 = st.tabs(["KQ 45s/75s", "KQ 200 Kỳ", "Biểu Đồ", "Soi Cầu"])

with tab1:
    render_tab_kq_short()
with tab2:
    render_tab_kq_200()
with tab3:
    render_tab_charts()
with tab4:
    render_tab_cau_vitri()