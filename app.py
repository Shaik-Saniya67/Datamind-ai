import streamlit as st
import pandas as pd
import numpy as np
from groq import Groq
import plotly.express as px
import plotly.graph_objects as go
from utils.data_processor import load_data, get_summary_stats, run_sql_query
from utils.visualizer import auto_visualize
import json
import os

st.set_page_config(page_title="DataMind AI", page_icon="🧠", layout="wide")

CHAT_FILE = "chat_session.json"

def save_chat():
    with open(CHAT_FILE, "w") as f:
        json.dump(st.session_state.chat, f)

def load_chat():
    if os.path.exists(CHAT_FILE):
        with open(CHAT_FILE, "r") as f:
            return json.load(f)
    return []

if "df" not in st.session_state:
    st.session_state.df = None
if "df2" not in st.session_state:
    st.session_state.df2 = None
if "chat" not in st.session_state:
    st.session_state.chat = load_chat()
if "font_size" not in st.session_state:
    st.session_state.font_size = 16
if "theme" not in st.session_state:
    st.session_state.theme = "dark"
if "auto_insights" not in st.session_state:
    st.session_state.auto_insights = None
if "data_dict" not in st.session_state:
    st.session_state.data_dict = None

fs = st.session_state.font_size
is_dark = st.session_state.theme == "dark"
bg = "#0a0a0f" if is_dark else "#f5f5ff"
card_bg = "#14141f" if is_dark else "#ffffff"
sidebar_bg = "#111118" if is_dark else "#eeeeff"
text_col = "#e8e8f0" if is_dark else "#111111"
border_col = "#2a2a3a" if is_dark else "#c0c8ff"
sub_text = "#666680" if is_dark else "#333333"
chat_q_bg = "#1a1a2e" if is_dark else "#e0e8ff"
chat_a_bg = "#0f1f1f" if is_dark else "#f0fff8"
chat_q_text = "#c8c8e8" if is_dark else "#111111"
chat_a_text = "#c8e8e8" if is_dark else "#111111"

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600&display=swap');
html, body, [class*="css"] {{ font-family: 'DM Sans', sans-serif; font-size: {fs}px; color: {text_col}; }}
.stApp {{ background: {bg}; color: {text_col}; }}
[data-testid="stSidebar"] {{ background: {sidebar_bg}; border-right: 1px solid {border_col}; }}
.stButton > button {{ background: transparent !important; border: 1px solid #00d4ff !important; color: #00d4ff !important; font-family: 'Space Mono', monospace; font-size: {fs-3}px; border-radius: 8px; transition: all 0.2s; }}
.stButton > button:hover {{ background: #00d4ff20 !important; }}
.stDownloadButton > button {{ background: transparent !important; border: 1px solid #00d4ff !important; color: #00d4ff !important; font-family: 'Space Mono', monospace; font-size: {fs-3}px; border-radius: 8px; }}
.stDownloadButton > button:hover {{ background: #00d4ff20 !important; }}
.stTextInput > div > div > input {{ background: {card_bg} !important; border: 1px solid {border_col} !important; color: {text_col} !important; border-radius: 8px; }}
.stTextArea > div > div > textarea {{ background: {card_bg} !important; border: 1px solid {border_col} !important; color: {text_col} !important; }}
.stTabs [data-baseweb="tab"] {{ color: #444460; font-family: 'Space Mono', monospace; font-size: {fs-3}px; }}
.stTabs [aria-selected="true"] {{ color: #00d4ff !important; border-bottom: 2px solid #00d4ff; }}
div[data-testid="stMetric"] {{ background: {card_bg}; border: 1px solid {border_col}; border-radius: 12px; padding: 1rem; }}
div[data-testid="stMetric"] label {{ color: {sub_text} !important; font-family: 'Space Mono', monospace; }}
div[data-testid="stMetric"] div {{ color: #00d4ff !important; font-family: 'Space Mono', monospace; }}
.chat-q {{ background: {chat_q_bg}; border: 1px solid {border_col}; border-radius: 12px 12px 4px 12px; padding: 0.8rem 1.2rem; margin: 0.5rem 0; margin-left: 15%; color: {chat_q_text} !important; font-size: {fs}px; }}
.chat-a {{ background: {chat_a_bg}; border-left: 3px solid #00d4ff; border-radius: 4px 12px 12px 12px; padding: 0.8rem 1.2rem; margin: 0.5rem 0; margin-right: 15%; color: {chat_a_text} !important; line-height: 1.7; font-size: {fs}px; }}
.kpi-card {{ background: {card_bg}; border: 1px solid {border_col}; border-radius: 12px; padding: 1rem; text-align: center; }}
.kpi-val {{ font-size: {fs+8}px; font-weight: 700; color: #00d4ff; font-family: 'Space Mono', monospace; }}
.kpi-lbl {{ font-size: {fs-3}px; color: {sub_text}; text-transform: uppercase; letter-spacing: 1px; font-family: 'Space Mono', monospace; }}
.insight-card {{ background: {card_bg}; border: 1px solid #00d4ff30; border-left: 3px solid #00d4ff; border-radius: 8px; padding: 0.8rem 1.2rem; margin: 0.4rem 0; color: {text_col}; }}
.anomaly-card {{ background: {card_bg}; border: 1px solid #ff6b6b30; border-left: 3px solid #ff6b6b; border-radius: 8px; padding: 0.8rem 1.2rem; margin: 0.4rem 0; color: {text_col}; }}
@media (max-width: 768px) {{
    .chat-q {{ margin-left: 5% !important; }}
    .chat-a {{ margin-right: 5% !important; }}
}}
p, label {{ color: {text_col} !important; }}
</style>
""", unsafe_allow_html=True)

# ── Groq Client ────────────────────────────────────────────────────────────────

client = Groq(api_key=st.secrets["GROQ_API_KEY"])


# ── AI Functions ───────────────────────────────────────────────────────────────
def ask_ai(df, question):
    try:
        num_cols = df.select_dtypes(include=np.number).columns.tolist()
        cat_cols = df.select_dtypes(include=["object","category"]).columns.tolist()
        cat_info = ""
        for c in cat_cols[:3]:
            cat_info += f"\n{c} value counts:\n{df[c].value_counts().head(10).to_string()}"
        corr_info = ""
        if len(num_cols) >= 2:
            corr_info = f"\nCorrelations:\n{df[num_cols].corr().to_string()}"
        data_info = f"""Shape: {df.shape}
Columns: {list(df.columns)}
Types: {df.dtypes.to_dict()}
Missing: {df.isnull().sum().to_dict()}
Stats:\n{df.describe().to_string()}
Sample (10 rows):\n{df.head(10).to_string()}
Categories:{cat_info}{corr_info}"""
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": "You are DataMind AI, expert data analyst. Give specific numbers. Use bullet points."},
                {"role": "user", "content": f"Dataset:\n{data_info}\n\nQuestion: {question}"}
            ],
            max_tokens=1200
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"❌ Error: {str(e)}"

def generate_auto_insights(df):
    try:
        num_cols = df.select_dtypes(include=np.number).columns.tolist()
        cat_cols = df.select_dtypes(include=["object","category"]).columns.tolist()
        cat_info = ""
        for c in cat_cols[:3]:
            cat_info += f"\n{c} value counts:\n{df[c].value_counts().head(5).to_string()}"
        data_info = f"""Shape: {df.shape}
Stats:\n{df.describe().to_string()}
Categories:{cat_info}"""
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": "You are DataMind AI. Generate exactly 6 key business insights. Each insight must be ONE sentence with specific numbers. Format: bullet points only. Example: '• Electronics contributes 42% of total revenue.'"},
                {"role": "user", "content": f"Dataset:\n{data_info}\n\nGenerate 6 key insights with specific numbers:"}
            ],
            max_tokens=600
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"❌ Error: {str(e)}"

def generate_data_dictionary(df):
    try:
        col_info = ""
        for col in df.columns:
            dtype = str(df[col].dtype)
            sample = str(df[col].dropna().head(3).tolist())
            col_info += f"\n- {col} (type: {dtype}, sample: {sample})"
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": "You are a data analyst. For each column, give a short 1-line description of what it likely represents. Return as: ColumnName: Description"},
                {"role": "user", "content": f"Columns:{col_info}\n\nDescribe each column in one line:"}
            ],
            max_tokens=600
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"❌ Error: {str(e)}"

def detect_anomalies(df):
    anomalies = []
    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    for col in num_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        outliers = df[(df[col] < lower) | (df[col] > upper)]
        if len(outliers) > 0:
            anomalies.append({
                "Column": col,
                "Outlier Count": len(outliers),
                "Normal Range": f"{lower:.2f} — {upper:.2f}",
                "Min Outlier": f"{outliers[col].min():.2f}",
                "Max Outlier": f"{outliers[col].max():.2f}"
            })
        # Negative values check
        neg = df[df[col] < 0]
        if len(neg) > 0:
            anomalies.append({
                "Column": col,
                "Outlier Count": len(neg),
                "Normal Range": "Should be positive",
                "Min Outlier": f"{neg[col].min():.2f}",
                "Max Outlier": "Negative values found"
            })
    return anomalies

def nl_to_sql(df, question):
    try:
        cols = list(df.columns)
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": f"Convert natural language to SQL. Table name is 'data'. Columns: {cols}. Return ONLY the SQL query, nothing else."},
                {"role": "user", "content": question}
            ],
            max_tokens=200
        )
        sql = response.choices[0].message.content.strip()
        sql = sql.replace("```sql", "").replace("```", "").strip()
        return sql
    except Exception as e:
        return f"-- Error: {str(e)}"

def auto_clean(df):
    cleaned = df.copy()
    report = []
    for col in cleaned.columns:
        nulls = cleaned[col].isnull().sum()
        if nulls > 0:
            if cleaned[col].dtype in [np.float64, np.int64]:
                med = cleaned[col].median()
                cleaned[col].fillna(med, inplace=True)
                report.append(f"✅ {col}: filled {nulls} missing with median ({med:.2f})")
            else:
                mode = cleaned[col].mode()[0]
                cleaned[col].fillna(mode, inplace=True)
                report.append(f"✅ {col}: filled {nulls} missing with mode ({mode})")
    if not report:
        report.append("✅ No missing values — data is already clean!")
    return cleaned, report

# ── Download helpers ───────────────────────────────────────────────────────────
def get_csv_download(df):
    return df.to_csv(index=False).encode('utf-8')

def get_full_report(df):
    report = "="*60 + "\nDATAMIND AI — FULL REPORT\n" + "="*60 + "\n\n"
    report += f"Shape: {df.shape}\nColumns: {list(df.columns)}\n\n"
    report += "STATISTICS\n" + df.describe().to_string() + "\n\n"
    if st.session_state.auto_insights:
        report += "="*60 + "\nAUTO INSIGHTS\n" + "="*60 + "\n"
        report += st.session_state.auto_insights + "\n\n"
    if st.session_state.chat:
        report += "="*60 + "\nAI CHAT HISTORY\n" + "="*60 + "\n\n"
        for i, (q, a) in enumerate(st.session_state.chat, 1):
            report += f"Q{i}: {q}\nA{i}: {a}\n\n" + "-"*40 + "\n\n"
    return report.encode('utf-8')

def get_all_charts_html(charts):
    html = "<html><head><title>DataMind AI Charts</title></head><body style='background:#0a0a0f;color:#e8e8f0;'>"
    for i, fig in enumerate(charts):
        html += f"<h2 style='color:#00d4ff;'>Chart {i+1}</h2>"
        html += fig.to_html(full_html=False, include_plotlyjs='cdn')
        html += "<hr style='border-color:#2a2a3a;'>"
    html += "</body></html>"
    return html.encode('utf-8')

def get_dashboard_html(figs):
    html = "<html><head><title>DataMind AI Dashboard</title></head><body style='background:#0a0a0f;'>"
    for fig in figs:
        html += fig.to_html(full_html=False, include_plotlyjs='cdn') + "<hr>"
    html += "</body></html>"
    return html.encode('utf-8')

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(f'<h2 style="color:#00d4ff;font-family:Space Mono,monospace;">🧠 DataMind</h2>', unsafe_allow_html=True)
    st.markdown(f'<p style="color:{sub_text};font-size:{fs-2}px;">AI Data Insights Engine v2.0</p>', unsafe_allow_html=True)
    st.markdown("---")

    st.markdown(f'<p style="color:#00d4ff;font-family:Space Mono,monospace;font-size:{fs-3}px;text-transform:uppercase;letter-spacing:1px;">🔤 Font Size</p>', unsafe_allow_html=True)
    col_f1, col_f2, col_f3 = st.columns(3)
    with col_f1:
        if st.button("A-"):
            if st.session_state.font_size > 12:
                st.session_state.font_size -= 2
                st.rerun()
    with col_f2:
        st.markdown(f'<p style="text-align:center;color:#00d4ff;font-family:Space Mono,monospace;">{fs}px</p>', unsafe_allow_html=True)
    with col_f3:
        if st.button("A+"):
            if st.session_state.font_size < 24:
                st.session_state.font_size += 2
                st.rerun()

    st.markdown(f'<p style="color:#00d4ff;font-family:Space Mono,monospace;font-size:{fs-3}px;text-transform:uppercase;letter-spacing:1px;margin-top:0.5rem;">🎨 Theme</p>', unsafe_allow_html=True)
    col_t1, col_t2 = st.columns(2)
    with col_t1:
        if st.button("🌙 Dark"):
            st.session_state.theme = "dark"
            st.rerun()
    with col_t2:
        if st.button("☀️ Light"):
            st.session_state.theme = "light"
            st.rerun()

    st.markdown("---")
    st.markdown(f'<p style="color:#00d4ff;font-family:Space Mono,monospace;font-size:{fs-3}px;text-transform:uppercase;letter-spacing:1px;">📂 Upload Data</p>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Primary Dataset", type=["csv", "xlsx"])
    if uploaded_file:
        df_loaded, msg = load_data(uploaded_file)
        if df_loaded is not None:
            st.session_state.df = df_loaded
            st.session_state.chat = []
            st.session_state.auto_insights = None
            st.session_state.data_dict = None
            save_chat()
            st.success(f"✅ {len(df_loaded)} rows × {len(df_loaded.columns)} cols")
            # Auto generate insights on upload
            with st.spinner("🧠 Generating insights..."):
                st.session_state.auto_insights = generate_auto_insights(df_loaded)

    uploaded_file2 = st.file_uploader("Compare Dataset (optional)", type=["csv", "xlsx"])
    if uploaded_file2:
        df2_loaded, msg2 = load_data(uploaded_file2)
        if df2_loaded is not None:
            st.session_state.df2 = df2_loaded
            st.success(f"✅ File 2: {len(df2_loaded)} rows")

    if st.session_state.df is not None:
        st.markdown("---")
        st.markdown(f'<p style="color:#00d4ff;font-family:Space Mono,monospace;font-size:{fs-3}px;text-transform:uppercase;letter-spacing:1px;">📥 Downloads</p>', unsafe_allow_html=True)
        st.download_button("📤 Export Data CSV", get_csv_download(st.session_state.df), "data_export.csv", "text/csv", use_container_width=True)
        if st.session_state.chat:
            st.download_button("📋 Full Report", get_full_report(st.session_state.df), "full_report.txt", "text/plain", use_container_width=True)

    st.markdown("---")
    st.markdown(f'<p style="color:#333350;font-family:Space Mono,monospace;font-size:{fs-4}px;text-align:center;">DataMind AI v2.0<br>FlowZint Hackathon 2026</p>', unsafe_allow_html=True)

# ── Main ───────────────────────────────────────────────────────────────────────
st.markdown(f'<h1 style="background:linear-gradient(135deg,#00d4ff,#7b2fff,#ff6b6b);-webkit-background-clip:text;-webkit-text-fill-color:transparent;font-family:Space Mono,monospace;font-size:{fs+14}px;">DataMind AI 🧠</h1>', unsafe_allow_html=True)
st.markdown(f'<p style="color:{sub_text};">Upload any dataset → Get instant AI-powered insights → Analyze → Decide</p>', unsafe_allow_html=True)

if st.session_state.df is None:
    c1, c2, c3, c4 = st.columns(4)
    for col, icon, title, desc in zip([c1,c2,c3,c4],["📂","💬","📊","🔢"],["Upload Data","AI Chat","Auto Charts","SQL Query"],["CSV or Excel","Ask anything","Auto visualize","Run queries"]):
        with col:
            st.markdown(f'<div class="kpi-card"><div style="font-size:{fs+10}px;">{icon}</div><div style="color:{text_col};font-weight:600;">{title}</div><div style="color:{sub_text};font-size:{fs-2}px;">{desc}</div></div>', unsafe_allow_html=True)
    st.info("👈 Upload a CSV file in the sidebar to get started!")
else:
    df = st.session_state.df
    stats = get_summary_stats(df)
    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    cat_cols = df.select_dtypes(include=["object","category"]).columns.tolist()
    date_cols = [c for c in df.columns if any(x in c.lower() for x in ["date", "month", "year", "time", "day", "week", "quarter", "period"])]
    # ── KPI Metrics ───────────────────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("📋 Total Rows", f"{stats['rows']:,}")
    c2.metric("📊 Columns", stats['cols'])
    c3.metric("🔢 Numeric Cols", stats['numeric_cols'])
    c4.metric("⚠️ Missing", stats['missing'])

    # ── KPI Trend Cards ───────────────────────────────────────────────────────
    if num_cols:
        st.markdown("---")
        kpi_cols = st.columns(min(4, len(num_cols)))
        for i, col_name in enumerate(num_cols[:4]):
            with kpi_cols[i]:
                val = df[col_name].sum()
                avg = df[col_name].mean()
                std = df[col_name].std()
                label = col_name.replace("_", " ").title()
                display_val = f"{val/1000:.1f}K" if val > 1000 else f"{val:.1f}"
                # Trend indicator
                half = len(df) // 2
                if half > 0:
                    first_half = df[col_name].iloc[:half].mean()
                    second_half = df[col_name].iloc[half:].mean()
                    if first_half > 0:
                        trend_pct = ((second_half - first_half) / first_half) * 100
                        trend = f"↑ {trend_pct:.1f}%" if trend_pct > 0 else f"↓ {abs(trend_pct):.1f}%"
                        trend_color = "#00ff9d" if trend_pct > 0 else "#ff6b6b"
                    else:
                        trend = "—"
                        trend_color = sub_text
                else:
                    trend = "—"
                    trend_color = sub_text
                st.markdown(f'''<div class="kpi-card">
                    <div class="kpi-lbl">{label}</div>
                    <div class="kpi-val">{display_val}</div>
                    <div style="color:#888;font-size:{fs-3}px;">avg: {avg:.1f}</div>
                    <div style="color:{trend_color};font-size:{fs-2}px;font-weight:600;">{trend}</div>
                </div>''', unsafe_allow_html=True)

    # ── Auto Insights Banner ──────────────────────────────────────────────────
    if st.session_state.auto_insights:
        st.markdown("---")
        st.markdown(f'<p style="color:#00d4ff;font-family:Space Mono,monospace;font-size:{fs-2}px;text-transform:uppercase;letter-spacing:2px;">⚡ Auto Insights</p>', unsafe_allow_html=True)
        insights = st.session_state.auto_insights.split("\n")
        cols_ins = st.columns(2)
        for i, insight in enumerate([x for x in insights if x.strip()]):
            with cols_ins[i % 2]:
                st.markdown(f'<div class="insight-card">{insight}</div>', unsafe_allow_html=True)

    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9 = st.tabs([
        "💬 AI Chat", "📊 Auto Charts", "🎯 Dashboard",
        "🗃 Data Explorer", "🔢 SQL Query", "🤖 NL to SQL",
        "🔄 Compare", "🧹 Auto Clean", "📖 Data Dictionary"
    ])

    # ── Tab 1: AI Chat ─────────────────────────────────────────────────────────
    with tab1:
        st.subheader("Ask anything about your data")
        st.caption("💡 Ask multiple questions — all answers saved below")
        # Dataset selector
        # Dataset selector
        active_df = df  # default
        if st.session_state.df2 is not None:
            dataset_choice = st.radio(
                "Ask about:",
                ["📁 Primary Dataset", "📁 Compare Dataset"],
                horizontal=True,
                key="dataset_radio"
            )
            if "Compare" in dataset_choice:
                active_df = st.session_state.df2
            st.caption(f"Asking about: {dataset_choice} ({active_df.shape[0]} rows × {active_df.shape[1]} cols)")
        else:
            active_df = df

        col1, col2, col3, col4 = st.columns(4)
        q_to_ask = None
        if col1.button("📊 Summarize"):
            q_to_ask = "Summarize this dataset in detail"
        if col2.button("🏆 Top Products"):
            q_to_ask = "What are the top products by sales?"
        if col3.button("❓ Missing Values"):
            q_to_ask = "Are there any missing values?"
        if col4.button("💡 Key Insights"):
            q_to_ask = "What are the key business insights?"

        col5, col6, col7, col8 = st.columns(4)
        if col5.button("📈 Sales Trend"):
            q_to_ask = "What is the sales trend?"
        if col6.button("🌍 Regional"):
            q_to_ask = "Give a regional breakdown."
        if col7.button("👥 Customers"):
            q_to_ask = "Analyze customer demographics."
        if col8.button("⚠️ Anomalies"):
            q_to_ask = "Any anomalies or outliers?"

        user_q = st.text_input("", placeholder="Type your own question...", key="typed_q", label_visibility="collapsed")
        col_ask, col_clear = st.columns([1, 5])
        with col_ask:
            if st.button("Ask →"):
                if user_q.strip():
                    q_to_ask = user_q.strip()
        with col_clear:
            if st.session_state.chat:
                if st.button("🗑 Clear All"):
                    st.session_state.chat = []
                    save_chat()
                    st.rerun()

        if q_to_ask:
            with st.spinner("🧠 Analyzing..."):
                ans = ask_ai(active_df, q_to_ask)
            st.session_state.chat.append((q_to_ask, ans))
            save_chat()

        if st.session_state.chat:
            st.markdown(f"**{len(st.session_state.chat)} question(s) asked:**")
        for i, (q, a) in enumerate(st.session_state.chat):
            st.markdown(f'<div class="chat-q">🧑 <b>Q{i+1}:</b> {q}</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="chat-a">🧠 <b>DataMind AI:</b><br>{a}</div>', unsafe_allow_html=True)
            if st.button(f"🗑 Delete Q{i+1}", key=f"del_{i}"):
                st.session_state.chat.pop(i)
                save_chat()
                st.rerun()

        if st.session_state.chat:
            st.markdown("---")
            st.download_button("📥 Download Full Report", get_full_report(df), "full_report.txt", "text/plain", use_container_width=True)

    # ── Tab 2: Auto Charts ─────────────────────────────────────────────────────
    with tab2:
        # Date filter
        if date_cols:
            st.markdown(f'<p style="color:#00d4ff;font-family:Space Mono,monospace;font-size:{fs-3}px;">📅 Filter by Date Column</p>', unsafe_allow_html=True)
            sel_date = st.selectbox("Filter by date/month/year:", date_cols)
            unique_vals = sorted(df[sel_date].dropna().unique().tolist())
            selected_vals = st.multiselect("Select values:", unique_vals, default=unique_vals)
            df_filtered = df[df[sel_date].isin(selected_vals)] if selected_vals else df
            st.caption(f"Showing {len(df_filtered)} of {len(df)} rows")
        else:
            df_filtered = df

        charts = auto_visualize(df_filtered)
        for fig in charts:
            st.plotly_chart(fig, use_container_width=True)
        st.markdown("---")
        st.download_button("📥 Download All Charts", get_all_charts_html(charts), "all_charts.html", "text/html", use_container_width=True)

    # ── Tab 3: Dashboard ───────────────────────────────────────────────────────
    with tab3:
        dashboard_figs = []
        if num_cols and cat_cols:
            # Date filter for dashboard
            if date_cols:
                sel_date_d = st.selectbox("Filter by:", date_cols, key="dash_date")
                unique_d = sorted(df[sel_date_d].dropna().unique().tolist())
                sel_d = st.multiselect("Period:", unique_d, default=unique_d, key="dash_period")
                df_dash = df[df[sel_date_d].isin(sel_d)] if sel_d else df
            else:
                df_dash = df

            col_a, col_b = st.columns(2)
            with col_a:
                x_col = st.selectbox("Category:", cat_cols)
            with col_b:
                y_col = st.selectbox("Value:", num_cols)
            top_n = st.slider("Top N:", 5, 20, 10)
            top_data = df_dash.groupby(x_col)[y_col].sum().nlargest(top_n).reset_index()

            fig1 = px.bar(top_data, x=x_col, y=y_col, title=f"Top {top_n} {x_col} by {y_col}", template="plotly_dark", color=y_col, color_continuous_scale="blues")
            fig1.update_layout(paper_bgcolor="#0a0a0f", plot_bgcolor="#0a0a0f")
            st.plotly_chart(fig1, use_container_width=True)
            dashboard_figs.append(fig1)

            col_c, col_d = st.columns(2)
            with col_c:
                fig2 = px.pie(top_data, names=x_col, values=y_col, title=f"{x_col} Share", template="plotly_dark", hole=0.4)
                fig2.update_layout(paper_bgcolor="#0a0a0f")
                st.plotly_chart(fig2, use_container_width=True)
                dashboard_figs.append(fig2)
            with col_d:
                if len(num_cols) >= 2:
                    fig3 = px.scatter(df_dash, x=num_cols[0], y=num_cols[1], color=cat_cols[0], title=f"{num_cols[0]} vs {num_cols[1]}", template="plotly_dark")
                    fig3.update_layout(paper_bgcolor="#0a0a0f", plot_bgcolor="#0a0a0f")
                    st.plotly_chart(fig3, use_container_width=True)
                    dashboard_figs.append(fig3)

            st.markdown("---")
            st.download_button("📥 Download Dashboard", get_dashboard_html(dashboard_figs), "dashboard.html", "text/html", use_container_width=True)
        else:
            st.info("Need both numeric and category columns!")

    # ── Tab 4: Data Explorer ───────────────────────────────────────────────────
    with tab4:
        col_a, col_b = st.columns(2)
        with col_a:
            n = st.slider("Rows:", 5, 50, 10)
            st.dataframe(df.head(n), use_container_width=True)
        with col_b:
            st.dataframe(df.describe(), use_container_width=True)

        # Missing value heatmap
        st.markdown("**🔥 Missing Value Heatmap**")
        missing_data = df.isnull().astype(int)
        if missing_data.sum().sum() > 0:
            fig_miss = px.imshow(missing_data.T, title="Missing Values (Red = Missing)", color_continuous_scale=["#0a0a0f", "#ff6b6b"], template="plotly_dark")
            fig_miss.update_layout(paper_bgcolor="#0a0a0f")
            st.plotly_chart(fig_miss, use_container_width=True)
        else:
            st.success("✅ No missing values — heatmap not needed!")

        st.markdown("**🧹 Data Quality Report**")
        quality = [{"Column": c, "Type": str(df[c].dtype), "Missing": df[c].isnull().sum(), "Unique": df[c].nunique(), "Status": "⚠️ Has nulls" if df[c].isnull().sum() > 0 else "✅ Good"} for c in df.columns]
        quality_df = pd.DataFrame(quality)
        st.dataframe(quality_df, use_container_width=True)

        st.markdown("---")
        col_d1, col_d2 = st.columns(2)
        with col_d1:
            st.download_button("📥 Download Raw Data", get_csv_download(df), "raw_data.csv", "text/csv", use_container_width=True)
        with col_d2:
            st.download_button("📥 Download Quality Report", quality_df.to_csv(index=False).encode(), "quality_report.csv", "text/csv", use_container_width=True)

    # ── Tab 5: SQL Query ───────────────────────────────────────────────────────
    with tab5:
        st.code(f"Table: data | Columns: {', '.join(df.columns.tolist())}")
        sql = st.text_area("SQL Query:", "SELECT * FROM data LIMIT 10", height=80)
        if st.button("▶ Run Query"):
            result, err = run_sql_query(df, sql)
            if err:
                st.error(err)
            else:
                st.success(f"✅ {len(result)} rows returned")
                st.dataframe(result, use_container_width=True)
                st.markdown("---")
                st.download_button("📥 Download Result", result.to_csv(index=False).encode(), "query_result.csv", "text/csv", use_container_width=True)

    # ── Tab 6: NL to SQL ───────────────────────────────────────────────────────
    with tab6:
        st.subheader("🤖 Natural Language to SQL")
        st.caption("Type a question in plain English — AI converts it to SQL and runs it!")
        st.code(f"Table: data | Columns: {', '.join(df.columns.tolist())}")

        nl_examples = [
            "Show top 5 rows by total sales",
            "Count rows by category",
            "Show average price by region",
        ]
        st.markdown("**Quick examples:**")
        col_e1, col_e2, col_e3 = st.columns(3)
        nl_q = None
        if col_e1.button(nl_examples[0]):
            nl_q = nl_examples[0]
        if col_e2.button(nl_examples[1]):
            nl_q = nl_examples[1]
        if col_e3.button(nl_examples[2]):
            nl_q = nl_examples[2]

        nl_input = st.text_input("Ask in plain English:", placeholder="e.g. Show average sales by region", key="nl_input")
        if st.button("🤖 Convert & Run"):
            if nl_input.strip():
                nl_q = nl_input.strip()

        if nl_q:
            with st.spinner("🤖 Converting to SQL..."):
                generated_sql = nl_to_sql(df, nl_q)
            st.markdown("**Generated SQL:**")
            st.code(generated_sql, language="sql")
            if not generated_sql.startswith("--"):
                result, err = run_sql_query(df, generated_sql)
                if err:
                    st.error(f"SQL Error: {err}")
                else:
                    st.success(f"✅ {len(result)} rows returned")
                    st.dataframe(result, use_container_width=True)
                    st.download_button("📥 Download Result", result.to_csv(index=False).encode(), "nl_result.csv", "text/csv")

    # ── Tab 7: Compare Files ───────────────────────────────────────────────────
    with tab7:
        st.subheader("🔄 Compare Two Datasets")
        if st.session_state.df2 is None:
            st.info("👈 Upload a second CSV in the sidebar to compare!")
        else:
            df2 = st.session_state.df2
            col_a, col_b = st.columns(2)
            with col_a:
                st.markdown("**📁 Dataset 1**")
                st.metric("Rows", df.shape[0])
                st.metric("Columns", df.shape[1])
                st.metric("Missing", df.isnull().sum().sum())
                st.dataframe(df.describe(), use_container_width=True)
            with col_b:
                st.markdown("**📁 Dataset 2**")
                st.metric("Rows", df2.shape[0])
                st.metric("Columns", df2.shape[1])
                st.metric("Missing", df2.isnull().sum().sum())
                st.dataframe(df2.describe(), use_container_width=True)

            common_cols = list(set(df.columns) & set(df2.columns))
            if common_cols:
                sel_col = st.selectbox("Compare column:", common_cols)
                if sel_col in df.select_dtypes(include=np.number).columns:
                    comp_data = pd.DataFrame({
                        "Dataset": ["Dataset 1", "Dataset 2"],
                        "Mean": [df[sel_col].mean(), df2[sel_col].mean()],
                        "Max": [df[sel_col].max(), df2[sel_col].max()],
                        "Min": [df[sel_col].min(), df2[sel_col].min()],
                    })
                    st.dataframe(comp_data, use_container_width=True)
                    fig_comp = px.bar(comp_data, x="Dataset", y="Mean", title=f"Average {sel_col} Comparison", template="plotly_dark", color="Dataset")
                    fig_comp.update_layout(paper_bgcolor="#0a0a0f", plot_bgcolor="#0a0a0f")
                    st.plotly_chart(fig_comp, use_container_width=True)
            st.markdown("---")
            comp_report = f"COMPARISON REPORT\n{'='*40}\nDataset 1: {df.shape}\nDataset 2: {df2.shape}\n\nDataset 1:\n{df.describe().to_string()}\n\nDataset 2:\n{df2.describe().to_string()}"
            st.download_button("📥 Download Comparison Report", comp_report.encode(), "comparison.txt", "text/plain", use_container_width=True)

    # ── Tab 8: Auto Clean ──────────────────────────────────────────────────────
    with tab8:
        st.subheader("🧹 Auto Data Cleaning")
        missing_count = df.isnull().sum().sum()
        if missing_count == 0:
            st.success("✅ Dataset is already clean!")
        else:
            st.warning(f"⚠️ Found {missing_count} missing values!")

        col_prev, col_info = st.columns(2)
        with col_prev:
            st.markdown("**Before Cleaning:**")
            before = [{"Column": c, "Missing": df[c].isnull().sum(), "Type": str(df[c].dtype)} for c in df.columns if df[c].isnull().sum() > 0]
            if before:
                st.dataframe(pd.DataFrame(before), use_container_width=True)
            else:
                st.info("No missing values!")
        with col_info:
            st.markdown("**Strategy:**")
            st.info("• Numeric → filled with **median**\n• Text → filled with **mode**")

        if st.button("🧹 Clean Data Now!"):
            cleaned_df, report = auto_clean(df)
            st.session_state.df = cleaned_df
            st.success("✅ Cleaned!")
            for r in report:
                st.markdown(r)
            st.dataframe(cleaned_df.head(10), use_container_width=True)
            st.download_button("📥 Download Cleaned Data", get_csv_download(cleaned_df), "cleaned_data.csv", "text/csv", use_container_width=True)

    # ── Tab 9: Data Dictionary ─────────────────────────────────────────────────
    with tab9:
        st.subheader("📖 AI-Powered Data Dictionary")
        st.caption("AI explains what each column means")

        if st.session_state.data_dict is None:
            if st.button("📖 Generate Data Dictionary"):
                with st.spinner("🤖 Analyzing columns..."):
                    st.session_state.data_dict = generate_data_dictionary(df)

        if st.session_state.data_dict:
            lines = st.session_state.data_dict.strip().split("\n")
            dict_data = []
            for line in lines:
                if ":" in line:
                    parts = line.split(":", 1)
                    dict_data.append({"Column": parts[0].strip(), "Description": parts[1].strip()})

            if dict_data:
                dict_df = pd.DataFrame(dict_data)
                st.dataframe(dict_df, use_container_width=True)
                st.markdown("---")
                st.download_button("📥 Download Data Dictionary", dict_df.to_csv(index=False).encode(), "data_dictionary.csv", "text/csv", use_container_width=True)
            else:
                st.write(st.session_state.data_dict)