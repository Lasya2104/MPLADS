
import os
from pathlib import Path
import sqlite3
import json
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
from PIL import Image

from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

try:
    from pymongo import MongoClient
except ImportError:
    MongoClient = None

APP_DIR = Path(__file__).parent
DATA_FILE = APP_DIR / "data" / "mplads_prototype_1000.csv"
DB_FILE = APP_DIR / "data" / "mplads.db"
DB_FILE.parent.mkdir(exist_ok=True)

st.set_page_config(
    page_title="MPLADS Work Intelligence",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ----------------------------
# Database
# ----------------------------
def get_sqlite():
    conn = sqlite3.connect(DB_FILE, check_same_thread=False)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS feedback (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            project_id TEXT NOT NULL,
            feedback_type TEXT NOT NULL,
            message TEXT,
            created_at TEXT NOT NULL
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS officer_audit (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            project_id TEXT NOT NULL,
            officer_name TEXT,
            verification TEXT NOT NULL,
            remarks TEXT,
            created_at TEXT NOT NULL
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS alerts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            project_id TEXT NOT NULL,
            alert_type TEXT NOT NULL,
            severity TEXT NOT NULL,
            message TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
    """)
    conn.commit()
    return conn

SQL = get_sqlite()

def save_feedback(project_id, feedback_type, message):
    SQL.execute(
        "INSERT INTO feedback(project_id, feedback_type, message, created_at) VALUES (?, ?, ?, ?)",
        (project_id, feedback_type, message, datetime.now().isoformat(timespec="seconds"))
    )
    SQL.commit()

def save_audit(project_id, officer_name, verification, remarks):
    SQL.execute(
        "INSERT INTO officer_audit(project_id, officer_name, verification, remarks, created_at) VALUES (?, ?, ?, ?, ?)",
        (project_id, officer_name, verification, remarks, datetime.now().isoformat(timespec="seconds"))
    )
    SQL.commit()

def save_alert(project_id, alert_type, severity, message):
    SQL.execute(
        "INSERT INTO alerts(project_id, alert_type, severity, message, created_at) VALUES (?, ?, ?, ?, ?)",
        (project_id, alert_type, severity, message, datetime.now().isoformat(timespec="seconds"))
    )
    SQL.commit()

# ----------------------------
# MongoDB
# ----------------------------
@st.cache_resource
def get_mongo():
    uri = os.getenv("MONGODB_URI", "").strip()
    if not uri or MongoClient is None:
        return None
    try:
        client = MongoClient(uri, serverSelectionTimeoutMS=3000)
        client.admin.command("ping")
        return client["mplads_intelligence"]
    except Exception:
        return None

MONGO = get_mongo()

def mongo_log(collection, document):
    if MONGO is not None:
        try:
            MONGO[collection].insert_one(document)
        except Exception:
            pass

# ----------------------------
# Data
# ----------------------------
@st.cache_data
def load_data():
    if not DATA_FILE.exists():
        return pd.DataFrame()
    df = pd.read_csv(DATA_FILE)
    date_cols = ["start_date", "expected_completion_date", "actual_completion_date"]
    for c in date_cols:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce")
    return df

df = load_data()

if df.empty:
    st.error("Dataset not found. Put mplads_prototype_1000.csv inside the data/ folder.")
    st.stop()

# ----------------------------
# Derived risk / anomaly logic
# ----------------------------
NUMERIC_RISK_COLS = [
    "fund_utilization_pct",
    "cost_overrun_pct",
    "delay_days",
    "progress_deviation_pct",
    "payment_count",
    "duplicate_similarity_pct",
    "location_consistency_pct",
    "public_negative_count",
    "abnormal_payment_flag",
    "expenditure_not_submitted",
]

@st.cache_data
def add_anomaly_flags(data):
    out = data.copy()
    X = out[NUMERIC_RISK_COLS].fillna(0).astype(float)
    model = IsolationForest(
        n_estimators=200,
        contamination=0.08,
        random_state=42
    )
    pred = model.fit_predict(X)
    out["anomaly_flag"] = np.where(pred == -1, 1, 0)
    out["anomaly_score"] = -model.score_samples(X)
    return out

df = add_anomaly_flags(df)

def risk_components(row):
    score = 0.0
    reasons = []

    fund = float(row.get("fund_utilization_pct", 0) or 0)
    overrun = float(row.get("cost_overrun_pct", 0) or 0)
    delay = float(row.get("delay_days", 0) or 0)
    deviation = abs(float(row.get("progress_deviation_pct", 0) or 0))
    dup = float(row.get("duplicate_similarity_pct", 0) or 0)
    loc = float(row.get("location_consistency_pct", 100) or 100)
    neg = float(row.get("public_negative_count", 0) or 0)
    abnormal = int(row.get("abnormal_payment_flag", 0) or 0)
    missing = int(row.get("expenditure_not_submitted", 0) or 0)
    anomaly = int(row.get("anomaly_flag", 0) or 0)

    if fund > 100:
        score += min(20, (fund - 100) * 0.8)
        reasons.append("Fund utilization exceeds sanctioned amount")
    elif fund < 30 and str(row.get("work_status", "")) == "Ongoing":
        score += 8
        reasons.append("Low fund utilization for an ongoing work")

    if overrun > 10:
        score += min(18, overrun * 0.8)
        reasons.append("Cost overrun is elevated")

    if delay > 30:
        score += min(18, delay * 0.18)
        reasons.append("Work is delayed")

    if deviation > 20:
        score += min(12, deviation * 0.25)
        reasons.append("Financial and physical progress are misaligned")

    if dup >= 75:
        score += 20
        reasons.append("High duplicate-work similarity")
    elif dup >= 60:
        score += 10
        reasons.append("Possible duplicate work")

    if loc < 75:
        score += 10
        reasons.append("Location consistency is low")

    if neg >= 3:
        score += min(8, neg * 1.2)
        reasons.append("Negative public feedback is elevated")

    if abnormal:
        score += 12
        reasons.append("Abnormal payment pattern detected")

    if missing:
        score += 8
        reasons.append("Expenditure submission is missing")

    if anomaly:
        score += 12
        reasons.append("Isolation Forest flagged the work as anomalous")

    score = round(min(100, score), 2)
    if score >= 75:
        level = "Critical"
    elif score >= 50:
        level = "High"
    elif score >= 25:
        level = "Medium"
    else:
        level = "Low"
    return score, level, reasons

risk_rows = df.apply(risk_components, axis=1, result_type="expand")
risk_rows.columns = ["computed_risk_score", "computed_risk_level", "risk_reasons"]
df = pd.concat([df, risk_rows], axis=1)

def money(x):
    try:
        return f"₹{float(x):,.0f}"
    except Exception:
        return "—"

def pct(x):
    try:
        return f"{float(x):.1f}%"
    except Exception:
        return "—"

def status_badge(status):
    return {
        "Completed": "🟢",
        "Ongoing": "🔵",
        "Delayed": "🔴",
        "Sanctioned": "🟡",
        "Recommended": "⚪",
    }.get(status, "⚪")

# ----------------------------
# Sidebar
# ----------------------------
st.sidebar.title("🏛️ MPLADS Intelligence")
st.sidebar.caption("Prototype: Work Monitoring & Risk Detection")

pages = [
    "🏠 Executive Dashboard",
    "🔎 Work Explorer",
    "🚨 Fund & Anomaly Detection",
    "♻️ Duplicate Works",
    "📍 Ground Reality Check",
    "🗣️ Public Feedback",
    "👮 Officer Verification",
    "📊 Decision Support",
    "🔁 Closed-Loop Audit",
]
page = st.sidebar.radio("Navigate", pages)

st.sidebar.divider()
st.sidebar.info(
    "This prototype uses the uploaded 1,000-row MPLADS dataset. "
    "Some monitoring/model fields are synthetic/prototype fields and should be replaced "
    "with verified official records before deployment."
)

# ----------------------------
# Common filters
# ----------------------------
def filters(data):
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        states = ["All"] + sorted(data["state"].dropna().unique().tolist())
        state = st.selectbox("State", states)
    temp = data if state == "All" else data[data.state == state]

    with c2:
        districts = ["All"] + sorted(temp["district"].dropna().unique().tolist())
        district = st.selectbox("District", districts)
    temp = temp if district == "All" else temp[temp.district == district]

    with c3:
        statuses = ["All"] + sorted(temp["work_status"].dropna().unique().tolist())
        status = st.selectbox("Status", statuses)
    temp = temp if status == "All" else temp[temp.work_status == status]

    with c4:
        sectors = ["All"] + sorted(temp["sector"].dropna().unique().tolist())
        sector = st.selectbox("Sector", sectors)
    temp = temp if sector == "All" else temp[temp.sector == sector]

    return temp

# ----------------------------
# Dashboard
# ----------------------------
if page == "🏠 Executive Dashboard":
    st.title("MPLADS Work Intelligence Dashboard")
    st.write("One place to monitor sanctioned works, expenditure, risk, anomalies and citizen feedback.")

    filtered = filters(df)

    total_works = len(filtered)
    sanctioned = filtered["sanctioned_amount"].sum()
    expenditure = filtered["expenditure_amount"].sum()
    high_risk = (filtered["computed_risk_level"].isin(["High", "Critical"])).sum()
    anomalies = filtered["anomaly_flag"].sum()

    a, b, c, d, e = st.columns(5)
    a.metric("Works", f"{total_works:,}")
    b.metric("Sanctioned", money(sanctioned))
    c.metric("Expenditure", money(expenditure))
    d.metric("High/Critical Risk", f"{high_risk:,}")
    e.metric("Anomalies", f"{anomalies:,}")

    st.divider()

    left, right = st.columns(2)
    with left:
        status_counts = filtered["work_status"].value_counts().reset_index()
        status_counts.columns = ["status", "count"]
        fig = px.pie(status_counts, names="status", values="count", hole=0.45,
                     title="Work Status")
        st.plotly_chart(fig, use_container_width=True)

    with right:
        risk_counts = filtered["computed_risk_level"].value_counts().reindex(
            ["Low", "Medium", "High", "Critical"], fill_value=0
        ).reset_index()
        risk_counts.columns = ["risk", "count"]
        fig = px.bar(risk_counts, x="risk", y="count", title="Computed Risk Level")
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Top works requiring attention")
    cols = [
        "project_id", "work_name", "state", "district", "work_status",
        "sanctioned_amount", "expenditure_amount", "progress_pct",
        "computed_risk_score", "computed_risk_level"
    ]
    top = filtered.sort_values("computed_risk_score", ascending=False).head(15)[cols].copy()
    top["sanctioned_amount"] = top["sanctioned_amount"].map(money)
    top["expenditure_amount"] = top["expenditure_amount"].map(money)
    top["progress_pct"] = top["progress_pct"].map(pct)
    st.dataframe(top, use_container_width=True, hide_index=True)

# ----------------------------
# Work Explorer
# ----------------------------
elif page == "🔎 Work Explorer":
    st.title("🔎 Particular Work Details")
    st.caption("Search one work and see its location, money flow, progress, dates, flags and feedback.")

    query = st.text_input("Search by Work ID / work name / district / constituency")
    result = df.copy()
    if query:
        q = query.lower()
        mask = (
            result["project_id"].astype(str).str.lower().str.contains(q, na=False)
            | result["work_name"].astype(str).str.lower().str.contains(q, na=False)
            | result["district"].astype(str).str.lower().str.contains(q, na=False)
            | result["constituency"].astype(str).str.lower().str.contains(q, na=False)
        )
        result = result[mask]

    if result.empty:
        st.warning("No work found.")
    else:
        choices = result["project_id"].tolist()
        selected_id = st.selectbox("Select Work", choices)
        row = result[result.project_id == selected_id].iloc[0]

        st.subheader(f"{status_badge(row.work_status)} {row.work_name}")
        st.caption(f"Work ID: {row.project_id}")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Recommended", money(row.recommended_amount))
        c2.metric("Sanctioned", money(row.sanctioned_amount))
        c3.metric("Expenditure", money(row.expenditure_amount))
        c4.metric("Physical Progress", pct(row.progress_pct))

        st.divider()
        l, r = st.columns(2)
        with l:
            st.markdown("### 📍 Location")
            st.write(f"**State:** {row.state}")
            st.write(f"**District:** {row.district}")
            st.write(f"**Constituency:** {row.constituency}")
            st.write(f"**Sector:** {row.sector}")
            st.write(f"**Sub-sector:** {row.sub_sector}")
            st.write(f"**MP ID:** {row.mp_id}")

        with r:
            st.markdown("### 📅 Timeline")
            st.write(f"**Status:** {row.work_status}")
            st.write(f"**Start:** {row.start_date.date() if pd.notna(row.start_date) else '—'}")
            st.write(f"**Expected completion:** {row.expected_completion_date.date() if pd.notna(row.expected_completion_date) else '—'}")
            st.write(f"**Actual completion:** {row.actual_completion_date.date() if pd.notna(row.actual_completion_date) else '—'}")
            st.write(f"**Delay:** {int(row.delay_days)} days")
            st.write(f"**Progress deviation:** {pct(row.progress_deviation_pct)}")

        st.markdown("### 💰 Fund utilization")
        utilization = min(max(float(row.fund_utilization_pct), 0), 130)
        st.progress(min(utilization / 100, 1.0))
        st.write(f"{pct(row.fund_utilization_pct)} utilized against sanctioned amount")

        st.markdown("### 🚨 Risk assessment")
        score, level, reasons = row.computed_risk_score, row.computed_risk_level, row.risk_reasons
        st.metric("Computed Risk Score", f"{score}/100", level)
        if reasons:
            for reason in reasons:
                st.warning(reason)
        else:
            st.success("No major rule-based risk signal detected.")

        st.markdown("### 🔍 Raw work record")
        display = row.to_frame("value")
        st.dataframe(display, use_container_width=True)

# ----------------------------
# Anomaly detection
# ----------------------------
elif page == "🚨 Fund & Anomaly Detection":
    st.title("🚨 Expenditure & Fund Anomaly Detection")
    st.write("Isolation Forest flags unusual combinations of utilization, delays, payment behavior and progress deviation.")

    threshold = st.slider("Show computed risk score above", 0, 100, 50)
    anomalies = df[df["computed_risk_score"] >= threshold].copy()
    anomalies = anomalies.sort_values("computed_risk_score", ascending=False)

    st.metric("Flagged works", len(anomalies))

    if len(anomalies):
        st.dataframe(
            anomalies[[
                "project_id", "work_name", "district", "work_status",
                "sanctioned_amount", "expenditure_amount",
                "fund_utilization_pct", "cost_overrun_pct", "delay_days",
                "progress_deviation_pct", "abnormal_payment_flag",
                "anomaly_flag", "computed_risk_score", "computed_risk_level"
            ]],
            use_container_width=True, hide_index=True
        )

        selected = st.selectbox("Open flagged work", anomalies.project_id.tolist())
        row = anomalies[anomalies.project_id == selected].iloc[0]
        st.subheader(row.work_name)
        for reason in row.risk_reasons:
            st.error(reason)

        if st.button("Create alert for this work"):
            msg = "; ".join(row.risk_reasons) or "High computed risk."
            save_alert(row.project_id, "FUND_RISK", row.computed_risk_level, msg)
            mongo_log("alerts", {
                "project_id": row.project_id,
                "alert_type": "FUND_RISK",
                "severity": row.computed_risk_level,
                "message": msg,
                "created_at": datetime.now()
            })
            st.success("Alert stored in SQLite and MongoDB if MongoDB is connected.")

# ----------------------------
# Duplicate works
# ----------------------------
elif page == "♻️ Duplicate Works":
    st.title("♻️ Duplicate Works Detection")
    st.write("Semantic similarity of work descriptions + location consistency can identify possible duplicate sanctions.")

    sample = df.copy()
    texts = sample["work_name"].fillna("").astype(str)

    @st.cache_resource
    def duplicate_model(texts_tuple):
        vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
        matrix = vec.fit_transform(list(texts_tuple))
        return vec, matrix

    vec, matrix = duplicate_model(tuple(texts.tolist()))

    selected = st.selectbox("Select a work", sample.project_id.tolist())
    idx = sample.index[sample.project_id == selected][0]
    sims = cosine_similarity(matrix[idx], matrix).flatten()
    candidates = []
    for j, sim in enumerate(sims):
        if j == list(sample.index).index(idx):
            continue
        row = sample.iloc[j]
        combined = 0.65 * float(sim * 100) + 0.35 * float(row.location_consistency_pct)
        candidates.append((combined, sim * 100, row))

    candidates.sort(key=lambda x: x[0], reverse=True)
    rows = []
    for combined, text_sim, row in candidates[:10]:
        rows.append({
            "project_id": row.project_id,
            "work_name": row.work_name,
            "district": row.district,
            "text_similarity_pct": round(text_sim, 2),
            "location_consistency_pct": round(float(row.location_consistency_pct), 2),
            "combined_duplicate_score": round(combined, 2),
            "status": row.work_status
        })
    dup_df = pd.DataFrame(rows)
    st.dataframe(dup_df, use_container_width=True, hide_index=True)

    if not dup_df.empty:
        top = dup_df.iloc[0]
        if top["combined_duplicate_score"] >= 75:
            st.error("⚠️ High-priority possible duplicate. Human verification required.")
        elif top["combined_duplicate_score"] >= 60:
            st.warning("Possible duplicate. Send for review.")
        else:
            st.success("No strong duplicate signal in the top matches.")

# ----------------------------
# Ground reality
# ----------------------------
elif page == "📍 Ground Reality Check":
    st.title("📍 Asset Creation & Ground Reality Check")
    st.write("Upload a field photograph for officer/citizen verification. Optional GPS fields can be stored with the evidence.")

    work_id = st.selectbox("Work", df.project_id.tolist())
    row = df[df.project_id == work_id].iloc[0]

    st.info(f"Selected work: **{row.work_name}** — {row.district}, {row.state}")

    uploaded = st.file_uploader("Upload field image", type=["jpg", "jpeg", "png", "webp"])
    lat = st.number_input("Latitude (optional)", value=0.0, format="%.6f")
    lon = st.number_input("Longitude (optional)", value=0.0, format="%.6f")
    note = st.text_area("Field observation")

    if uploaded:
        image = Image.open(uploaded)
        st.image(image, caption="Uploaded field evidence", use_container_width=True)

        # Lightweight image metadata/quality check; actual YOLO requires a separately
        # installed object-detection model and should be added after field requirements are fixed.
        arr = np.asarray(image.convert("RGB"))
        brightness = float(arr.mean())
        st.write(f"Image brightness score: **{brightness:.1f}/255**")

        if brightness < 35:
            st.warning("Image is very dark; retake the photograph.")
        else:
            st.success("Image is readable enough for manual verification.")

        if st.button("Save ground-reality evidence"):
            evidence = {
                "project_id": work_id,
                "latitude": lat if lat != 0 else None,
                "longitude": lon if lon != 0 else None,
                "note": note,
                "filename": uploaded.name,
                "created_at": datetime.now()
            }
            mongo_log("ground_evidence", evidence)
            st.success("Evidence saved to MongoDB when connected.")

# ----------------------------
# Public feedback
# ----------------------------
elif page == "🗣️ Public Feedback":
    st.title("🗣️ Grassroots Monitoring & Transparency")
    st.write("Simple citizen feedback workflow. For a production system, add IVRS/SMS/WhatsApp ingestion and multilingual speech-to-text.")

    work_id = st.selectbox("Select work", df.project_id.tolist(), key="feedback_work")
    row = df[df.project_id == work_id].iloc[0]
    st.info(f"**{row.work_name}** | {row.district} | Current status: {row.work_status}")

    feedback_type = st.radio("Feedback", ["Positive", "Negative", "Needs attention"])
    message = st.text_area("Citizen comment")
    if st.button("Submit feedback"):
        save_feedback(work_id, feedback_type, message)
        mongo_log("public_feedback", {
            "project_id": work_id,
            "feedback_type": feedback_type,
            "message": message,
            "created_at": datetime.now()
        })
        st.success("Feedback recorded.")

    feedback = pd.read_sql_query(
        "SELECT * FROM feedback WHERE project_id = ? ORDER BY id DESC",
        SQL, params=(work_id,)
    )
    if not feedback.empty:
        st.subheader("Recent feedback")
        st.dataframe(feedback, use_container_width=True, hide_index=True)

# ----------------------------
# Officer verification
# ----------------------------
elif page == "👮 Officer Verification":
    st.title("👮 Officer Verification")
    st.write("Human-in-the-loop review of AI/rule-based alerts.")

    work_id = st.selectbox("Work", df.project_id.tolist(), key="officer_work")
    row = df[df.project_id == work_id].iloc[0]

    st.subheader(row.work_name)
    c1, c2, c3 = st.columns(3)
    c1.metric("Risk", f"{row.computed_risk_score}/100")
    c2.metric("Duplicate similarity", pct(row.duplicate_similarity_pct))
    c3.metric("Location consistency", pct(row.location_consistency_pct))

    officer = st.text_input("Officer name")
    verification = st.selectbox("Verification result", ["Pending", "Verified", "Needs Recheck", "Rejected"])
    remarks = st.text_area("Officer remarks")

    if st.button("Save verification"):
        save_audit(work_id, officer, verification, remarks)
        mongo_log("officer_audits", {
            "project_id": work_id,
            "officer_name": officer,
            "verification": verification,
            "remarks": remarks,
            "created_at": datetime.now()
        })
        st.success("Officer verification saved.")

    audits = pd.read_sql_query(
        "SELECT * FROM officer_audit WHERE project_id = ? ORDER BY id DESC",
        SQL, params=(work_id,)
    )
    if not audits.empty:
        st.subheader("Audit history")
        st.dataframe(audits, use_container_width=True, hide_index=True)

# ----------------------------
# Decision support
# ----------------------------
elif page == "📊 Decision Support":
    st.title("📊 Actionable Decision Support")
    st.write("Composite Red / Amber / Green view for MPs, District Collectors and administrators.")

    work = df.copy()
    def traffic(level):
        if level == "Critical": return "🔴 RED"
        if level == "High": return "🔴 RED"
        if level == "Medium": return "🟠 AMBER"
        return "🟢 GREEN"

    work["priority"] = work["computed_risk_level"].map(traffic)
    work["recommended_action"] = np.select(
        [
            work.computed_risk_level.eq("Critical"),
            work.computed_risk_level.eq("High"),
            work.computed_risk_level.eq("Medium"),
        ],
        [
            "Immediate district-level investigation",
            "Field verification + financial review",
            "Monitor monthly",
        ],
        default="Routine monitoring"
    )

    action = st.multiselect(
        "Priority",
        ["🔴 RED", "🟠 AMBER", "🟢 GREEN"],
        default=["🔴 RED", "🟠 AMBER", "🟢 GREEN"]
    )
    view = work[work.priority.isin(action)].copy()
    st.dataframe(
        view[[
            "project_id", "work_name", "state", "district",
            "work_status", "computed_risk_score", "priority",
            "recommended_action"
        ]].sort_values("computed_risk_score", ascending=False),
        use_container_width=True, hide_index=True
    )

    st.download_button(
        "⬇️ Download decision-support CSV",
        view.to_csv(index=False).encode("utf-8"),
        "mplads_decision_support.csv",
        "text/csv"
    )

# ----------------------------
# Closed loop
# ----------------------------
elif page == "🔁 Closed-Loop Audit":
    st.title("🔁 Closed-Loop Governance")
    st.write("Track whether flagged works were actually reviewed and resolved.")

    alerts = pd.read_sql_query(
        "SELECT * FROM alerts ORDER BY id DESC", SQL
    )
    audits = pd.read_sql_query(
        "SELECT * FROM officer_audit ORDER BY id DESC", SQL
    )

    a, b, c = st.columns(3)
    a.metric("Generated alerts", len(alerts))
    b.metric("Officer audits", len(audits))
    if len(audits):
        verified = (audits.verification == "Verified").sum()
        c.metric("Verified audits", int(verified))
    else:
        c.metric("Verified audits", 0)

    if not alerts.empty:
        st.subheader("Alerts")
        st.dataframe(alerts, use_container_width=True, hide_index=True)

    if not audits.empty:
        st.subheader("Officer audit trail")
        st.dataframe(audits, use_container_width=True, hide_index=True)

    st.info(
        "Production workflow: Alert → District/field officer assignment → "
        "photo/GPS evidence → officer decision → resolution → monthly re-check."
    )
