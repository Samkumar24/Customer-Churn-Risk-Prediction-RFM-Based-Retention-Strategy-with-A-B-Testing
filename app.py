import streamlit as st
import pandas as pd
import requests


# ============================================================
# CONFIG
# ============================================================

API_URL = "http://localhost:8000"

# Must match fast_api.py's Customer_data schema exactly — these are the
# columns pulled from an uploaded CSV and sent to /predict.
API_FEATURE_COLUMNS = [
    "country", "age", "gender", "membership_tier", "total_orders",
    "total_spend_usd", "avg_order_value_usd", "days_since_last_purchase",
    "preferred_category", "preferred_device", "preferred_payment_method",
    "acquisition_channel", "reviews_given", "avg_review_score",
    "returns_made", "wishlist_items", "newsletter_subscribed"
]

# What marketing should actually do for each risk tier — this is the
# "Business Action" column from the decision framework, made explicit
# so it shows up on every table instead of living only in someone's head.
ACTION_MAP = {
    "🔴 Immediate Risk": "Escalate to account manager — personal retention call this week",
    "🟠 Priority Risk": "Priority outreach — targeted retention offer",
    "🟡 Targeted Risk": "Add to nurture campaign — win-back email/promo",
    "🟢 Low Risk": "Monitor — no action needed",
}

DISPLAY_COLUMNS = ["customer_id", "churn_probability", "risk", "risk_factors", "action"]
DISPLAY_HEADERS = ["Customer ID", "Churn Probability", "Risk", "Key Risk Factors", "Recommended Action"]


# ============================================================
# PAGE CONFIG + STYLE
# ============================================================

st.set_page_config(
    page_title="Churn Analytics",
    page_icon="📉",
    layout="wide"
)

with open("style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


def hero(icon, title, subtitle):
    st.markdown(
        f"""
        <div class="hero">
            <div class="hero-icon">{icon}</div>
            <div>
                <div class="hero-title">{title}</div>
                <div class="hero-subtitle">{subtitle}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


UPLOAD_ILLUSTRATION = """
<svg width="120" height="120" viewBox="0 0 120 120">
  <circle cx="60" cy="60" r="56" fill="#B9FAF8" opacity="0.5"/>
  <path d="M35 68 Q30 68 30 58 Q30 48 40 47 Q42 34 56 34 Q68 34 71 45 Q82 45 82 56
           Q82 68 70 68 Z" fill="#FFFFFF" stroke="#A663CC" stroke-width="2"/>
  <line x1="60" y1="52" x2="60" y2="78" stroke="#6F2DBD" stroke-width="3" stroke-linecap="round"/>
  <path d="M50 62 L60 52 L70 62" fill="none" stroke="#6F2DBD" stroke-width="3"
        stroke-linecap="round" stroke-linejoin="round"/>
  <rect x="42" y="82" width="6" height="12" rx="2" fill="#B298DC"/>
  <rect x="52" y="78" width="6" height="16" rx="2" fill="#A663CC"/>
  <rect x="62" y="74" width="6" height="20" rx="2" fill="#6F2DBD"/>
  <rect x="72" y="80" width="6" height="14" rx="2" fill="#B298DC"/>
</svg>
"""


# ============================================================
# SIDEBAR NAVIGATION
# ============================================================

st.sidebar.markdown(
    "<div class='sidebar-brand'><span class='sidebar-brand-icon'>📉</span>"
    "<span class='sidebar-brand-text'>Churn Analytics</span></div>",
    unsafe_allow_html=True,
)

page = st.sidebar.radio(
    "Navigation",
    [
        "📁 Upload & Overview",
        "👥 Customer Risk",
        "🔍 Customer Details",
        "🎯 Risk Segments",
    ],
    label_visibility="collapsed",
)

st.sidebar.markdown("<div class='sidebar-footer'>Customer Churn Prediction<br>Analytics Dashboard</div>", unsafe_allow_html=True)


# ============================================================
# SAMPLE DATA
# ============================================================
# Shown until a real CSV is uploaded on the Overview page.

sample_data = pd.DataFrame({
    "customer_id": [
        "C001", "C002", "C003", "C004", "C005",
        "C006", "C007", "C008", "C009", "C010"
    ],
    "churn_probability": [
        0.942, 0.827, 0.513, 0.084, 0.731,
        0.921, 0.351, 0.673, 0.124, 0.889
    ],
    "age": [
        42, 31, 27, 35, 48,
        52, 29, 41, 24, 56
    ],
    "membership_tier": [
        "Basic", "Gold", "Silver", "Gold", "Basic",
        "Basic", "Gold", "Silver", "Gold", "Basic"
    ],
    "total_orders": [
        2, 15, 7, 23, 3,
        1, 11, 6, 21, 2
    ],
    "total_spend_usd": [
        120, 1450, 620, 2300, 180,
        90, 980, 540, 2100, 150
    ],
    "days_since_last_purchase": [
        186, 94, 71, 12, 143,
        210, 45, 63, 18, 172
    ],
    "returns_made": [
        3, 1, 2, 0, 4,
        5, 1, 2, 0, 3
    ]
})


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def get_risk(probability):

    if probability >= 0.90:
        return "🔴 Immediate Risk"

    elif probability >= 0.70:
        return "🟠 Priority Risk"

    elif probability >= 0.40:
        return "🟡 Targeted Risk"

    else:
        return "🟢 Low Risk"


def get_risk_factors(row):
    # Uses row.get(...) with a default so this still works on uploaded
    # files that don't have every one of these columns — those checks
    # just never trigger instead of crashing the app.

    factors = []

    if row.get("days_since_last_purchase", 0) > 120:
        factors.append("Long purchase inactivity")

    if row.get("total_orders", 999) < 5:
        factors.append("Low order frequency")

    if row.get("returns_made", 0) >= 3:
        factors.append("High returns")

    if row.get("total_spend_usd", 999999) < 300:
        factors.append("Low spending")

    if len(factors) == 0:
        factors.append("No major risk factors")

    return ", ".join(factors)


def find_customer_id_column(df):
    # Uploaded files won't always name the column exactly "customer_id" —
    # check the common variants, then fall back to the first column.
    candidates = ["customer_id", "Customer ID", "CustomerID", "customerid", "id", "ID"]
    for c in candidates:
        if c in df.columns:
            return c
    return df.columns[0]


def score_via_api(df, api_url=API_URL):
    """Sends every row to FastAPI /predict in one call and returns a list
    of churn probabilities in row order. Raises requests.RequestException
    on network/API failure — caller shows the message."""
    records = df[API_FEATURE_COLUMNS].to_dict(orient="records")
    response = requests.put(f"{api_url}/predict", json=records, timeout=60)
    response.raise_for_status()
    return response.json()["prediction"]


def enrich(df):
    df = df.copy()

    id_col = find_customer_id_column(df)
    if id_col != "customer_id":
        df = df.rename(columns={id_col: "customer_id"})

    if "churn_probability" in df.columns:
        df["risk"] = df["churn_probability"].apply(get_risk)
    else:
        df["risk"] = "🟢 Low Risk"

    df["risk_factors"] = df.apply(get_risk_factors, axis=1)
    df["action"] = df["risk"].map(ACTION_MAP)

    return df


def to_display_table(df):
    """Trims any dataset — sample or uploaded — down to the five columns
    marketing actually needs, with friendly headers."""
    out = df[DISPLAY_COLUMNS].copy()
    out["churn_probability"] = (out["churn_probability"] * 100).round(1).astype(str) + "%"
    out.columns = DISPLAY_HEADERS
    return out


# ============================================================
# SESSION STATE — persists the working dataset across pages
# ============================================================

if "data" not in st.session_state:
    st.session_state.data = enrich(sample_data)

if "using_sample" not in st.session_state:
    st.session_state.using_sample = True


# ============================================================
# PAGE 1
# UPLOAD & OVERVIEW
# ============================================================

if page == "📁 Upload & Overview":

    col_hero, col_art = st.columns([4, 1])
    with col_hero:
        hero("📁", "Upload & Overview", "Upload your customer dataset to begin the churn analysis.")
    with col_art:
        st.markdown(f"<div style='text-align:right;'>{UPLOAD_ILLUSTRATION}</div>", unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Upload Customer CSV",
        type=["csv"]
    )

    if uploaded_file is not None:

        raw_df = pd.read_csv(uploaded_file)
        id_col_used = find_customer_id_column(raw_df)

        if "churn_probability" not in raw_df.columns:
            missing = [c for c in API_FEATURE_COLUMNS if c not in raw_df.columns]
            if missing:
                st.error(
                    "This CSV has no 'churn_probability' column, and is missing "
                    f"the columns the prediction API needs to compute one: {missing}"
                )
                st.stop()
            try:
                with st.spinner(f"Scoring {len(raw_df):,} customers via FastAPI..."):
                    raw_df["churn_probability"] = score_via_api(raw_df)
            except requests.RequestException as e:
                st.error(f"Could not reach the prediction API at {API_URL}: {e}")
                st.stop()

        st.session_state.data = enrich(raw_df)
        st.session_state.using_sample = False

        st.success(
            f"Successfully loaded {len(raw_df):,} customers "
            f"(using '{id_col_used}' as customer_id)."
        )

    elif st.session_state.using_sample:

        st.info(
            "No CSV uploaded yet. Showing sample data for the dashboard design."
        )

    df = st.session_state.data

    # --------------------------------------------------------
    # KPI CARDS
    # --------------------------------------------------------

    total_customers = len(df)
    high_risk = len(df[df["churn_probability"] >= 0.70])
    average_risk = df["churn_probability"].mean() * 100

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("👥 Total Customers", f"{total_customers:,}")

    with col2:
        st.metric("⚠️ High Risk Customers", f"{high_risk:,}")

    with col3:
        st.metric("📊 Average Churn Risk", f"{average_risk:.1f}%")

    st.subheader("Customer Overview")

    st.dataframe(
        to_display_table(df).head(20),
        use_container_width=True,
        hide_index=True
    )


# ============================================================
# PAGE 2
# CUSTOMER RISK
# ============================================================

elif page == "👥 Customer Risk":

    hero("👥", "Customer Risk", "Search and filter customers based on their predicted churn risk.")

    # --------------------------------------------------------
    # SEARCH + FILTER
    # --------------------------------------------------------

    search_col, filter_col = st.columns([2, 1])
    with search_col:
        search = st.text_input("🔎 Search Customer ID", placeholder="Example: C001")
    with filter_col:
        risk_filter = st.selectbox(
            "Risk Level",
            ["All", "🔴 Immediate Risk", "🟠 Priority Risk", "🟡 Targeted Risk", "🟢 Low Risk"]
        )

    df = st.session_state.data.copy()

    if search:
        df = df[df["customer_id"].astype(str).str.contains(search, case=False, na=False)]

    if risk_filter != "All":
        df = df[df["risk"] == risk_filter]

    st.subheader(f"Customers Found: {len(df)}")

    st.dataframe(
        to_display_table(df),
        use_container_width=True,
        hide_index=True
    )


# ============================================================
# PAGE 3
# CUSTOMER DETAILS
# ============================================================

elif page == "🔍 Customer Details":

    hero("🔍", "Customer Details", "Select a customer to view their individual churn profile.")

    data = st.session_state.data

    customer_ids = data["customer_id"].tolist()

    selected_customer = st.selectbox("Select Customer", customer_ids)

    customer = data[data["customer_id"] == selected_customer].iloc[0]

    # --------------------------------------------------------
    # CHURN PROBABILITY + ACTION
    # --------------------------------------------------------

    col1, col2 = st.columns([1, 2])

    with col1:
        st.metric("Churn Probability", f"{customer['churn_probability'] * 100:.1f}%")
        st.markdown(f"### {customer['risk']}")

    with col2:
        st.subheader("🎯 Recommended Action")
        st.info(customer["action"])

        st.subheader("⚠️ Key Risk Factors")
        for factor in customer["risk_factors"].split(", "):
            st.warning(factor)

    # --------------------------------------------------------
    # CUSTOMER PROFILE
    # --------------------------------------------------------

    st.subheader("👤 Customer Profile")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Customer ID", customer["customer_id"])

    with col2:
        st.metric("Age", customer.get("age", "—"))

    with col3:
        st.metric("Membership", customer.get("membership_tier", "—"))

    with col4:
        st.metric("Total Orders", customer.get("total_orders", "—"))

    # --------------------------------------------------------
    # CUSTOMER BEHAVIOUR
    # --------------------------------------------------------

    st.subheader("📊 Customer Behaviour")

    col1, col2, col3 = st.columns(3)

    with col1:
        spend = customer.get("total_spend_usd")
        st.metric("Total Spend", f"${spend:,.2f}" if pd.notna(spend) else "—")

    with col2:
        days = customer.get("days_since_last_purchase")
        st.metric("Days Since Purchase", int(days) if pd.notna(days) else "—")

    with col3:
        returns = customer.get("returns_made")
        st.metric("Returns", int(returns) if pd.notna(returns) else "—")


# ============================================================
# PAGE 4
# RISK SEGMENTS
# ============================================================

elif page == "🎯 Risk Segments":

    hero("🎯", "Risk Segments", "Customers are grouped into actionable risk segments.")

    df = st.session_state.data.copy()

    # --------------------------------------------------------
    # SEGMENT COUNTS
    # --------------------------------------------------------

    immediate = len(df[df["churn_probability"] >= 0.90])
    priority = len(df[(df["churn_probability"] >= 0.70) & (df["churn_probability"] < 0.90)])
    targeted = len(df[(df["churn_probability"] >= 0.40) & (df["churn_probability"] < 0.70)])
    low = len(df[df["churn_probability"] < 0.40])

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("🔴 Immediate", immediate)
        st.caption("≥ 90%")

    with col2:
        st.metric("🟠 Priority", priority)
        st.caption("70–89%")

    with col3:
        st.metric("🟡 Targeted", targeted)
        st.caption("40–69%")

    with col4:
        st.metric("🟢 Low Risk", low)
        st.caption("< 40%")

    # --------------------------------------------------------
    # SELECT SEGMENT
    # --------------------------------------------------------

    selected_segment = st.selectbox(
        "Select Segment",
        ["🔴 Immediate Risk", "🟠 Priority Risk", "🟡 Targeted Risk", "🟢 Low Risk"]
    )

    segment_df = df[df["risk"] == selected_segment]

    st.subheader(f"{selected_segment} Customers")

    st.dataframe(
        to_display_table(segment_df),
        use_container_width=True,
        hide_index=True
    )