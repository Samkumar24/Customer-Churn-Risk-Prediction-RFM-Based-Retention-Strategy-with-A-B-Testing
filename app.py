import streamlit as st
import pandas as pd


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Churn Analytics",
    page_icon="📉",
    layout="wide"
)


# ============================================================
# SIDEBAR NAVIGATION
# ============================================================

st.sidebar.title("📉 Churn Analytics")

st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigation",
    [
        "📁 Upload & Overview",
        "👥 Customer Risk",
        "🔍 Customer Details",
        "🎯 Risk Segments",
        "⚠️ Risk Factors"
    ]
)

st.sidebar.markdown("---")

st.sidebar.caption("Customer Churn Prediction")
st.sidebar.caption("Analytics Dashboard")


# ============================================================
# SAMPLE DATA
# ============================================================
# This is temporary.
# Later this will come from your CSV / FastAPI prediction output.

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
# HELPER FUNCTION
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

    factors = []

    if row["days_since_last_purchase"] > 120:
        factors.append("Long purchase inactivity")

    if row["total_orders"] < 5:
        factors.append("Low order frequency")

    if row["returns_made"] >= 3:
        factors.append("High returns")

    if row["total_spend_usd"] < 300:
        factors.append("Low spending")

    if len(factors) == 0:
        factors.append("No major risk factors")

    return ", ".join(factors)


sample_data["risk"] = sample_data["churn_probability"].apply(get_risk)

sample_data["risk_factors"] = sample_data.apply(
    get_risk_factors,
    axis=1
)


# ============================================================
# PAGE 1
# UPLOAD & OVERVIEW
# ============================================================

if page == "📁 Upload & Overview":

    st.title("📁 Upload & Overview")

    st.markdown(
        "Upload your customer dataset to begin the churn analysis."
    )

    st.markdown("---")

    uploaded_file = st.file_uploader(
        "Upload Customer CSV",
        type=["csv"]
    )

    if uploaded_file is not None:

        df = pd.read_csv(uploaded_file)

        st.success(
            f"Successfully loaded {len(df):,} customers."
        )

    else:

        st.info(
            "No CSV uploaded yet. Showing sample data for the dashboard design."
        )

        df = sample_data

    st.markdown("---")

    # --------------------------------------------------------
    # KPI CARDS
    # --------------------------------------------------------

    total_customers = len(df)

    if "churn_probability" in df.columns:

        high_risk = len(
            df[df["churn_probability"] >= 0.70]
        )

        average_risk = (
            df["churn_probability"].mean() * 100
        )

    else:

        high_risk = 0
        average_risk = 0

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "👥 Total Customers",
            f"{total_customers:,}"
        )

    with col2:
        st.metric(
            "⚠️ High Risk Customers",
            f"{high_risk:,}"
        )

    with col3:
        st.metric(
            "📊 Average Churn Risk",
            f"{average_risk:.1f}%"
        )

    st.markdown("---")

    st.subheader("Customer Data Preview")

    st.dataframe(
        df.head(20),
        use_container_width=True,
        hide_index=True
    )


# ============================================================
# PAGE 2
# CUSTOMER RISK
# ============================================================

elif page == "👥 Customer Risk":

    st.title("👥 Customer Risk")

    st.markdown(
        "Search and filter customers based on their predicted churn risk."
    )

    st.markdown("---")

    # --------------------------------------------------------
    # SEARCH
    # --------------------------------------------------------

    search = st.text_input(
        "🔎 Search Customer ID",
        placeholder="Example: C001"
    )

    # --------------------------------------------------------
    # RISK FILTER
    # --------------------------------------------------------

    risk_filter = st.selectbox(
        "Risk Level",
        [
            "All",
            "🔴 Immediate Risk",
            "🟠 Priority Risk",
            "🟡 Targeted Risk",
            "🟢 Low Risk"
        ]
    )

    df = sample_data.copy()

    # Search
    if search:

        df = df[
            df["customer_id"]
            .astype(str)
            .str.contains(
                search,
                case=False,
                na=False
            )
        ]

    # Risk filter
    if risk_filter != "All":

        df = df[
            df["risk"] == risk_filter
        ]

    st.markdown("---")

    st.subheader(
        f"Customers Found: {len(df)}"
    )

    # --------------------------------------------------------
    # CUSTOMER TABLE
    # --------------------------------------------------------

    display_df = df[
        [
            "customer_id",
            "churn_probability",
            "risk",
            "risk_factors"
        ]
    ].copy()

    display_df["churn_probability"] = (
        display_df["churn_probability"] * 100
    ).round(1).astype(str) + "%"

    display_df.columns = [
        "Customer ID",
        "Churn Probability",
        "Risk",
        "Key Risk Factors"
    ]

    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True
    )


# ============================================================
# PAGE 3
# CUSTOMER DETAILS
# ============================================================

elif page == "🔍 Customer Details":

    st.title("🔍 Customer Details")

    st.markdown(
        "Select a customer to view their individual churn profile."
    )

    st.markdown("---")

    customer_ids = sample_data["customer_id"].tolist()

    selected_customer = st.selectbox(
        "Select Customer",
        customer_ids
    )

    customer = sample_data[
        sample_data["customer_id"] == selected_customer
    ].iloc[0]

    st.markdown("---")

    # --------------------------------------------------------
    # CHURN PROBABILITY
    # --------------------------------------------------------

    col1, col2 = st.columns([1, 2])

    with col1:

        st.metric(
            "Churn Probability",
            f"{customer['churn_probability'] * 100:.1f}%"
        )

        st.markdown(
            f"### {customer['risk']}"
        )

    with col2:

        st.subheader("⚠️ Key Risk Factors")

        factors = customer["risk_factors"].split(", ")

        for factor in factors:

            st.warning(factor)

    st.markdown("---")

    # --------------------------------------------------------
    # CUSTOMER PROFILE
    # --------------------------------------------------------

    st.subheader("👤 Customer Profile")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Customer ID",
            customer["customer_id"]
        )

    with col2:
        st.metric(
            "Age",
            customer["age"]
        )

    with col3:
        st.metric(
            "Membership",
            customer["membership_tier"]
        )

    with col4:
        st.metric(
            "Total Orders",
            customer["total_orders"]
        )

    st.markdown("---")

    # --------------------------------------------------------
    # CUSTOMER BEHAVIOUR
    # --------------------------------------------------------

    st.subheader("📊 Customer Behaviour")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Total Spend",
            f"${customer['total_spend_usd']:,.2f}"
        )

    with col2:
        st.metric(
            "Days Since Purchase",
            int(customer["days_since_last_purchase"])
        )

    with col3:
        st.metric(
            "Returns",
            int(customer["returns_made"])
        )


# ============================================================
# PAGE 4
# RISK SEGMENTS
# ============================================================

elif page == "🎯 Risk Segments":

    st.title("🎯 Risk Segments")

    st.markdown(
        "Customers are grouped into actionable risk segments."
    )

    st.markdown("---")

    df = sample_data.copy()

    # --------------------------------------------------------
    # SEGMENT COUNTS
    # --------------------------------------------------------

    immediate = len(
        df[df["churn_probability"] >= 0.90]
    )

    priority = len(
        df[
            (df["churn_probability"] >= 0.70)
            & (df["churn_probability"] < 0.90)
        ]
    )

    targeted = len(
        df[
            (df["churn_probability"] >= 0.40)
            & (df["churn_probability"] < 0.70)
        ]
    )

    low = len(
        df[df["churn_probability"] < 0.40]
    )

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        st.metric(
            "🔴 Immediate",
            immediate
        )

        st.caption("≥ 90%")

    with col2:

        st.metric(
            "🟠 Priority",
            priority
        )

        st.caption("70–89%")

    with col3:

        st.metric(
            "🟡 Targeted",
            targeted
        )

        st.caption("40–69%")

    with col4:

        st.metric(
            "🟢 Low Risk",
            low
        )

        st.caption("< 40%")

    st.markdown("---")

    # --------------------------------------------------------
    # SELECT SEGMENT
    # --------------------------------------------------------

    selected_segment = st.selectbox(
        "Select Segment",
        [
            "🔴 Immediate Risk",
            "🟠 Priority Risk",
            "🟡 Targeted Risk",
            "🟢 Low Risk"
        ]
    )

    segment_df = df[
        df["risk"] == selected_segment
    ]

    st.subheader(
        f"{selected_segment} Customers"
    )

    segment_display = segment_df[
        [
            "customer_id",
            "churn_probability",
            "risk_factors"
        ]
    ].copy()

    segment_display["churn_probability"] = (
        segment_display["churn_probability"] * 100
    ).round(1).astype(str) + "%"

    segment_display.columns = [
        "Customer ID",
        "Churn Probability",
        "Risk Factors"
    ]

    st.dataframe(
        segment_display,
        use_container_width=True,
        hide_index=True
    )


# ============================================================
# PAGE 5
# RISK FACTORS
# ============================================================

elif page == "⚠️ Risk Factors":

    st.title("⚠️ Risk Factors")

    st.markdown(
        "Understand the behavioural characteristics associated "
        "with higher churn risk."
    )

    st.markdown("---")

    df = sample_data.copy()

    # --------------------------------------------------------
    # RISK FACTOR COUNTS
    # --------------------------------------------------------

    inactivity = len(
        df[df["days_since_last_purchase"] > 120]
    )

    low_orders = len(
        df[df["total_orders"] < 5]
    )

    high_returns = len(
        df[df["returns_made"] >= 3]
    )

    low_spending = len(
        df[df["total_spend_usd"] < 300]
    )

    col1, col2 = st.columns(2)

    with col1:

        st.subheader("🕐 Purchase Inactivity")

        st.metric(
            "Customers with >120 days inactivity",
            inactivity
        )

    with col2:

        st.subheader("🛒 Low Order Frequency")

        st.metric(
            "Customers with <5 orders",
            low_orders
        )

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:

        st.subheader("↩️ High Returns")

        st.metric(
            "Customers with ≥3 returns",
            high_returns
        )

    with col2:

        st.subheader("💰 Low Spending")

        st.metric(
            "Customers spending < $300",
            low_spending
        )

    st.markdown("---")

    st.subheader("📋 Risk Factor Summary")

    factor_summary = pd.DataFrame({
        "Risk Factor": [
            "Long purchase inactivity",
            "Low order frequency",
            "High returns",
            "Low spending"
        ],
        "Customers Affected": [
            inactivity,
            low_orders,
            high_returns,
            low_spending
        ]
    })

    st.dataframe(
        factor_summary,
        use_container_width=True,
        hide_index=True
    )