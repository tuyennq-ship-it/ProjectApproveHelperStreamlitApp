import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np
from scipy.stats import gaussian_kde

# ---------------------------
# CONFIG
# ---------------------------
st.set_page_config(layout="wide")

# ---------------------------
# CSS (dashboard style)
# ---------------------------
st.markdown("""
<style>
section[data-testid="stSidebar"] {
    width: 260px !important;
}

.block-container {
    padding-top: 1rem;
}

div[role="radiogroup"] label {
    padding: 4px 8px;
    margin: 2px 0;
    border-radius: 6px;
}

div[role="radiogroup"] label:hover {
    background-color: #f0f2f6;
}

div[data-testid="stHorizontalBlock"] > div {
    align-self: flex-start !important;
}
</style>
""", unsafe_allow_html=True)

# ---------------------------
# FILE PATHS
# ---------------------------
TOP3_PATH = "data/Top3_resubcluster_with_subclusters_analyse_09_17_22_04_std.csv"
TOP4_PATH = "data/Top4_resubcluster_with_subclusters_analyse_13_56_00_std.csv"
# ---------------------------
# LOAD DATA
# ---------------------------
@st.cache_data
def load_data(path):
    return pd.read_csv(path)

# ---------------------------
# SIDEBAR
# ---------------------------
st.sidebar.title("⚙️ Control Panel")

dataset = st.sidebar.radio("Dataset", ["Top 3", "Top 4"])

df = load_data(TOP3_PATH) if dataset == "Top 3" else load_data(TOP4_PATH)

# search
search = st.sidebar.text_input("🔍 Search sub_cluster")

sub_clusters = sorted(df["sub_cluster"].dropna().unique())

if search:
    sub_clusters = [sc for sc in sub_clusters if search.lower() in str(sc).lower()]

selected_subcluster = st.sidebar.radio(
    "Sub Cluster",
    sub_clusters,
    index=0 if sub_clusters else None
)

# ---------------------------
# MAIN
# ---------------------------
st.title("📊 Sub Cluster Dashboard")

if not selected_subcluster:
    st.warning("No sub_cluster found")
    st.stop()

sub_df = df[df["sub_cluster"] == selected_subcluster]

# header
st.markdown(f"### 🔹 Sub Cluster: `{selected_subcluster}`")

col1, col2 = st.columns([1, 1.3])

# ---------------------------
# TABLE (highlight đỏ)
# ---------------------------
def highlight_rows(row):
    if not row["is_in_hensachi_range"]:
        return ["background-color: #ffe6e6"] * len(row)
    return [""] * len(row)

# prepare display dataframe and format Price per unit to show no decimals
display_df = sub_df[["item_label", "Price per unit", "is_in_hensachi_range"]].copy()
# keep numeric values but format display without decimals
styled_df = display_df.style.apply(highlight_rows, axis=1).format({"Price per unit": "{:.0f}"})

with col1:
    st.markdown("#### 📋 Data")
    st.dataframe(
        styled_df,
        use_container_width=True,
        height=500
    )

# ---------------------------
# CHART
# ---------------------------
with col2:
    st.markdown("#### 📈 Distribution")

    prices = sub_df["Price per unit"].dropna().values

    if len(prices) > 1:
        fig = go.Figure()

        # Histogram
        fig.add_trace(go.Histogram(
            x=prices,
            histnorm='probability density',
            opacity=0.5,
            name='Histogram'
        ))

        # KDE
        if np.std(prices) > 0:
            kde = gaussian_kde(prices)
            x_range = np.linspace(prices.min(), prices.max(), 100)
            kde_values = kde(x_range)

            fig.add_trace(go.Scatter(
                x=x_range,
                y=kde_values,
                mode='lines',
                name='KDE',
                line=dict(color='black', width=2)
            ))

            max_y = kde_values.max()
        else:
            max_y = 1

        # Bounds
        expected = sub_df["expected"].iloc[0]
        lower_std = sub_df["lower_std"].iloc[0]
        upper_std = sub_df["upper_std"].iloc[0]

        # Vertical lines with VALUE in legend
        fig.add_trace(go.Scatter(
            x=[expected, expected],
            y=[0, max_y],
            mode='lines',
            name=f'expected ({expected:.2f})',
            line=dict(color='green')
        ))

        fig.add_trace(go.Scatter(
            x=[lower_std, lower_std],
            y=[0, max_y],
            mode='lines',
            name=f'lower_std ({lower_std:.2f})',
            line=dict(color='red', dash='dash')
        ))

        fig.add_trace(go.Scatter(
            x=[upper_std, upper_std],
            y=[0, max_y],
            mode='lines',
            name=f'upper_std ({upper_std:.2f})',
            line=dict(color='blue', dash='dash')
        ))

        # 🔥 Annotation (hiển thị trực tiếp trên chart)
        # fig.add_annotation(
        #     x=expected,
        #     y=max_y,
        #     text=f"expected: {expected:.2f}",
        #     showarrow=False,
        #     arrowhead=1,
        #     yshift=10,
        #     font=dict(color="green")
        # )

        # fig.add_annotation(
        #     x=lower_std,
        #     y=max_y,
        #     text=f"lower: {lower_std:.2f}",
        #     showarrow=False,
        #     arrowhead=1,
        #     yshift=10,
        #     font=dict(color="red")
        # )

        # fig.add_annotation(
        #     x=upper_std,
        #     y=max_y,
        #     text=f"upper: {upper_std:.2f}",
        #     showarrow=False,
        #     arrowhead=1,
        #     yshift=10,
        #     font=dict(color="blue")
        # )

        fig.update_layout(
            margin=dict(t=10),
            xaxis_title="Price per unit",
            yaxis_title="Density",
            showlegend=True
        )

        st.plotly_chart(fig, use_container_width=True)

    else:
        st.warning("Not enough data for KDE")