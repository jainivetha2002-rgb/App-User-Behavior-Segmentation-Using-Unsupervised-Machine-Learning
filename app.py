import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

st.set_page_config(
    page_title="App User Behavior Segmentation",
    layout="wide"
)

# --------------------------------------------------
# Load Data
# --------------------------------------------------
@st.cache_data
def load_data():
    return pd.read_csv("outputs/final_clustered_users.csv")

df = load_data()

# --------------------------------------------------
# Sidebar
# --------------------------------------------------
st.sidebar.title("🔍 Filters")

selected_clusters = st.sidebar.multiselect(
    "Select Cluster(s)",
    sorted(df["cluster"].unique()),
    default=sorted(df["cluster"].unique())
)

filtered_df = df[df["cluster"].isin(selected_clusters)]

# --------------------------------------------------
# Title
# --------------------------------------------------
st.title("📊 App User Behavior Segmentation Dashboard")
st.markdown(
    "Interactive visualization of **user engagement clusters** created using "
    "**K-Means Unsupervised Learning**."
)

# --------------------------------------------------
# KPIs
# --------------------------------------------------
c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Users", len(filtered_df))
c2.metric("Avg Engagement", round(filtered_df["engagement_score"].mean(), 2))
c3.metric("Avg Sessions / Week", round(filtered_df["sessions_per_week"].mean(), 2))
c4.metric("Avg Churn Risk", round(filtered_df["churn_risk_score"].mean(), 2))

st.divider()

# --------------------------------------------------
# Cluster Distribution
# --------------------------------------------------
st.subheader("👥 Cluster Distribution")

fig, ax = plt.subplots()
sns.countplot(data=filtered_df, x="cluster", ax=ax)
ax.set_title("Users per Cluster")
st.pyplot(fig)

# --------------------------------------------------
# Heatmap (VERY POWERFUL)
# --------------------------------------------------
st.subheader("🔥 Cluster Behavior Heatmap")

heatmap_data = (
    filtered_df
    .groupby("cluster")[[
        "sessions_per_week",
        "avg_session_duration_min",
        "daily_active_minutes",
        "feature_clicks_per_session",
        "engagement_score",
        "churn_risk_score"
    ]]
    .mean()
)

fig, ax = plt.subplots(figsize=(10, 5))
sns.heatmap(heatmap_data, annot=True, fmt=".1f", cmap="coolwarm", ax=ax)
st.pyplot(fig)

# --------------------------------------------------
# Boxplots
# --------------------------------------------------
st.subheader("📦 Feature Distribution Across Clusters")

feature_choice = st.selectbox(
    "Select feature",
    [
        "sessions_per_week",
        "avg_session_duration_min",
        "daily_active_minutes",
        "feature_clicks_per_session",
        "engagement_score",
        "churn_risk_score"
    ]
)

fig, ax = plt.subplots()
sns.boxplot(data=filtered_df, x="cluster", y=feature_choice, ax=ax)
st.pyplot(fig)

# --------------------------------------------------
# Scatter: Engagement vs Churn
# --------------------------------------------------
st.subheader("📈 Engagement vs Churn Risk")

fig, ax = plt.subplots()
sns.scatterplot(
    data=filtered_df,
    x="engagement_score",
    y="churn_risk_score",
    hue="cluster",
    alpha=0.6,
    ax=ax
)
st.pyplot(fig)

# --------------------------------------------------
# Top Users Table
# --------------------------------------------------
st.subheader("🏆 Top Users by Engagement")

top_users = (
    filtered_df
    .sort_values("engagement_score", ascending=False)
    .head(10)
)

st.dataframe(top_users)

# --------------------------------------------------
# Download Button
# --------------------------------------------------
st.subheader("⬇️ Download Clustered Data")

csv = filtered_df.to_csv(index=False).encode("utf-8")
st.download_button(
    "Download CSV",
    csv,
    "filtered_clustered_users.csv",
    "text/csv"
)

# --------------------------------------------------
# Raw Data
# --------------------------------------------------
with st.expander("📄 View Raw Data"):
    st.dataframe(filtered_df.head(100))
