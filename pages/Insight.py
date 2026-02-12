import streamlit as st

st.set_page_config(page_title="Insights", layout="wide")
st.title("Insights: What the Segments Tell Us")
st.caption("Personalization → Retention.")

st.subheader("1) Revenue is concentrated in high-value cohorts")
st.markdown("""
A small group of customers (e.g., **Champions** and **Loyal Accounts**) contributes the majority of total CLV.
This means retention initiatives should prioritize protecting these customers first.
""")

st.subheader("2) Churn risk is a major opportunity")
st.markdown("""
The **At Risk** and **Lost** segments represent a sizable portion of the base.
These cohorts are the clearest targets for win-back campaigns and re-engagement flows.
""")

st.subheader("3) Purchase frequency is a growth lever")
st.markdown("""
Many customers purchase infrequently. Increasing repeat purchase cadence can generate more value
than broad acquisition efforts—especially for mid-tier segments.
""")

st.subheader("4) Value is highly skewed → tiered marketing spend")
st.markdown("""
CLV is not evenly distributed. Marketing should be value-tiered:
high-touch personalization for high-CLV customers and automated nurture for lower-CLV cohorts.
""")

#st.info("Next step: go to **Strategy** to see segment-based actions derived from these insights.")