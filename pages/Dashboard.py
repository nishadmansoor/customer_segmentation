import streamlit as st
import importlib.util
from pathlib import Path

st.set_page_config(page_title="Dashboard", layout="wide")
st.title("Executive Dashboard")
st.caption("High-level segment distribution, CLV concentration, and engagement patterns (Power BI).")

st.subheader("How to read this dashboard")
st.markdown("""
1. Start with **Total CLV by Segment** to see where value concentrates.
2. Check **Customer Count by Segment** to understand scale.
3. Compare **Engagement / RFM patterns** to identify churn risk and growth opportunities.
""")

st.subheader("What to notice")
st.markdown("""
- **Champions** drive a disproportionate share of total CLV (retention priority).
- **At Risk** and **Lost** segments are sizable (win-back opportunity).
- CLV is **skewed**: a small set of customers contributes outsized value.
- Many customers show **low purchase frequency**, suggesting repeat-purchase interventions.
""")

st.divider()

# Run your existing Report page unchanged
report_path = Path(__file__).parent.parent / "src" / "Report.py"
spec = importlib.util.spec_from_file_location("report_module", report_path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)