import streamlit as st
import pandas as pd
import contextlib
import importlib.util
from pathlib import Path

st.set_page_config(page_title="Analysis", layout="wide")

# -----------------------------
# Paths
# -----------------------------
REPO_ROOT = Path(__file__).resolve().parents[1]
CLV_PATH = REPO_ROOT / "clvData.csv"
NOTEBOOK_PATH = Path(__file__).parent / "src" / "Notebook.py"

# -----------------------------
# Page Header
# -----------------------------
st.title("Analysis: RFM + CLV Segmentation")
st.caption("Curated technical summary. Full notebook derivation available below.")

# -----------------------------
# Load Data
# -----------------------------
if not CLV_PATH.exists():
    st.error("clvData.csv not found in repo root.")
    st.stop()

clv = pd.read_csv(CLV_PATH)
clv.columns = [c.strip() for c in clv.columns]

COL_CUSTOMER = "CustomerID" if "CustomerID" in clv.columns else None
COL_SEGMENT = "Segment" if "Segment" in clv.columns else None

CLV_CANDIDATES = ["Customer Lifetime Value", "CustomerLifetimeValue", "CLV"]
COL_CLV = next((c for c in CLV_CANDIDATES if c in clv.columns), None)

AOV_CANDIDATES = ["Average Order Value", "AOV"]
COL_AOV = next((c for c in AOV_CANDIDATES if c in clv.columns), None)

FREQ_CANDIDATES = ["Purchase Frequency", "Frequency"]
COL_FREQ = next((c for c in FREQ_CANDIDATES if c in clv.columns), None)

for col in [COL_CLV, COL_AOV, COL_FREQ]:
    if col:
        clv[col] = pd.to_numeric(clv[col], errors="coerce")

# -----------------------------
# Headline Metrics
# -----------------------------
total_customers = clv[COL_CUSTOMER].nunique() if COL_CUSTOMER else len(clv)
total_clv = clv[COL_CLV].sum() if COL_CLV else None
avg_clv = clv[COL_CLV].mean() if COL_CLV else None

c1, c2, c3 = st.columns(3)
c1.metric("Customers", f"{total_customers:,}")
c2.metric("Total CLV", f"{total_clv:,.0f}" if total_clv else "—")
c3.metric("Avg CLV", f"{avg_clv:,.2f}" if avg_clv else "—")

st.divider()

# -----------------------------
# Segment Insights
# -----------------------------
if COL_SEGMENT and COL_CLV:

    seg_clv = clv.groupby(COL_SEGMENT)[COL_CLV].sum().sort_values(ascending=False)
    seg_counts = clv[COL_SEGMENT].value_counts()

    top_segment = seg_clv.index[0]
    top_share = seg_clv.iloc[0] / seg_clv.sum()

    st.subheader("Key Insights")

    st.markdown(f"""
- **Value concentration:** {top_segment} contributes ~{top_share:.0%} of total CLV.
- **Retention opportunity:** Churn-risk segments (At Risk, Lost, About to Sleep, Need Attention) represent meaningful upside.
- **Growth lever:** Increasing purchase frequency among mid-tier customers will significantly increase CLV.
""")

    st.divider()

    st.subheader("Segment Size")
    st.bar_chart(seg_counts)

    st.subheader("Total CLV by Segment")
    st.bar_chart(seg_clv)

# -----------------------------
# Behavior vs Value
# -----------------------------
if COL_AOV and COL_CLV:
    st.subheader("AOV vs CLV")
    scatter = clv[[COL_AOV, COL_CLV]].dropna()
    if len(scatter) > 0:
        st.scatter_chart(scatter, x=COL_AOV, y=COL_CLV)

if COL_FREQ:
    st.subheader("Purchase Frequency Distribution")
    freq = clv[COL_FREQ].fillna(0)
    st.bar_chart(freq.value_counts().sort_index())

st.divider()

# -----------------------------
# Optional: Full Notebook Output
# -----------------------------
with st.expander("Full Technical Notebook Output", expanded=False):

    if NOTEBOOK_PATH.exists():
        st.echo = lambda *args, **kwargs: contextlib.nullcontext()
        spec = importlib.util.spec_from_file_location("notebook_module", NOTEBOOK_PATH)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    else:
        st.warning("Notebook file not found.")