import streamlit as st
import importlib.util
from pathlib import Path

st.set_page_config(page_title="Strategy", layout="wide")
st.title("Segment-Based Personalization Plan")
st.caption("Action framework that translates segmentation into retention and growth initiatives.")

st.subheader("Segment-based strategy map")
st.markdown("""
| Segment | Objective | Strategy |
|---|---|---|
| Champions | Protect & Upsell | VIP loyalty, premium support, early access, personalized recommendations |
| Loyal Accounts | Grow CLV | Cross-sell bundles, tailored offers, loyalty incentives |
| At Risk | Recover Revenue | Win-back emails, time-limited discounts, retargeting reminders |
| Lost | Low-cost Reacquisition | Lightweight reactivation offers, seasonal campaigns, remarketing |
| Promising / New Active | Nurture | Onboarding flows, education, first-to-second purchase incentives |
""")

st.subheader("How to operationalize")
st.markdown("""
- Prioritize **Champions** for retention and premium experience.
- Use **At Risk** cohorts for targeted win-back experiments.
- Build automated nurture pipelines to increase purchase frequency for mid-tier segments.
""")

st.divider()

# OPTIONAL: Run your existing Recommendations page unchanged
# (Keeps your previous code intact; if you later replace tokens with segment logic, it will show here automatically.)
#st.subheader("Interactive Recommendations")
#rec_path = Path(__file__).parent / "Recommendations.py"
#spec = importlib.util.spec_from_file_location("rec_module", rec_path)
#module = importlib.util.module_from_spec(spec)
#spec.loader.exec_module(module)