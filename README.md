# Customer Segmentation Streamlit App

A Streamlit app that segments retail customers using **RFM + CLV** and translates segments into **personalized retention strategies**.  

---

## Project Tasks
* Engineers **RFM** features (Recency, Frequency, Monetary)
* Estimates/aggregates **Customer Lifetime Value (CLV)**
* Assigns customers into meaningful segments (e.g., Champions, Loyal, At Risk, Lost)

---

## Pages
* **Home** — Project overview + how to navigate the app
* **Analysis** — Curated technical summary (RFM/CLV logic + key charts)
* **Dashboard** — Executive dashboard view (PowerBI visuals)
* **Insights** — Takeaways that tie metrics → retention opportunities
* **Strategy** — Segment-based marketing actions (what to do next)

---

## Tech Stack
- Python, Pandas, NumPy
- Scikit-learn (clustering)
- Streamlit (web app)
- PowerBI (dashboard visuals / report-style charts)

---

## How to Run
```bash
git clone [https://github.com/nishadmansoor/customer_segmentation.git](https://github.com/nishadmansoor/customer_segmentation.git)
cd customer_segmentation

pip install -r requirements.txt
streamlit run Home.py
