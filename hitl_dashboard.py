"""Authenticated Pharma Forecasting HITL review dashboard."""
import json
import os
from io import StringIO

import pandas as pd
import streamlit as st
from google.cloud import storage
from tools.cloud_hitl import load_latest_reviews, submit_review_once
from tools.hitl_review import observation_hash
from tools.iap_auth import get_verified_reviewer

BUCKET = os.getenv("PHARMA_GCS_BUCKET", "pharma-auto-forecasting-data")
st.set_page_config(page_title="Pharma Forecasting | HITL", page_icon="💊", layout="wide")
st.title("Pharma Forecasting — HITL Review")
st.caption("Authenticated anomaly assessment · decisions are recorded in Google Cloud Storage")
reviewer = get_verified_reviewer()
if reviewer:
    st.sidebar.success(f"Verified reviewer: {reviewer['email']}")
else:
    st.sidebar.warning("Identity not verified — read-only mode")

@st.cache_data(ttl=120)
def load_review_queue():
    blob = storage.Client().bucket(BUCKET).blob("audit/audit_log.jsonl")
    if not blob.exists():
        raise FileNotFoundError("Forecast audit log unavailable")
    latest = {}
    for line in blob.download_as_text().splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        if record.get("action") != "FORECAST_AUDIT":
            continue
        details = record.get("details") or {}
        code = details.get("product_code")
        if code and (code not in latest or record.get("timestamp", "") >= latest[code]["timestamp"]):
            latest[code] = {"timestamp": record.get("timestamp", ""), "status": record.get("status"),
                            "pending_months": details.get("pending_months", []), "reason": details.get("reason")}
    rows = [{"product_code": code, "month": str(month), "reason": item["reason"]}
            for code, item in latest.items() if item["status"] == "REVIEW_REQUIRED"
            for month in item["pending_months"]]
    return pd.DataFrame(rows, columns=["product_code", "month", "reason"])

@st.cache_data(ttl=300)
def load_prescribing_data():
    blob = storage.Client().bucket(BUCKET).blob("raw/nhs_monthly_prescribing.csv")
    df = pd.read_csv(StringIO(blob.download_as_text()), dtype={"bnf_chemical_substance_code": str})
    df["month"] = df["month"].astype(str)
    df["items"] = pd.to_numeric(df["items"], errors="coerce")
    return df

try:
    data = load_prescribing_data()
    historical_queue = load_review_queue()
    latest_reviews = load_latest_reviews()
    observed = data[["bnf_chemical_substance_code", "month", "items"]].rename(
        columns={"bnf_chemical_substance_code": "product_code"})
    queue = historical_queue.merge(observed, on=["product_code", "month"], how="left", validate="one_to_one")
    def reviewed(row):
        r = latest_reviews.get((row["product_code"], row["month"]))
        return (r is not None and pd.notna(row["items"]) and
                r.get("observation_hash") == observation_hash(row["product_code"], row["month"], row["items"]))
    valid_count = int(queue.apply(reviewed, axis=1).sum()) if len(queue) else 0
    queue = queue.loc[~queue.apply(reviewed, axis=1)].reset_index(drop=True) if len(queue) else queue
except Exception as exc:
    st.error(f"Unable to load review queue: {exc}")
    st.stop()

c1, c2, c3, c4 = st.columns(4)
c1.metric("Pending observations", len(queue))
c2.metric("Medicines awaiting review", queue["product_code"].nunique())
c3.metric("Flagged months", queue["month"].nunique())
c4.metric("Reviewed from queue", valid_count)
st.subheader("Review queue")
st.dataframe(queue, use_container_width=True, hide_index=True)
if queue.empty:
    st.success("No pending observations in the current audit queue.")
    st.stop()

st.divider()
st.subheader("Observation analysis")
selected_code = st.selectbox("Select medicine", sorted(queue["product_code"].unique()))
medicine = data.loc[data["bnf_chemical_substance_code"] == selected_code].sort_values("month")
st.line_chart(medicine.set_index("month")["items"], x_label="Month", y_label="Prescribing items")
flagged = queue.loc[queue["product_code"] == selected_code]
st.dataframe(flagged, use_container_width=True, hide_index=True)

st.divider()
st.subheader("Submit HITL decision")
if reviewer is None:
    st.warning("Verified IAP identity required for submissions.")
else:
    with st.form("hitl_review"):
        selected_month = st.selectbox("Flagged month", flagged["month"].tolist())
        row = flagged.loc[flagged["month"] == selected_month].iloc[0]
        st.write(f"Observed prescribing items: {row['items']:,.0f}" if pd.notna(row["items"]) else "Observation unavailable")
        decision = st.radio("Decision", ["APPROVED", "REJECTED"], horizontal=True)
        justification = st.text_area("Scientific justification", max_chars=5000)
        submitted = st.form_submit_button("Save decision", type="primary")
    if submitted:
        # Revalidate identity on the submission rerun; do not accept a client-supplied name.
        verified = get_verified_reviewer()
        if not verified:
            st.error("IAP identity verification failed; no decision saved.")
        elif pd.isna(row["items"]):
            st.error("Observation is unavailable; no decision saved.")
        else:
            try:
                submit_review_once(selected_code, selected_month, decision, verified["email"],
                                   justification, row["items"])
                st.cache_data.clear()
                st.success("Decision saved to GCS.")
                st.rerun()
            except ValueError as exc:
                st.error(str(exc))
            except Exception:
                st.error("Submission failed. Check Cloud Run logs; no success was confirmed.")
