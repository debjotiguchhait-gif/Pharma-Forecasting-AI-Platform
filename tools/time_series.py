
import pandas as pd


from tools.anomaly_detection import detect_yoy_anomalies
from tools.hitl_review import get_latest_review, observation_hash

def prepare_training_window(series):
    """
    Fill excluded/missing observations using only
    information available within the training window.

    Does not modify the original series.
    """
    clean = series.astype(float).copy()

    if clean.notna().sum() < 24:
        raise ValueError("Insufficient valid observations")

    # Use preceding observations only.
    clean = clean.ffill()

    if clean.isna().any():
        raise ValueError(
            "Missing observations at beginning of history"
        )

    return clean

def classify_missing_gaps(series):
    missing = series.isna()
    missing_count = int(missing.sum())

    if missing_count == 0:
        return "COMPLETE"

    longest_gap = int(
        missing.astype(int)
        .groupby((~missing).cumsum())
        .sum()
        .max()
    )

    if missing_count <= 3 and longest_gap <= 2:
        return "ISOLATED_GAPS"

    if missing_count <= 6 and longest_gap <= 3:
        return "MODERATE_GAPS"

    return "EXTENDED_GAPS"

def prepare_model_data(file_path, product_code):
    prepared = prepare_time_series(file_path, product_code)
    series = prepared["series"].copy()

    gap_category = classify_missing_gaps(series)

    if gap_category in ("MODERATE_GAPS", "EXTENDED_GAPS"):
        return {
            "status": "BLOCKED",
            "reason": "Missing monthly observations require treatment",
            "gap_category": gap_category,
            "series": None,
        }

    anomalies = detect_yoy_anomalies(series, threshold_pct=50)

    pending = []
    excluded = []
    approved = []

    for anomaly in anomalies["anomalies"]:
        month = anomaly["month"]

        review = get_latest_review(product_code, month)

        current_hash = observation_hash(
            product_code,
            month,
            anomaly["actual_items"]
        )

        if (
            review is None
            or review.get("observation_hash") != current_hash
        ):
            pending.append(month)

        elif review["decision"] == "APPROVED":
            approved.append(month)

        elif review["decision"] == "REJECTED":
            excluded.append(month)
            series.loc[month] = float("nan")

    if pending:
        return {
            "status": "REVIEW_REQUIRED",
            "reason": "Unreviewed anomalies detected",
            "pending_months": pending,
            "series": None
        }

    return {
        "status": "READY",
        "product_code": product_code,
        "series": series,
        "approved_months": approved,
        "excluded_months": excluded,
        "missing_months": int(series.isna().sum())
    }

def prepare_time_series(file_path, product_code):
    df = pd.read_csv(
        file_path,
        dtype={"bnf_chemical_substance_code": str}
    )

    df["month"] = pd.to_datetime(df["month"])
    df["items"] = pd.to_numeric(df["items"], errors="coerce")

    product = df.loc[
        df["bnf_chemical_substance_code"] == str(product_code),
        ["month", "items"]
    ].copy()

    if product.empty:
        raise ValueError(f"Product not found: {product_code}")

    if product["month"].isna().any() or product["items"].isna().any():
        raise ValueError("Invalid dates or prescription values")

    if product["month"].dt.to_period("M").duplicated().any():
        raise ValueError("Duplicate product-month observations")

    product["month"] = product["month"].dt.to_period("M")
    product = product.set_index("month").sort_index()

    full_index = pd.period_range(
        product.index.min(),
        product.index.max(),
        freq="M"
    )

    series = product["items"].reindex(full_index)

    return {
        "product_code": str(product_code),
        "observed_months": int(series.notna().sum()),
        "calendar_months": len(series),
        "missing_months": int(series.isna().sum()),
        "first_month": str(series.index.min()),
        "last_month": str(series.index.max()),
        "continuous_history": bool(series.notna().all()),
        "series": series
    }
