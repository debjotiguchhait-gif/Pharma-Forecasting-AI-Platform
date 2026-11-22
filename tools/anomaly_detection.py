
import pandas as pd


def detect_yoy_anomalies(series, threshold_pct=50):
    """
    Detect unusually large year-over-year changes.

    Returns review flags without modifying observations.
    """
    if threshold_pct <= 0:
        raise ValueError("Threshold must be positive")

    history = series.astype(float).copy()

    previous_year = history.shift(12)

    yoy_pct = (
        (history - previous_year)
        / previous_year.replace(0, float("nan"))
    ) * 100

    missing_yoy_comparisons = int(
        (
            history.iloc[12:].isna()
            | previous_year.iloc[12:].isna()
        ).sum()
    )
    anomalies = []

    for month, change in yoy_pct.items():
        if pd.notna(change) and abs(change) > threshold_pct:
            anomalies.append({
                "month": str(month),
                "actual_items": round(float(history.loc[month]), 2),
                "previous_year_items": round(
                    float(previous_year.loc[month]), 2
                ),
                "yoy_change_pct": round(float(change), 2),
                "status": "REVIEW_REQUIRED"
            })

    return {
        "product_code": None,
        "threshold_pct": threshold_pct,
        "anomaly_count": len(anomalies),
        "anomalies": anomalies,
        "missing_yoy_comparisons": missing_yoy_comparisons
    }
