def monitor_forecast(
    latest_actual_month,
    last_training_month,
    current_wape=None,
    pending_anomalies=None,
    wape_threshold=10.0,
):
    pending_anomalies = pending_anomalies or []

    if pending_anomalies:
        return {
            "status": "HITL_REQUIRED",
            "retrain": False,
            "reason": "Pending anomaly review",
        }

    if current_wape is not None and current_wape > wape_threshold:
        return {
            "status": "REASSESSMENT_REQUIRED",
            "retrain": True,
            "reason": "Forecast performance degradation",
        }

    if latest_actual_month > last_training_month:
        return {
            "status": "REASSESSMENT_REQUIRED",
            "retrain": True,
            "reason": "New monthly data available",
        }

    return {
        "status": "CURRENT",
        "retrain": False,
        "reason": "No reassessment required",
    }