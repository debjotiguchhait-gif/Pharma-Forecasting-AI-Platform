import numpy as np
import pandas as pd

from tools.time_series import prepare_time_series


DATA_PATH = "data/raw/nhs_monthly_prescribing.csv"


def validate_imputation(product_code):
    prepared = prepare_time_series(DATA_PATH, product_code)
    series = prepared["series"].astype(float)

    missing = series.isna()
    longest_gap = int(
        missing.astype(int)
        .groupby((~missing).cumsum())
        .sum()
        .max()
    )

    if longest_gap not in (1, 2):
        return {
            "status": "UNSUPPORTED_GAP_PATTERN",
            "product_code": product_code,
        }

    errors = {
        "forward_fill": [],
        "linear_interpolation": [],
    }

    # Evaluate only windows with known actual observations.
    # Restrict each experiment to its available history.
    for end in range(36, len(series) + 1):
        history = series.iloc[:end].copy()

        if len(history) < longest_gap + 2:
            continue

        actual = history.iloc[-longest_gap:].copy()

        if actual.isna().any():
            continue

        masked = history.copy()
        masked.iloc[-longest_gap:] = float("nan")

        predictions = {
            "forward_fill": masked.ffill().iloc[-longest_gap:],
            "linear_interpolation": masked.interpolate(
                method="linear"
            ).iloc[-longest_gap:],
        }

        for method, predicted in predictions.items():
            if predicted.isna().any():
                continue

            errors[method].extend(
                abs(actual.values - predicted.values).tolist()
            )

    mae = {
        method: round(sum(values) / len(values), 2)
        for method, values in errors.items()
        if values
    }

    if not mae:
        return {
            "status": "INSUFFICIENT_VALIDATION_DATA",
            "product_code": product_code,
        }

    return {
        "status": "VALIDATED",
        "product_code": product_code,
        "gap_length": longest_gap,
        "mae": mae,
        "recommended_method": min(mae, key=mae.get),
    }

if __name__ == "__main__":
    result = validate_imputation("0202030C0")
    print(result)