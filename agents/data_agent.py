
from pathlib import Path
import pandas as pd


def validate_data(file_path):
    path = Path(file_path)

    if not path.is_file():
        return {
            "status": "FAILED",
            "error": f"Dataset not found: {path}"
        }

    try:
        df = pd.read_csv(path)

        required_columns = {
            "month",
            "bnf_chemical_substance_code",
            "items"
        }

        missing_columns = sorted(required_columns - set(df.columns))

        result = {
            "dataset": path.name,
            "status": "VALID" if not missing_columns else "FAILED",
            "rows": len(df),
            "columns": len(df.columns),
            "products": df["bnf_chemical_substance_code"].nunique()
                if "bnf_chemical_substance_code" in df.columns else None,
            "months": df["month"].nunique()
                if "month" in df.columns else None,
            "missing_values": df.isna().sum().to_dict(),
            "duplicate_rows": int(df.duplicated().sum()),
            "missing_required_columns": missing_columns
        }

        return result

    except Exception as exc:
        return {
            "status": "FAILED",
            "error": str(exc)
        }

def assess_product_eligibility(file_path):
    import pandas as pd

    df = pd.read_csv(file_path)

    required = ["month", "bnf_chemical_substance_code", "items"]
    if any(col not in df.columns for col in required):
        raise ValueError("Missing required columns")

    df["month"] = pd.to_datetime(
        df["month"].astype(str), errors="coerce"
    )
    df["items"] = pd.to_numeric(df["items"], errors="coerce")

    invalid_dates = int(df["month"].isna().sum())
    invalid_items = int(
        (df["items"].isna() | (df["items"] < 0)).sum()
    )

    if invalid_dates:
        raise ValueError(f"{invalid_dates} invalid month values")

    if df["bnf_chemical_substance_code"].isna().any():
        raise ValueError("Missing product codes")

    df["month"] = df["month"].dt.to_period("M")

    duplicate_product_months = int(
        df.duplicated(
            ["bnf_chemical_substance_code", "month"]
        ).sum()
    )

    total_months = len(
        pd.period_range(
            df["month"].min(),
            df["month"].max(),
            freq="M"
        )
    )

    product_months = df.groupby(
        "bnf_chemical_substance_code"
    )["month"].nunique()

    # Initial eligibility rules; refine during modelling.
    complete = int((product_months == total_months).sum())
    eligible = int((product_months >= 36).sum())
    limited = int(
        ((product_months >= 12) & (product_months < 36)).sum()
    )
    insufficient = int((product_months < 12).sum())

    return {
        "total_products": int(len(product_months)),
        "total_calendar_months": total_months,
        "complete_history_products": complete,
        "products_with_missing_months": int(
            (product_months < total_months).sum()
        ),
        "provisionally_eligible_36_plus_months": eligible,
        "limited_history_12_to_35_months": limited,
        "insufficient_history_below_12_months": insufficient,
        "duplicate_product_months": duplicate_product_months,
        "invalid_item_values": invalid_items
    }
