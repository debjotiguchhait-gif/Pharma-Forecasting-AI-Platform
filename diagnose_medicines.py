
import pandas as pd

DATA = "nhs_monthly_prescribing.csv"

CODES = [
    "0202030C0",
    "020400080",
    "0204000AB",
    "0204000E0",
]

df = pd.read_csv(
    DATA,
    dtype={"bnf_chemical_substance_code": str}
)
df["month"] = pd.to_datetime(df["month"], errors="coerce")
df["items"] = pd.to_numeric(df["items"], errors="coerce")

for code in CODES:
    data = df[df["bnf_chemical_substance_code"] == code].copy()
    data = data.dropna(subset=["month", "items"])
    data = data.sort_values("month")

    print(f"\n{'=' * 55}")
    print(f"MEDICINE: {code}")
    print(f"Available months: {data['month'].nunique()}")

    if data.empty:
        print("No valid observations")
        continue

    series = data.groupby("month")["items"].sum()
    calendar = pd.date_range(
        series.index.min(), series.index.max(), freq="MS"
    )
    series = series.reindex(calendar)

    missing = series[series.isna()]
    print(f"Missing months: {len(missing)}")
    print("Missing month list:", missing.index.strftime("%Y-%m").tolist())

    previous_year = series.shift(12)
    yoy = (series - previous_year).abs() / previous_year.replace(0, float("nan")) * 100
    anomalies = yoy[yoy > 50]

    print(f"YoY changes above 50%: {len(anomalies)}")

    for month, change in anomalies.items():
        print(
            month.strftime("%Y-%m"),
            f"change={change:.1f}%",
            f"items={series.loc[month]:.0f}"
        )
