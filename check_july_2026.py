
import pandas as pd

df = pd.read_csv(
    "nhs_monthly_prescribing.csv",
    dtype={"bnf_chemical_substance_code": str}
)

df["items"] = pd.to_numeric(df["items"], errors="coerce")

# Aggregate any duplicate product-month records.
data = df[df["month"].isin(["2025-07", "2026-07"])]

pivot = data.pivot_table(
    index="bnf_chemical_substance_code",
    columns="month",
    values="items",
    aggfunc="sum"
)

# Compare only medicines present in both months.
pivot = pivot.dropna(subset=["2025-07", "2026-07"])
pivot = pivot[pivot["2025-07"] > 0]

pivot["yoy_pct"] = (
    (pivot["2026-07"] / pivot["2025-07"] - 1) * 100
)

total = len(pivot)
drop_30 = (pivot["yoy_pct"] < -30).sum()
drop_50 = (pivot["yoy_pct"] < -50).sum()

print("Medicines compared:", total)
print("Decline greater than 30%:", drop_30)
print("Decline greater than 50%:", drop_50)
print("Median YoY change:", round(pivot["yoy_pct"].median(), 2), "%")

print("\nTop 15 declines:")
print(
    pivot.sort_values("yoy_pct")
    .head(15)
    .round(2)
    .to_string()
)

print("\nTotal prescribing items in matched medicines:")
print("July 2025:", int(pivot["2025-07"].sum()))
print("July 2026:", int(pivot["2026-07"].sum()))
