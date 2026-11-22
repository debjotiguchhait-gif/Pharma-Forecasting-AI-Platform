def generate_business_insights(forecast):
    values = list(forecast.values())

    if not values:
        return {"status": "NO_FORECAST"}

    total = sum(values)
    average = total / len(values)

    first_half = sum(values[:6])
    second_half = sum(values[6:])

    growth_pct = (
        (second_half - first_half) / first_half * 100
        if first_half else None
    )

    return {
        "status": "SUCCESS",
        "forecast_total": round(total),
        "monthly_average": round(average),
        "half_year_growth_pct": (
            round(growth_pct, 2)
            if growth_pct is not None else None
        ),
        "peak_month": max(forecast, key=forecast.get),
        "lowest_month": min(forecast, key=forecast.get),
        "demand_trend": (
            "INCREASING" if growth_pct is not None and growth_pct > 2
            else "DECLINING" if growth_pct is not None and growth_pct < -2
            else "STABLE"
        ),
    }