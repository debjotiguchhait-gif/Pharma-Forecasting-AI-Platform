
import pandas as pd
import streamlit as st
from chatbot import answer_question
from dashboard_data import (
    get_products,
    get_forecast,
    get_history,
    get_model_performance,
    get_medicine_catalog,
)

st.set_page_config(
    page_title="Pharma Forecast Intelligence",
    layout="wide"
)

st.markdown("""
<style>
.block-container {
    max-width: 1350px;
    padding-top: 2rem;
}

h1, h2, h3 {
    letter-spacing: -0.5px;
}

div[data-testid="stMetric"] {
    background: rgba(120, 140, 170, 0.08);
    border: 1px solid rgba(120, 140, 170, 0.20);
    border-radius: 12px;
    padding: 16px;
}

div[data-testid="stTabs"] {
    margin-top: 1rem;
}
</style>
""", unsafe_allow_html=True)

st.title("Pharma Forecast Intelligence")
st.caption("Cloud-based pharmaceutical forecasting decision support")

@st.cache_data(ttl=300)
def load_products():
    return get_products()

products = load_products()

if not products:
    st.warning("No saved forecasts available.")
    st.stop()


@st.cache_data(ttl=300)
def load_catalog():
    return get_medicine_catalog()

catalog = load_catalog()

product = st.selectbox(
    "Search medicine by name or BNF code",
    options=products,
    format_func=lambda code: catalog.get(code, code),
    placeholder="Type medicine name or code...",
)

forecast_data = get_forecast(product)
model_data = get_model_performance(product)
history = get_history(product)

tab1, tab2, tab3 = st.tabs([
    "Forecast Dashboard",
    "Model Intelligence",
    "Ask AI"
])

with tab1:
    st.subheader("Historical prescribing data")
    if not history.empty:
        historical = history.copy()
        historical["month"] = pd.to_datetime(historical["month"])
        st.line_chart(
            historical.set_index("month")["items"]
        )

    st.subheader("Saved 12-month forecast")

    if forecast_data:
        result = forecast_data.get("forecast_result", {})
        predictions = result.get("forecast", {})

        if predictions:
            forecast_df = pd.DataFrame(
                list(predictions.items()),
                columns=["month", "forecast_items"]
            )
            forecast_df["month"] = pd.to_datetime(
                forecast_df["month"]
            )
            st.line_chart(
                forecast_df.set_index("month")["forecast_items"]
            )
            st.dataframe(forecast_df, hide_index=True)
        else:
            st.info("No saved forecast values for this medicine.")

with tab2:
    st.subheader("Model Intelligence")

    if model_data:
        c1, c2 = st.columns(2)

        c1.metric("Champion", model_data["champion"])
        c2.metric("Challenger", model_data["challenger"] or "N/A")

        comparison = pd.DataFrame([
            {
                "Model": "Champion",
                **model_data["champion_metrics"]
            },
            {
                "Model": "Challenger",
                **model_data["challenger_metrics"]
            }
        ])

        st.dataframe(comparison, hide_index=True)
        st.caption(f"Registry version: {model_data['version']}")
    else:
        st.info("No saved model evaluation found.")


with tab3:
    st.subheader("Ask Forecast AI")
    st.caption(
        "Answers are restricted to stored forecasts "
        "and model evaluation results."
    )

    question = st.chat_input(
        "Ask about forecasts or model performance..."
    )

    if question:
        with st.chat_message("user"):
            st.write(question)

        with st.chat_message("assistant"):
            with st.spinner("Checking stored records..."):
                try:
                    answer = answer_question(
                        question,
                        product,
                        forecast_data,
                        model_data,
                    )
                    st.write(answer)
                except Exception as exc:
                    st.error(
                        "The chatbot could not retrieve an answer. "
                        f"Error: {exc}"
                    )
