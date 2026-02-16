import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import os

st.set_page_config(
    page_title="Quant Portfolio Optimization Dashboard",
    layout="wide"
)

st.title("📊 Quantitative Portfolio Optimization Dashboard")

st.markdown("---")

DATA_PATH = "data/processed/"

# -------------------------------------------------
# Helper Function
# -------------------------------------------------
def load_csv(filename):
    path = os.path.join(DATA_PATH, filename)
    if os.path.exists(path):
        return pd.read_csv(path)
    else:
        return None


# -------------------------------------------------
# Portfolio Weights Section
# -------------------------------------------------
st.header("📌 Portfolio Allocation")

weights = load_csv("portfolio_weights.csv")

if weights is not None:
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Optimized Weights")
        st.dataframe(weights)

    with col2:
        fig, ax = plt.subplots()
        ax.pie(
            weights["Weight"],
            labels=weights["Asset"],
            autopct="%1.1f%%"
        )
        ax.set_title("Portfolio Allocation")
        st.pyplot(fig)

else:
    st.warning("Portfolio weights file not found.")


st.markdown("---")

# -------------------------------------------------
# Performance Metrics Section
# -------------------------------------------------
st.header("📈 Performance Metrics")

metrics = load_csv("backtest_metrics.csv")

if metrics is not None:
    st.dataframe(metrics)

    # Display key metrics in big format
    if "Sharpe Ratio" in metrics.columns:
        sharpe = metrics["Sharpe Ratio"].values[0]
    else:
        sharpe = None

    if "Annual Return" in metrics.columns:
        annual_return = metrics["Annual Return"].values[0]
    else:
        annual_return = None

    if "Max Drawdown" in metrics.columns:
        max_dd = metrics["Max Drawdown"].values[0]
    else:
        max_dd = None

    col1, col2, col3 = st.columns(3)

    if sharpe is not None:
        col1.metric("Sharpe Ratio", round(sharpe, 3))

    if annual_return is not None:
        col2.metric("Annual Return", f"{round(annual_return*100,2)}%")

    if max_dd is not None:
        col3.metric("Max Drawdown", f"{round(max_dd*100,2)}%")

else:
    st.warning("Backtest metrics file not found.")


st.markdown("---")

# -------------------------------------------------
# Backtest Performance Section
# -------------------------------------------------
st.header("📊 Backtest Cumulative Performance")

cumulative = load_csv("backtest_cumulative.csv")

if cumulative is not None:

    if "Date" in cumulative.columns:
        cumulative["Date"] = pd.to_datetime(cumulative["Date"])

    fig2, ax2 = plt.subplots()

    for col in cumulative.columns:
        if col != "Date":
            if "Date" in cumulative.columns:
                ax2.plot(cumulative["Date"], cumulative[col], label=col)
            else:
                ax2.plot(cumulative[col], label=col)

    ax2.set_title("Cumulative Returns")
    ax2.legend()
    st.pyplot(fig2)

else:
    st.warning("Backtest cumulative file not found.")


st.markdown("---")

# -------------------------------------------------
# Forecast Section
# -------------------------------------------------
st.header("🔮 Forecast Results")

lstm = load_csv("tsla_lstm_forecast.csv")
arima = load_csv("tsla_arima_forecast.csv")

if lstm is not None or arima is not None:

    fig3, ax3 = plt.subplots()

    if lstm is not None:
        if "Date" in lstm.columns:
            lstm["Date"] = pd.to_datetime(lstm["Date"])
            ax3.plot(lstm["Date"], lstm["Forecast"], label="LSTM Forecast")
        else:
            ax3.plot(lstm["Forecast"], label="LSTM Forecast")

    if arima is not None:
        if "Date" in arima.columns:
            arima["Date"] = pd.to_datetime(arima["Date"])
            ax3.plot(arima["Date"], arima["Forecast"], label="ARIMA Forecast")
        else:
            ax3.plot(arima["Forecast"], label="ARIMA Forecast")

    ax3.set_title("Forecast Comparison")
    ax3.legend()
    st.pyplot(fig3)

else:
    st.info("Forecast files not found.")


st.markdown("---")

st.success("Dashboard loaded successfully.")
