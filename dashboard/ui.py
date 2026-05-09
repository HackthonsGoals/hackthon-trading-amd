from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st


SIGNAL_COLORS = {
    "BUY": "background-color: #0f766e; color: white;",
    "SELL": "background-color: #b91c1c; color: white;",
    "HOLD": "background-color: #475569; color: white;",
}


def _style_signals(frame: pd.DataFrame):
    return frame.style.map(lambda value: SIGNAL_COLORS.get(value, ""), subset=["signal"])


def render_dashboard(
    market_data: pd.DataFrame,
    signals: list[dict],
    trades: list[dict],
    inference_metrics: dict,
    benchmark: dict,
    headlines: pd.DataFrame,
    sentiment_records: list[dict],
    sentiment_benchmark: dict,
    trade_summary: dict,
) -> None:
    st.set_page_config(page_title="AMD AI Trading Demo", layout="wide")
    st.title("AMD GPU-Accelerated AI Signal Pipeline")
    st.caption('Agentic Pipeline: Signal Agent → Sentiment Agent → Reasoning Agent (Qwen3-8B on AMD)')
    st.caption(
        "Open-weight sentiment + batch inference + fake execution. Built for AMD GPU visibility, not real trading."
    )

    signal_frame = pd.DataFrame(signals)
    sentiment_frame = pd.DataFrame(sentiment_records)
    trade_frame = pd.DataFrame(trades)

    best_speedup = benchmark.get("best_speedup")
    cols = st.columns(5)
    cols[0].metric("Market Device", inference_metrics["device"])
    cols[1].metric("Market Latency", f"{inference_metrics['latency_ms']:.3f} ms")
    cols[2].metric("Signals/sec", f"{inference_metrics['throughput_rows_per_second']:.0f}")
    cols[3].metric("GPU Speedup", f"{best_speedup:.2f}x" if best_speedup else "GPU pending")
    cols[4].metric("Demo P&L", f"{trade_summary.get('total_pnl', 0):.2f}")

    st.subheader("Live Signals")
    st.dataframe(_style_signals(signal_frame), use_container_width=True, hide_index=True)
    if "llm_reason" in signal_frame.columns:
        st.subheader("AI Signal Reasoning")
        for _, row in signal_frame.iterrows():
            color = "🟢" if row['signal'] == 'BUY' else "🔴" if row['signal'] == 'SELL' else "🟡"
            st.markdown(f"{color} **{row['symbol']} — {row['signal']}** ({row['confidence']:.1%} confidence): {row['llm_reason']}")

    st.subheader("Headline Sentiment")
    if sentiment_frame.empty:
        st.info("No headlines loaded.")
    else:
        chart_left, chart_right = st.columns(2)
        with chart_left:
            st.dataframe(
                sentiment_frame[
                    ["timestamp", "symbol", "text", "sentiment", "score", "signed_score", "backend", "device"]
                ],
                use_container_width=True,
                hide_index=True,
            )
        with chart_right:
            distribution = sentiment_frame["sentiment"].value_counts().rename_axis("sentiment").reset_index(name="count")
            fig = px.bar(
                distribution,
                x="sentiment",
                y="count",
                title="POSITIVE vs NEGATIVE vs NEUTRAL",
                color="sentiment",
                color_discrete_map={"POSITIVE": "#16a34a", "NEGATIVE": "#dc2626", "NEUTRAL": "#64748b"},
            )
            st.plotly_chart(fig, use_container_width=True)

        fig = px.line(
            sentiment_frame,
            x="timestamp",
            y="signed_score",
            color="symbol",
            markers=True,
            title="Sentiment Score Over Time",
        )
        st.plotly_chart(fig, use_container_width=True)

    left, right = st.columns([1.2, 1])
    with left:
        st.subheader("Market Data")
        st.line_chart(market_data.pivot_table(index="timestamp", columns="symbol", values="close"))

    with right:
        st.subheader("Trade Simulation")
        sim_cols = st.columns(3)
        sim_cols[0].metric("Closed Trades", trade_summary.get("trades", 0))
        sim_cols[1].metric("Win Rate", f"{trade_summary.get('win_rate', 0) * 100:.1f}%")
        sim_cols[2].metric("Avg Return", f"{trade_summary.get('avg_return_pct', 0):.3f}%")

    st.subheader("Performance Metrics")
    market_records = pd.DataFrame(benchmark.get("records", []))
    sentiment_records_df = pd.DataFrame(sentiment_benchmark.get("records", []))
    perf_left, perf_right = st.columns(2)
    with perf_left:
        if market_records.empty:
            st.info("Market benchmark data unavailable.")
        else:
            fig = px.bar(
                market_records,
                x="batch_size",
                y="throughput_signals_per_second",
                color="device_type",
                barmode="group",
                title="Market Batch Throughput",
            )
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(market_records, use_container_width=True, hide_index=True)
    with perf_right:
        if sentiment_records_df.empty:
            st.info("Sentiment benchmark data unavailable.")
        else:
            fig = px.bar(
                sentiment_records_df,
                x="device_type",
                y="batch_throughput_texts_per_second",
                color="device_type",
                title="Sentiment Batch Throughput",
            )
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(sentiment_records_df, use_container_width=True, hide_index=True)

    st.subheader("Trade Lifecycle")
    if trade_frame.empty:
        st.info("No BUY/SELL trades were generated in this run.")
        return

    trade_frame["cumulative_pnl"] = trade_frame["pnl"].cumsum()
    st.dataframe(trade_frame, use_container_width=True, hide_index=True)
    fig = px.line(trade_frame, x="closed_at", y="cumulative_pnl", markers=True, title="Cumulative Simulated P&L")
    st.plotly_chart(fig, use_container_width=True)
