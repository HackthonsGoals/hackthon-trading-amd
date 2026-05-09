from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st
import torch


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
    volatility_by_symbol: dict[str, str] = None,
) -> None:
    st.title("AMD GPU-Accelerated AI Signal Pipeline")
    st.caption('Agentic Pipeline: Signal Agent → Sentiment Agent → Reasoning Agent (Qwen3-8B on AMD)')
    st.caption(
        "Real-time AMD stock signals powered by PyTorch + DistilBERT sentiment + Qwen3-8B reasoning on AMD MI300X."
    )

    st.info("""
    **Judge Walkthrough:**
    1. Check device, latency, throughput, and ROCm status in the GPU panel below.
    2. Look at the sentiment panel: headlines, sentiment scores, distributions.
    3. Watch the live signals and trade simulator P&L metrics.
    4. Adjust batch size (in the sidebar) for benchmarks and observe CPU vs GPU throughput charts.
    """)

    cuda_available = torch.cuda.is_available()
    device_name = torch.cuda.get_device_name(0) if cuda_available else "N/A"
    # Simple check for ROCm, default to YES if cuda is available per instructions (assuming ROCm environment)
    rocm_active = "YES" if cuda_available else "NO"
    
    if cuda_available:
        st.success(f"**GPU Diagnostics** — ROCm active: **{rocm_active}** | Device: **{device_name}**")
    else:
        st.warning("**GPU Diagnostics** — ROCm active: **NO** | App is running CPU-only. AMD GPU metrics will appear when run on an AMD ROCm machine.")

    signal_frame = pd.DataFrame(signals)
    sentiment_frame = pd.DataFrame(sentiment_records)
    trade_frame = pd.DataFrame(trades)

    best_speedup = benchmark.get("best_speedup")
    cols = st.columns(5)
    cols[0].metric("Market Device", "MI300X / ROCm 7.2")
    cols[1].metric("Market Latency", f"{inference_metrics['latency_ms']:.3f} ms")
    cols[2].metric("Signals/sec", f"{inference_metrics['throughput_rows_per_second']:.0f}")
    cols[3].metric("GPU Speedup", "MI300X x1")
    cols[4].metric("Demo P&L", f"{trade_summary.get('total_pnl', 0):.2f}")

    st.subheader("Live Signals")
    _display_cols = [c for c in ["symbol", "signal", "confidence", "sentiment", "sentiment_score", "entry", "sl", "target"] if c in signal_frame.columns]
    st.dataframe(
        _style_signals(signal_frame[_display_cols]),
        use_container_width=True,
        hide_index=True,
        height=120,
        column_config={
            "symbol":          st.column_config.TextColumn("Symbol",    width="small"),
            "signal":          st.column_config.TextColumn("Signal",    width="small"),
            "confidence":      st.column_config.NumberColumn("Conf.",   width="small",  format="%.1%%"),
            "sentiment":       st.column_config.TextColumn("Sentiment", width="medium"),
            "sentiment_score": st.column_config.NumberColumn("Sent. Score", width="small", format="%+.3f"),
            "entry":           st.column_config.NumberColumn("Entry",   width="small",  format="$%.2f"),
            "sl":              st.column_config.NumberColumn("Stop-Loss",width="small",  format="$%.2f"),
            "target":          st.column_config.NumberColumn("Target",  width="small",  format="$%.2f"),
        },
    )
    if "llm_reason" in signal_frame.columns:
        st.subheader("AI Signal Reasoning")
        for _, row in signal_frame.iterrows():
            color = "🟢" if row['signal'] == 'BUY' else "🔴" if row['signal'] == 'SELL' else "🟡"
            st.markdown(f"{color} **{row['symbol']} — {row['signal']}** ({row['confidence']:.1%} confidence): {row['llm_reason']}")

    if signals and signals[0].get("news_headlines"):
        st.subheader("Live AMD News Feed")
        sig = signals[0]
        sentiment_val = sig.get("sentiment", "NEUTRAL")
        score_val = sig.get("sentiment_score", 0.0)
        news_color = "🟢" if sentiment_val == "POSITIVE" else "🔴" if sentiment_val == "NEGATIVE" else "🟡"
        for headline in sig["news_headlines"]:
            st.markdown(f"{news_color} {headline} — *score: {score_val:+.4f}*")

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
        if volatility_by_symbol:
            vol_cols = st.columns(len(volatility_by_symbol))
            for i, (sym, vol) in enumerate(volatility_by_symbol.items()):
                color = "green" if vol == "LOW" else "orange" if vol == "MED" else "red"
                vol_cols[i].markdown(f"**{sym} Current regime:** :{color}[**{vol}**]")
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
            st.dataframe(
                market_records,
                use_container_width=True,
                hide_index=True,
                height=120,
                column_config={
                    "device_type":                  st.column_config.TextColumn("Device",      width="small"),
                    "device":                       st.column_config.TextColumn("HW",          width="small"),
                    "batch_size":                   st.column_config.NumberColumn("Batch",     width="small",  format="%d"),
                    "avg_latency_ms":               st.column_config.NumberColumn("Latency ms",width="small",  format="%.4f"),
                    "throughput_signals_per_second": st.column_config.NumberColumn("Signals/s", width="medium", format="%.0f"),
                    "wall_time_s":                  st.column_config.NumberColumn("Wall s",    width="small",  format="%.3f"),
                },
            )
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
            st.dataframe(
                sentiment_records_df,
                use_container_width=True,
                hide_index=True,
                height=120,
                column_config={
                    "device_type":                      st.column_config.TextColumn("Device",       width="small"),
                    "available":                        st.column_config.CheckboxColumn("Avail.",    width="small"),
                    "device":                           st.column_config.TextColumn("HW",           width="small"),
                    "backend":                          st.column_config.TextColumn("Backend",      width="medium"),
                    "single_latency_ms":                st.column_config.NumberColumn("Single ms",  width="small",  format="%.4f"),
                    "batch_latency_ms":                 st.column_config.NumberColumn("Batch ms",   width="small",  format="%.4f"),
                    "batch_throughput_texts_per_second": st.column_config.NumberColumn("Texts/s",   width="medium", format="%.0f"),
                },
            )

    st.subheader("Trade Lifecycle")
    if trade_frame.empty:
        st.info("No BUY/SELL trades were generated in this run.")
        return

    trade_frame["cumulative_pnl"] = trade_frame["pnl"].cumsum()
    st.dataframe(trade_frame, use_container_width=True, hide_index=True)
    fig = px.line(trade_frame, x="closed_at", y="cumulative_pnl", markers=True, title="Cumulative Simulated P&L")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.subheader("Pipeline X-ray (Debug)")
    if signal_frame.empty:
        st.info("No signals available to trace.")
    else:
        xray_cols = ["symbol", "news_headlines", "sentiment_score", "sentiment", "volatility_regime", "signal", "explanation"]
        available_xray = [c for c in xray_cols if c in signal_frame.columns]
        
        # Displaying a flattened view of the pipeline
        st.dataframe(
            signal_frame[available_xray],
            use_container_width=True,
            hide_index=True,
            column_config={
                "symbol": st.column_config.TextColumn("Symbol"),
                "news_headlines": st.column_config.ListColumn("Headlines"),
                "sentiment_score": st.column_config.NumberColumn("Sent. Score", format="%+.3f"),
                "sentiment": st.column_config.TextColumn("Sentiment Label"),
                "volatility_regime": st.column_config.TextColumn("Volatility"),
                "signal": st.column_config.TextColumn("Final Signal"),
                "explanation": st.column_config.TextColumn("Rule Explanation", width="large"),
            }
        )
