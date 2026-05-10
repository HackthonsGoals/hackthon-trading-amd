from __future__ import annotations

import time

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
**Demo Guide**
1. Inspect device, latency, throughput, and ROCm status in the GPU panel below.
2. Review news sentiment: headlines, sentiment scores, and distributions.
3. Examine live signals and simulated P&L metrics.
4. Adjust batch size in the sidebar and observe CPU vs GPU throughput in Performance Metrics.
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
    
    # Honest hardware reporting. Shows CPU if running locally without ROCm, prevents fake "GPU Speedup" claims.
    device_str = "GPU (AMD-ready, via PyTorch)" if "cuda" in str(inference_metrics.get("device", "")).lower() else "CPU (No GPU detected)"
    
    cols[0].metric("Market Device", device_str)
    cols[1].metric("Market Latency", f"{inference_metrics['latency_ms']:.3f} ms")
    cols[2].metric("Signals/sec", f"{inference_metrics['throughput_rows_per_second']:.0f}")
    cols[3].metric("GPU Speedup", f"{best_speedup:.2f}x" if best_speedup else "Pending (No GPU)")
    cols[4].metric("Demo P&L", f"{trade_summary.get('total_pnl', 0):.2f}")

    st.caption(
        f"Last updated: {pd.Timestamp.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC"
    )

    st.subheader("Live Signals & AI Reasoning")
    if volatility_by_symbol:
        signal_frame["volatility"] = signal_frame["symbol"].map(
            lambda s: volatility_by_symbol.get(s, "N/A")
        )
    _display_cols = [c for c in ["symbol", "signal", "confidence", "sentiment", "sentiment_score", "volatility", "entry", "sl", "target"] if c in signal_frame.columns]
    st.dataframe(
        _style_signals(signal_frame[_display_cols]),
        use_container_width=True,
        hide_index=True,
        height=350,  # Increased to show the full tech sector (8 tickers) without scrolling
        column_config={
            "symbol":          st.column_config.TextColumn("Symbol",    width="small"),
            "signal":          st.column_config.TextColumn("Signal",    width="small"),
            "confidence":      st.column_config.NumberColumn("Conf.",   width="small",  format="%.1f%%"),
            "sentiment":       st.column_config.TextColumn("Sentiment", width="medium"),
            "sentiment_score": st.column_config.NumberColumn("Sent. Score", width="small", format="%+.3f"),
            "volatility":      st.column_config.TextColumn("Volatility", width="small"),
            "entry":           st.column_config.NumberColumn("Entry",   width="small",  format="$%.2f"),
            "sl":              st.column_config.NumberColumn("Stop-Loss",width="small",  format="$%.2f"),
            "target":          st.column_config.NumberColumn("Target",  width="small",  format="$%.2f"),
        },
    )
    if "llm_reason" in signal_frame.columns and not signal_frame.empty:
        if "confidence" in signal_frame.columns:
            signal_frame = signal_frame.sort_values("confidence", ascending=False)
        st.caption("Showing top 5 signals by confidence for readability.")
        for _, row in signal_frame.head(5).iterrows():
            color = "🟢" if row['signal'] == 'BUY' else "🔴" if row['signal'] == 'SELL' else "🟡"
            st.markdown(
                f"{color} **{row['symbol']} — {row['signal']}** "
                f"({row['confidence']:.1%} confidence): {row['llm_reason']}"
            )

    if signals and signals[0].get("news_headlines"):
        st.subheader("Live AMD News Feed")
        sig = signals[0]
        sentiment_val = sig.get("sentiment", "NEUTRAL")
        score_val = sig.get("sentiment_score", 0.0)
        news_color = "🟢" if sentiment_val == "POSITIVE" else "🔴" if sentiment_val == "NEGATIVE" else "🟡"
        for headline in sig["news_headlines"]:
            st.markdown(f"{news_color} {headline} — *score: {score_val:+.4f}*")

    st.subheader("News Sentiment Analytics")
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

    # Sector Signal Heatmap
    st.subheader("Sector Signal Heatmap")
    if signals:
        heatmap_data = pd.DataFrame(signals)[["symbol", "signal", "confidence", "sentiment"]]

        def _color_signal(val: str) -> str:
            if val == "BUY":
                return "background-color: #16a34a; color: white"   # green
            elif val == "SELL":
                return "background-color: #dc2626; color: white"   # red
            return "background-color: #64748b; color: white"       # neutral/hold

        # UI/UX SPECIALIST NOTE: Using 'Tint & Shade' badges. 
        # By forcing both background (tint) and text color (shade), we ensure legibility in both Dark and Light modes.
        def _color_sentiment(val: str) -> str:
            if val == "POSITIVE":
                return "background-color: #dcfce7; color: #166534;"  # emerald tint/shade
            elif val == "NEGATIVE":
                return "background-color: #fee2e2; color: #991b1b;"  # rose tint/shade
            return "background-color: #f1f5f9; color: #475569;"      # slate tint/shade

        styled_df = (
            heatmap_data.style
            .map(_color_signal, subset=["signal"])
            .map(_color_sentiment, subset=["sentiment"])
            .format({"confidence": "{:.2%}"})
        )
        st.dataframe(
            styled_df,
            use_container_width=True,
            column_config={
                "symbol": st.column_config.TextColumn("symbol", width="small"),
                "signal": st.column_config.TextColumn("signal", width="small"),
                "confidence": st.column_config.NumberColumn("confidence", width="small", format="%.2f%%"),
                "sentiment": st.column_config.TextColumn("sentiment", width="small"),
            },
        )

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
        st.subheader("Simulated Trade Summary")
        sim_cols = st.columns(3)
        sim_cols[0].metric("Closed Trades", trade_summary.get("trades", 0))
        sim_cols[1].metric("Win Rate", f"{trade_summary.get('win_rate', 0) * 100:.1f}%")
        sim_cols[2].metric("Avg Return", f"{trade_summary.get('avg_return_pct', 0):.3f}%")

    st.subheader("Performance Metrics")
    
    if best_speedup:
        st.success(f"**Best GPU Speedup**: {best_speedup:.2f}x")
    else:
        st.info("GPU results pending (no compatible device detected)")
        st.caption("Run this app on AMD Developer Cloud with ROCm to populate GPU metrics.")

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

    # --- Live GPU Inference Monitor ---
    with st.expander("⚡ Live GPU Inference Monitor", expanded=False):
        st.subheader("⚡ Live GPU Inference Monitor")
        _gpu_col1, _gpu_col2 = st.columns(2)
        with _gpu_col1:
            gpu_latency_placeholder = st.empty()
        with _gpu_col2:
            cpu_latency_placeholder = st.empty()
        inference_log = st.empty()
    
        if st.button("🔄 Run Live Benchmark (10 iterations)", key="live_gpu_benchmark_btn"):
            from engine.ai_inference import run_batch_inference as _run_batch
    
            progress_bar = st.progress(0)
            for i in range(10):
                _test_data = {
                    "symbol": ["AMD"] * 100,
                    "timestamp": pd.date_range("2024-01-01", periods=100, freq="1min"),
                    "open": 150.0,
                    "high": 152.0,
                    "low": 149.0,
                    "close": 151.0,
                    "volume": 1_000_000,
                }
                _df_test = pd.DataFrame(_test_data)
                gpu_result = _run_batch(_df_test, prefer_gpu=True)
                cpu_result = _run_batch(_df_test, prefer_gpu=False)
                gpu_latency_placeholder.metric(
                    "GPU Latency",
                    f"{gpu_result.latency_ms:.1f}ms",
                    delta=f"{gpu_result.throughput_rows_per_second:.0f} rows/s",
                )
                cpu_latency_placeholder.metric(
                    "CPU Latency",
                    f"{cpu_result.latency_ms:.1f}ms",
                    delta=f"{cpu_result.throughput_rows_per_second:.0f} rows/s",
                )
                speedup = cpu_result.latency_ms / max(gpu_result.latency_ms, 1e-9)
                inference_log.success(f"Iteration {i + 1}/10 | GPU Speedup: {speedup:.2f}x")
                progress_bar.progress((i + 1) / 10)
                time.sleep(0.5)
            st.balloons()

    st.subheader("Trade Lifecycle")
    if trade_frame.empty:
        st.info("No BUY/SELL trades were generated in this run.")
    else:
        trade_frame["cumulative_pnl"] = trade_frame["pnl"].cumsum()
        st.dataframe(trade_frame, use_container_width=True, hide_index=True)
        fig = px.line(trade_frame, x="closed_at", y="cumulative_pnl", markers=True, title="Cumulative Simulated P&L")
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    with st.expander("🔍 Pipeline X-ray (Debug)", expanded=False):
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
