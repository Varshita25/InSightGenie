# core/charts.py
import math
from typing import List, Tuple
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

sns.set_context("talk")


def _auto_fig(w=6, h=4):
    sns.set_style("whitegrid")
    fig, ax = plt.subplots(figsize=(w, h))
    return fig, ax


def numeric_histograms(df: pd.DataFrame, max_cols: int = 12) -> List[plt.Figure]:
    figs = []
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()[:max_cols]
    for c in num_cols:
        fig, ax = _auto_fig(5, 3.5)
        sns.histplot(df[c].dropna(), bins=30, kde=True, ax=ax, color="#6366f1")
        ax.set_title(f"{c} Distribution", fontsize=12, fontweight='bold', pad=10)
        ax.set_xlabel(c, fontsize=10)
        ax.set_ylabel("Frequency", fontsize=10)
        fig.tight_layout()
        figs.append(fig)
    return figs


def categorical_bars(df: pd.DataFrame, max_cols: int = 12, top_k: int = 10) -> List[plt.Figure]:
    figs = []
    cat_cols = [
        c for c in df.columns
        if not pd.api.types.is_numeric_dtype(df[c]) and not pd.api.types.is_datetime64_any_dtype(df[c])
    ]
    cat_cols = cat_cols[:max_cols]
    for c in cat_cols:
        vc = df[c].astype(str).value_counts(dropna=False).head(top_k)
        fig, ax = _auto_fig(5, 3.5)
        sns.barplot(x=vc.index, y=vc.values, ax=ax, palette="magma")
        ax.set_title(f"Top {top_k}: {c}", fontsize=12, fontweight='bold', pad=10)
        ax.set_xticks(range(len(vc)))
        ax.set_xticklabels([str(x)[:12] for x in vc.index], rotation=45, ha="right", fontsize=9)
        ax.set_ylabel("Count", fontsize=10)
        ax.set_xlabel("")
        fig.tight_layout()
        figs.append(fig)
    return figs


def correlation_heatmap(df: pd.DataFrame):
    num_df = df.select_dtypes(include=[np.number])
    if num_df.shape[1] < 2:
        return None
    corr = num_df.corr(numeric_only=True)
    # Reduced size as requested
    fig, ax = plt.subplots(figsize=(4.5, 3.5)) 
    sns.heatmap(corr, ax=ax, annot=True, fmt=".2f", cmap="vlag", center=0, annot_kws={"size": 8})
    ax.set_title("Correlation Map", fontsize=12, fontweight='bold', pad=10)
    plt.xticks(fontsize=9)
    plt.yticks(fontsize=9)
    fig.tight_layout()
    return fig


def render_figs(figs: List[plt.Figure], cols: int = 3):
    cols = max(1, cols)
    rows = math.ceil(len(figs) / cols)
    for r in range(rows):
        ccols = st.columns(cols)
        for i in range(cols):
            idx = r * cols + i
            if idx < len(figs):
                with ccols[i]:
                    st.pyplot(figs[idx], use_container_width=True)


def anomaly_plot(df: pd.DataFrame, date_col: str, value_col: str, window: int = 30, z: float = 3.0) -> Tuple[plt.Figure, pd.DataFrame]:
    s = df[[date_col, value_col]].dropna().copy()
    s[date_col] = pd.to_datetime(s[date_col], errors="coerce")
    s = s.dropna().sort_values(date_col)
    s["rolling_mean"] = s[value_col].rolling(window).mean()
    s["rolling_std"] = s[value_col].rolling(window).std()
    s["z"] = (s[value_col] - s["rolling_mean"]) / s["rolling_std"]
    anomalies = s[abs(s["z"]) >= z]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(s[date_col], s[value_col], linewidth=1.5)
    if not anomalies.empty:
        ax.scatter(anomalies[date_col], anomalies[value_col])
    ax.set_title(f"Anomaly Plot: {value_col} (window={window}, z>={z})")
    ax.set_xlabel(date_col)
    ax.set_ylabel(value_col)
    fig.tight_layout()
    return fig, anomalies


# 🆕 NEW: Prophet-based forecast plot used by Visual Builder → "Time Series Forecast"
def generate_forecast_plot(df: pd.DataFrame, date_col: str, value_col: str):
    """
    Simple Prophet forecast with Plotly overlay.
    pip install prophet plotly
    """
    from prophet import Prophet
    import plotly.graph_objects as go

    ts = df[[date_col, value_col]].dropna().copy()
    ts[date_col] = pd.to_datetime(ts[date_col], errors="coerce")
    ts = ts.dropna().sort_values(date_col)
    if ts.empty:
        raise ValueError("No valid datetime/value rows for forecast")

    m = Prophet()
    m.fit(ts.rename(columns={date_col: "ds", value_col: "y"}))
    future = m.make_future_dataframe(periods=30)  # 30 days ahead
    fcst = m.predict(future)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ts[date_col], y=ts[value_col], name="Actual"))
    fig.add_trace(go.Scatter(x=fcst["ds"], y=fcst["yhat"], name="Forecast"))
    fig.add_trace(go.Scatter(x=fcst["ds"], y=fcst["yhat_upper"], name="Upper", line=dict(dash="dot")))
    fig.add_trace(go.Scatter(x=fcst["ds"], y=fcst["yhat_lower"], name="Lower", line=dict(dash="dot")))
    fig.update_layout(title=f"Forecast for {value_col}", xaxis_title=date_col, yaxis_title=value_col)
    return fig, fcst
