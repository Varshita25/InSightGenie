# app.py — InSightGenie (updated)
# Full application combining EDA studio, GPT summaries, hypothesis testing, data-aware visuals,
# compact layout, export, and guided visual builder.

import os
from io import StringIO
from dotenv import load_dotenv
from typing import List, Dict, Optional
from core.qa import answer as qa_answer, suggest_questions

# Load environment variables first
load_dotenv()
print("Google API Key Exists:", bool(os.getenv("GOOGLE_API_KEY")))


import math
import numpy as np
import pandas as pd
import streamlit as st
import altair as alt
import matplotlib.pyplot as plt
import seaborn as sns
import tempfile 

import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI

# Configure Google Gemini API
try:
    api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not api_key:
        # Fallback to the new provided key
        api_key = "AIzaSyDfitBn1Nyr--00-rq_tz_VRQM6uhJk4Yg"
    genai.configure(api_key=api_key)
except Exception as e:
    st.error(f"Error configuring Google AI: {str(e)}")
    st.stop()

# GPT Insights function
@st.cache_data(show_spinner=False)
def gpt_insights(df_subset: pd.DataFrame, section: str = "Overview") -> str:
    """Generate GPT insights for different sections using Gemini API"""
    # Use a subset of data for caching efficiency if needed, but here we pass the whole df normally.
    # We rename the argument to df_subset to avoid confusion with the global df.
    df = df_subset 
    try:
        from core.gemini_helper import ask_gemini as _ask
        
        # Create context based on section
        if section == "Overview":
            context = f"""
            Dataset summary:
            - Shape: {df.shape[0]} rows × {df.shape[1]} columns
            - Columns: {', '.join(df.columns)}
            - Data types: {df.dtypes.to_dict()}
            
            Provide a brief, clear overview of this dataset in 2-3 sentences.
            """
        elif section == "EDA":
            context = f"""
            Based on the exploratory data analysis of this dataset:
            - Shape: {df.shape[0]} rows × {df.shape[1]} columns
            - Numeric columns: {', '.join(df.select_dtypes(include='number').columns)}
            - Categorical columns: {', '.join(df.select_dtypes(exclude='number').columns)}
            
            Summarize key EDA findings in 3-4 bullet points.
            """
        elif section == "Hypotheses":
            context = f"""
            Looking at the relationships between variables in this dataset:
            - Numeric variables: {', '.join(df.select_dtypes(include='number').columns)}
            - Categorical variables: {', '.join(df.select_dtypes(exclude='number').columns)}
            
            Suggest 2-3 interesting hypotheses that could be tested using statistical methods.
            Focus on relationships between variables that might yield meaningful insights.
            """
        elif section == "Suggested":
            context = f"""
            Based on the dataset structure:
            - Columns: {', '.join(df.columns)}
            - Data types: {df.dtypes.to_dict()}
            
            Suggest 2-3 interesting analyses that could reveal insights.
            """
        else:
            context = "Provide a general summary of the dataset."

        # Generate response
        return _ask(context)
        
    except Exception as e:
        return f"GPT insights unavailable: {str(e)}"

# SciPy optional
try:
    from scipy import stats as spstats
    _HAS_SCIPY = True
except Exception:
    _HAS_SCIPY = False

# ---------------- Core imports (your project modules) ----------------
from core.loader import load_table

# Wrap expensive functions with cache
@st.cache_data(show_spinner=False)
def cached_basic_profile(df: pd.DataFrame):
    from core.profiler import basic_profile
    return basic_profile(df)

@st.cache_data(show_spinner=False)
def cached_generate_insights(df: pd.DataFrame, profile: dict, top_k: int = 8):
    from core.insights import generate_insights
    return generate_insights(df, profile, top_k)

from core.charts import (
    numeric_histograms,
    categorical_bars,
    correlation_heatmap,
    render_figs,
    generate_forecast_plot,
)
from core.report import build_html_report
from core.utils import fig_to_base64
from core.suggester import recommend_pairs, beginner_questions
from core.qa import answer as qa_answer
from core.safeops import safe_dataframe

from core.exporter import build_pdf, build_ppt
from core.hypothesis import generate_hypotheses as _core_generate_hypotheses

# Cached wrapper (Renamed to v2 to force cache invalidation)
@st.cache_data(show_spinner=False)
def v2_cached_generate_hypotheses(df, n=5, alpha=0.05):
    return _core_generate_hypotheses(df, n=n, alpha=alpha)

from core.qa import suggest_questions as _core_suggest_questions
@st.cache_data(show_spinner=False)
def cached_suggest_questions(df):
    return _core_suggest_questions(df)

# ---------------- App setup ----------------
load_dotenv()
st.set_page_config(
    page_title="InSightGenie - AI Data Insight Assistant", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load custom CSS
def load_css():
    css_file = os.path.join(os.path.dirname(__file__), "static", "style.css")
    if os.path.exists(css_file):
        with open(css_file) as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

load_css()

# Simple Hero Section
st.markdown("""
<div style="text-align: center; padding: 2rem 0; margin-bottom: 1rem;">
    <h1 class="gradient-text" style="font-size: 3.5rem !important; margin-bottom: 0.5rem; line-height: 1.1;">InSightGenie</h1>
    <p style="color: #94a3b8; font-size: 1.2rem; max-width: 800px; margin: 0 auto; line-height: 1.3;">
        Turn <span style="color: #f8fafc; font-weight: 600;">Raw Data</span> into 
        <span style="color: #f8fafc; font-weight: 600;">Actionable Insights</span> – Instantly
    </p>
</div>
""", unsafe_allow_html=True)

sns.set(style="whitegrid")
plt.rcParams.update({"figure.autolayout": True})

# ---------------- Helpers: compact plots + altair helpers ----------------
def small_plt_style(fig: plt.Figure):
    """Apply compact fonts to matplotlib figures for a denser UI."""
    try:
        for ax in fig.axes:
            ax.title.set_fontsize(8)
            ax.xaxis.label.set_fontsize(7)
            ax.yaxis.label.set_fontsize(7)
            for lbl in ax.get_xticklabels() + ax.get_yticklabels():
                lbl.set_fontsize(6)
    except Exception:
        pass
    return fig

def st_bar_from_series(series: pd.Series, title: str | None = None, height: int = 260):
    if not isinstance(series, pd.Series):
        series = pd.Series(series)
    dfc = series.reset_index()
    dfc.columns = ["index", "value"]
    chart = (
        alt.Chart(dfc)
        .mark_bar()
        .encode(
            x=alt.X("index:N", sort="-y", title=dfc.columns[0]),
            y=alt.Y("value:Q", title=dfc.columns[1]),
            tooltip=["index", "value"]
        )
        .properties(width="container", height=height, title=title or "")
    )
    st.altair_chart(chart, use_container_width=True)

def st_line_from_series(series: pd.Series, title: str | None = None, height: int = 260):
    if not isinstance(series, pd.Series):
        series = pd.Series(series)
    dfc = series.reset_index()
    dfc.columns = ["index", "value"]
    # if index is datetime, use temporal encoding
    x_enc = "index:T" if np.issubdtype(dfc["index"].dtype, np.datetime64) else "index:N"
    chart = (
        alt.Chart(dfc)
        .mark_line(point=True)
        .encode(
            x=alt.X("index", type="temporal" if x_enc == "index:T" else "nominal", title=dfc.columns[0]),
            y=alt.Y("value:Q", title=dfc.columns[1]),
            tooltip=["index", "value"]
        )
        .properties(width="container", height=height, title=title or "")
    )
    st.altair_chart(chart, use_container_width=True)

# ---------------- Smart chart helpers ----------------
def _is_binary(series: pd.Series) -> bool:
    s = series.dropna().astype(str).str.lower()
    uniq = set(s.unique())
    return uniq.issubset({"0","1","yes","no","true","false","y","n"}) or len(uniq) == 2

def _to01(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.lower()
    mapping = {"yes":1,"y":1,"true":1,"1":1, "no":0,"n":0,"false":0,"0":0}
    return s.map(mapping)

def smart_chart(x: str, y: str, df: pd.DataFrame, chart_type: str = "auto"):
    """
    Render an appropriate chart for x,y based on dtypes or explicit chart_type.
    Chart types supported: histogram, boxplot, scatter, bar_rate, bar, line, table, auto.
    Uses compact sizes and friendly warnings.
    """
    try:
        # Basic input checks
        if x not in df.columns or (y not in df.columns and chart_type not in ("histogram","table","boxplot")):
            st.error("Selected columns not available in dataset.")
            return

        # explicit charts
        if chart_type == "histogram":
            vals = pd.to_numeric(df[x], errors="coerce").dropna()
            if vals.empty:
                st.warning(f"No numeric data in {x} for histogram.")
                return
            fig, ax = plt.subplots(figsize=(5.6, 3))
            sns.histplot(vals, bins=20, kde=True, ax=ax)
            ax.set_title(f"{x} — Histogram & KDE")
            fig = small_plt_style(fig)
            st.pyplot(fig, use_container_width=True)
            return

        if chart_type == "boxplot":
            vals = pd.to_numeric(df[x], errors="coerce").dropna()
            if vals.empty:
                st.warning(f"No numeric data in {x} for boxplot.")
                return
            
            # Check if y is also provided and categorical for a bivariate boxplot
            if y in df.columns and not pd.api.types.is_numeric_dtype(df[y]):
                fig, ax = plt.subplots(figsize=(5.6, 3))
                sns.boxplot(x=df[y].astype(str), y=vals, ax=ax)
                ax.set_title(f"{x} by {y} — Boxplot")
            else:
                fig, ax = plt.subplots(figsize=(5.2, 2.4))
                sns.boxplot(x=vals, ax=ax)
                ax.set_title(f"{x} — Boxplot")
            
            fig = small_plt_style(fig)
            st.pyplot(fig, use_container_width=True)
            return

        if chart_type == "scatter":
            xv = pd.to_numeric(df[x], errors="coerce")
            yv = pd.to_numeric(df[y], errors="coerce")
            m = xv.notna() & yv.notna()
            if m.sum() < 2:
                st.warning("Not enough numeric points for scatter.")
                return
            fig, ax = plt.subplots(figsize=(5.6, 3))
            sns.scatterplot(x=xv[m], y=yv[m], ax=ax, s=18)
            ax.set_title(f"{x} vs {y}")
            fig = small_plt_style(fig)
            st.pyplot(fig, use_container_width=True)
            return

        if chart_type == "bar_rate":
            if not _is_binary(df[y]):
                st.warning(f"Column {y} doesn't look binary for rate chart.")
                return
            yb = _to01(df[y])
            data = pd.DataFrame({x: df[x], "y": yb}).dropna()
            if data.empty:
                st.warning("No rows after cleaning to compute rates.")
                return
            rate = data.groupby(x)["y"].mean().sort_values(ascending=False) * 100
            st_bar_from_series(rate, title=f"{y} rate (%) by {x}", height=240)
            return

        if chart_type == "bar":
            if pd.api.types.is_numeric_dtype(df[y]):
                g = pd.DataFrame({x: df[x], y: pd.to_numeric(df[y], errors="coerce")}).dropna()
                if g.empty:
                    st.warning("No valid rows to compute group means.")
                    return
                series = g.groupby(x)[y].mean().sort_values(ascending=False).rename(f"avg_{y}")
                st_bar_from_series(series, title=f"Avg {y} by {x}")
                return
            else:
                ct = pd.crosstab(df[x], df[y])
                if ct.shape[0] > 25:
                    top_x = ct.sum(axis=1).sort_values(ascending=False).head(25).index
                    ct = ct.loc[top_x]
                fig, ax = plt.subplots(figsize=(6, 3.6))
                ct.plot(kind="bar", stacked=True, ax=ax, legend=False)
                ax.set_title(f"{x} × {y} — stacked counts")
                fig = small_plt_style(fig)
                st.pyplot(fig, use_container_width=True)
                return

        if chart_type == "line":
            d = pd.to_datetime(df[x], errors="coerce")
            yv = pd.to_numeric(df[y], errors="coerce")
            g = pd.DataFrame({"_d": d, y: yv}).dropna()
            if g.empty:
                st.warning("Need datetime X and numeric Y for a line chart.")
                return
            sers = g.groupby(pd.Grouper(key="_d", freq="M"))[y].mean()
            if sers.dropna().shape[0] < 2:
                st.warning("Not enough monthly points to plot trend.")
                return
            st_line_from_series(sers, title=f"{y} trend over {x}")
            return

        if chart_type == "table":
            st.dataframe(df.head(20))
            return

        # auto inference
        x_is_num = pd.api.types.is_numeric_dtype(df[x])
        y_is_num = pd.api.types.is_numeric_dtype(df[y])
        x_is_dt = pd.api.types.is_datetime64_any_dtype(df[x])

        # y numeric & x categorical -> grouped mean bar
        if y_is_num and (not x_is_num) and (not x_is_dt):
            g = pd.DataFrame({x: df[x], y: pd.to_numeric(df[y], errors="coerce")}).dropna()
            if g.empty:
                st.warning("No valid rows for grouped means.")
                return
            series = g.groupby(x)[y].mean().sort_values(ascending=False).rename(f"avg_{y}")
            st_bar_from_series(series, title=f"Avg {y} by {x}")
            return

        # both categorical -> stacked counts
        if (not y_is_num) and (not x_is_num):
            ct = pd.crosstab(df[x], df[y])
            if ct.shape[0] > 25:
                top_x = ct.sum(axis=1).sort_values(ascending=False).head(25).index
                ct = ct.loc[top_x]
            fig, ax = plt.subplots(figsize=(6, 3.6))
            ct.plot(kind="bar", stacked=True, ax=ax, legend=False)
            ax.set_title(f"{x} × {y} — stacked counts")
            fig = small_plt_style(fig)
            st.pyplot(fig, use_container_width=True)
            return

        # numeric - numeric -> scatter
        if x_is_num and y_is_num:
            xv = pd.to_numeric(df[x], errors="coerce")
            yv = pd.to_numeric(df[y], errors="coerce")
            mask = xv.notna() & yv.notna()
            if mask.sum() < 2:
                st.warning("Not enough numeric points to draw a scatter.")
                return
            fig, ax = plt.subplots(figsize=(5.6, 3))
            sns.scatterplot(x=xv[mask], y=yv[mask], ax=ax, s=18)
            ax.set_title(f"{x} vs {y}")
            fig = small_plt_style(fig)
            st.pyplot(fig, use_container_width=True)
            return

        # datetime x + numeric y -> monthly line
        if x_is_dt and y_is_num:
            d = pd.to_datetime(df[x], errors="coerce")
            yv = pd.to_numeric(df[y], errors="coerce")
            g = pd.DataFrame({"_d": d, y: yv}).dropna()
            if g.empty:
                st.warning("Need datetime X and numeric Y for a line chart.")
                return
            sers = g.groupby(pd.Grouper(key="_d", freq="M"))[y].mean()
            st_line_from_series(sers, title=f"{y} trend over {x}")
            return

        # numeric x & categorical y -> boxplot
        if x_is_num and (not y_is_num):
            fig, ax = plt.subplots(figsize=(5.6, 3))
            sns.boxplot(x=df[y].astype(str), y=pd.to_numeric(df[x], errors="coerce"), ax=ax)
            ax.set_xlabel(y); ax.set_ylabel(x)
            fig = small_plt_style(fig)
            st.pyplot(fig, use_container_width=True)
            return

        st.warning("Could not determine a good chart for this selection.")
    except Exception as e:
        st.error(f"Chart error: {e}")

# ---------------- EDA foundation functions (self-contained) ----------------
@st.cache_data(show_spinner=False)
def detect_types(df: pd.DataFrame) -> Dict[str, str]:
    types = {}
    n = len(df)
    for c in df.columns:
        s = df[c]
        if pd.api.types.is_datetime64_any_dtype(s) or pd.api.types.is_datetime64_dtype(s):
            types[c] = "datetime"
        elif pd.api.types.is_numeric_dtype(s):
            types[c] = "numeric"
        else:
            nunq = s.astype("object").nunique(dropna=False)
            types[c] = "categorical" if nunq <= max(50, n//20) else "text"
    return types

def missing_by_col(df: pd.DataFrame) -> pd.DataFrame:
    miss = df.isna().sum().rename("missing_count").to_frame()
    miss["missing_%"] = (miss["missing_count"] / len(df) * 100).round(2) if len(df) else 0.0
    miss["column"] = miss.index
    return miss[["column", "missing_count", "missing_%"]].sort_values("missing_%", ascending=False)

def iqr_outlier_counts(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for c in df.select_dtypes(include="number").columns:
        s = pd.to_numeric(df[c], errors="coerce").dropna()
        if s.empty:
            continue
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        iqr = q3 - q1
        lo, hi = q1 - 1.5*iqr, q3 + 1.5*iqr
        cnt = int(((s < lo) | (s > hi)).sum())
        rows.append({"column": c, "outliers_iqr": cnt, "lower": float(lo), "upper": float(hi)})
    return pd.DataFrame(rows).sort_values("outliers_iqr", ascending=False)

def duplicates_count(df: pd.DataFrame) -> int:
    return int(df.duplicated().sum())

def encoding_recommendations(df: pd.DataFrame) -> pd.DataFrame:
    recs = []
    t = detect_types(df)
    for c, kind in t.items():
        if kind in ("categorical", "text"):
            nunq = df[c].astype("object").nunique(dropna=False)
            if nunq <= 15:
                enc = "One-Hot"
            elif nunq <= 50:
                enc = "Target/Mean / Hashing"
            else:
                enc = "Text/Embedding / Hashing"
            recs.append({"column": c, "cardinality": int(nunq), "recommendation": enc})
    return pd.DataFrame(recs)

def scaling_recommendations(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    nums = df.select_dtypes(include="number").columns
    for c in nums:
        s = pd.to_numeric(df[c], errors="coerce").dropna()
        if s.empty:
            continue
        skew = float(s.skew())
        kurt = float(s.kurt())
        has_outliers = False
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        if q3 - q1 > 0:
            lo, hi = q1 - 1.5*(q3-q1), q3 + 1.5*(q3-q1)
            has_outliers = ((s < lo) | (s > hi)).sum() > 0
        if has_outliers or abs(skew) > 1:
            rec = "RobustScaler"
        elif s.min() >= 0 and s.max() <= 1:
            rec = "Already 0-1 scaled"
        elif s.min() >= 0:
            rec = "MinMaxScaler"
        else:
            rec = "StandardScaler"
        rows.append({"column": c, "skew": round(skew, 3), "kurtosis": round(kurt, 3), "recommendation": rec})
    return pd.DataFrame(rows)

def continuous_summary(s: pd.Series) -> pd.Series:
    x = pd.to_numeric(s, errors="coerce").dropna()
    if x.empty:
        return pd.Series({"count": 0})
    q = x.quantile([0.25, 0.5, 0.75])
    out = {
        "count": int(x.size),
        "mean": float(x.mean()),
        "median": float(x.median()),
        "std": float(x.std(ddof=1)) if x.size > 1 else 0.0,
        "min": float(x.min()),
        "q1": float(q.get(0.25, np.nan)),
        "q3": float(q.get(0.75, np.nan)),
        "max": float(x.max()),
        "iqr": float(q.get(0.75, np.nan) - q.get(0.25, np.nan)),
        "skewness": float(x.skew()),
        "kurtosis": float(x.kurt()),
    }
    # Normality test (optional)
    if _HAS_SCIPY and x.size >= 8:
        try:
            if x.size <= 5000:
                stat, p = spstats.shapiro(x.sample(min(5000, x.size)))
                out["normality_test"] = "Shapiro-Wilk"
                out["p_value"] = float(p)
            else:
                stat, p = spstats.normaltest(x.sample(10000, replace=True))
                out["normality_test"] = "D’Agostino K²"
                out["p_value"] = float(p)
        except Exception:
            pass
    return pd.Series(out)

def categorical_summary(s: pd.Series):
    v = s.astype("object")
    mode = v.mode(dropna=True)
    mode_val = None if mode.empty else mode.iloc[0]
    counts = v.value_counts(dropna=False).head(25)
    out = pd.Series({
        "unique": int(v.nunique(dropna=False)),
        "mode": mode_val,
        "mode_freq": int(counts.iloc[0]) if len(counts) else 0
    })
    return out, counts

# plotting helpers (compact)
def plot_hist_kde(s: pd.Series, name: str):
    x = pd.to_numeric(s, errors="coerce").dropna()
    fig, ax = plt.subplots(figsize=(4.2, 2.8))
    if x.empty:
        ax.text(0.5, 0.5, f"No numeric values in {name}", ha="center", va="center")
        return fig
    sns.histplot(x, bins=20, kde=True, ax=ax, color="#ff7e7e", alpha=0.8)
    ax.set_title(f"{name} Distribution", fontsize=9, fontweight='bold', pad=8)
    ax.set_xlabel(name, fontsize=8)
    ax.set_ylabel("Frequency", fontsize=8)
    sns.despine()
    fig.tight_layout()
    fig = small_plt_style(fig)
    return fig

def plot_box(s: pd.Series, name: str):
    x = pd.to_numeric(s, errors="coerce").dropna()
    fig, ax = plt.subplots(figsize=(4.2, 2.2))
    if x.empty:
        ax.text(0.5, 0.5, f"No numeric values in {name}", ha="center", va="center")
        return fig
    sns.boxplot(x=x, ax=ax, color="#9b59b6", width=0.5)
    ax.set_title(f"{name} Boxplot", fontsize=9, fontweight='bold', pad=8)
    ax.set_xlabel(name, fontsize=8)
    sns.despine()
    fig.tight_layout()
    fig = small_plt_style(fig)
    return fig

def plot_counts(s: pd.Series, name: str):
    # Clean up labels if they are timestamps
    if pd.api.types.is_datetime64_any_dtype(s):
        c = s.dt.date.value_counts(dropna=False).head(20)
    else:
        # Check if the index strings look like timestamps and clean them
        c = s.astype(str).value_counts(dropna=False).head(20)
        c.index = [x.split(' ')[0] if ' 00:00:00' in x else x for x in c.index]
        
    fig, ax = plt.subplots(figsize=(4.5, 3))
    sns.barplot(x=c.index, y=c.values, ax=ax, palette="rocket")
    ax.set_title(f"Top Categories: {name}", fontsize=9, fontweight='bold', pad=8)
    ax.set_xlabel("")
    ax.set_ylabel("Count", fontsize=8)
    plt.xticks(rotation=45, ha="right", fontsize=7)
    sns.despine()
    fig.tight_layout()
    fig = small_plt_style(fig)
    return fig

def plot_scatter_with_corr(x: pd.Series, y: pd.Series, xn: str, yn: str):
    xv = pd.to_numeric(x, errors="coerce")
    yv = pd.to_numeric(y, errors="coerce")
    m = xv.notna() & yv.notna()
    fig, ax = plt.subplots(figsize=(4.5, 3))
    if m.sum() < 2:
        ax.text(0.5, 0.5, "Not enough numeric points", ha="center", va="center")
        return fig, np.nan
    sns.regplot(x=xv[m], y=yv[m], ax=ax, scatter_kws={'s':20, 'alpha':0.6}, line_kws={'color':'#ff7e7e'})
    r = float(np.corrcoef(xv[m], yv[m])[0,1]) if m.sum() >= 2 else np.nan
    ax.set_title(f"{xn} vs {yn} (r≈{r:.2f})", fontsize=9, fontweight='bold', pad=8)
    ax.set_xlabel(xn, fontsize=8); ax.set_ylabel(yn, fontsize=8)
    sns.despine()
    fig.tight_layout()
    fig = small_plt_style(fig)
    return fig, r

def plot_bar_mean_by_cat(cat: pd.Series, num: pd.Series, catn: str, numn: str):
    g = pd.DataFrame({catn: cat, numn: pd.to_numeric(num, errors="coerce")}).dropna()
    fig, ax = plt.subplots(figsize=(4.5, 3))
    if g.empty:
        ax.text(0.5, 0.5, "No valid rows after cleaning", ha="center", va="center")
        return fig
    m = g.groupby(catn)[numn].mean().sort_values(ascending=False).head(20)
    sns.barplot(x=m.index, y=m.values, ax=ax, palette="flare")
    ax.set_title(f"Avg {numn} by {catn}", fontsize=9, fontweight='bold', pad=8)
    ax.set_xlabel("")
    ax.set_ylabel(f"Average {numn}", fontsize=8)
    plt.xticks(rotation=45, ha="right", fontsize=7)
    sns.despine()
    fig.tight_layout()
    fig = small_plt_style(fig)
    return fig

def plot_stacked_counts(a: pd.Series, b: pd.Series, an: str, bn: str):
    ct = pd.crosstab(a.astype(str), b.astype(str))
    if ct.shape[0] > 20:
        top = ct.sum(axis=1).sort_values(ascending=False).head(20).index
        ct = ct.loc[top]
    fig, ax = plt.subplots(figsize=(4.8, 3.2))
    ct.plot(kind="bar", stacked=True, ax=ax, cmap="rocket")
    ax.set_title(f"{an} × {bn} Proportions", fontsize=9, fontweight='bold', pad=8)
    ax.set_xlabel("")
    ax.set_ylabel("Count", fontsize=8)
    plt.xticks(rotation=45, ha="right", fontsize=7)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=7)
    sns.despine()
    fig.tight_layout()
    fig = small_plt_style(fig)
    return fig

def plot_corr_heatmap(df: pd.DataFrame, cols: Optional[List[str]] = None):
    nums = df[cols] if cols else df.select_dtypes(include="number")
    if nums.shape[1] < 2:
        return None
    c = nums.corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(4.8, 3.8))
    sns.heatmap(c, cmap="rocket_r", annot=True, fmt=".2f", ax=ax, annot_kws={"size": 7}, center=0)
    ax.set_title("Correlation Matrix", fontsize=9, fontweight='bold', pad=8)
    ax.tick_params(axis='both', which='major', labelsize=7)
    fig.tight_layout()
    fig = small_plt_style(fig)
    return fig

# Initialize session state
if 'df' not in st.session_state:
    st.session_state.df = None
if 'info' not in st.session_state:
    st.session_state.info = None
if 'profile' not in st.session_state:
    st.session_state.profile = None
if 'types' not in st.session_state:
    st.session_state.types = None

def _load_from_url(url: str):
    url = url.strip()
    if not url:
        return None, None
    try:
        if url.lower().endswith(".csv"):
            df = pd.read_csv(url)
        elif url.lower().endswith((".xlsx", ".xls")):
            df = pd.read_excel(url)
        else:
            try:
                df = pd.read_csv(url)
            except Exception:
                df = pd.read_excel(url)
        info = {
            "rows": df.shape[0],
            "cols": df.shape[1],
            "memory_mb": float(df.memory_usage(deep=True).sum() / (1024**2)),
        }
        return df, info
    except Exception as e:
        st.error(f"URL load error: {e}")
        return None, None

# ---------------- Sidebar: file / url loader ----------------
with st.sidebar:
    st.markdown("""
<div style="text-align: center; padding: 1rem 0;">
    <h2 style="font-size: 1.5rem; margin-bottom: 0.5rem;">Data Source</h2>
    <p style="color: #64748b; font-size: 0.9rem;">Upload or connect your dataset</p>
</div>
""", unsafe_allow_html=True)
    
    file = st.file_uploader(
        "Upload CSV or Excel", 
        type=["csv", "xlsx", "xls"],
        help="Drag and drop your file here or click to browse"
    )
    
    st.markdown("---")
    st.markdown("### Or use a URL")
    url = st.text_input(
        "Data URL", 
        placeholder="https://example.com/data.csv",
        help="Paste a direct link to CSV or Excel file"
    )
    
    st.markdown("---")
    if st.button("📈 Use Sample Dataset", use_container_width=True):
        # Provide a simple sample dataset
        sample_df = pd.DataFrame({
            "Date": pd.date_range("2023-01-01", periods=100),
            "Category": np.random.choice(["Tech", "Sales", "HR", "Legal"], 100),
            "Revenue": np.random.normal(5000, 1500, 100),
            "Expenses": np.random.normal(3000, 800, 100),
            "Active_Users": np.random.randint(100, 1000, 100)
        })
        st.session_state.df = sample_df
        st.session_state.info = {
            "rows": 100, "cols": 5, "memory_mb": 0.01
        }
        st.rerun()
    
    if not file and not url:
        st.markdown("---")
        st.markdown("""
<div class="glass-card" style="padding: 1.2rem; margin-top: 1rem;">
    <h4 style="margin: 0 0 0.8rem 0; color: #f8fafc; font-family: 'Outfit';">Quick Start</h4>
    <p style="font-size: 0.9rem; color: #94a3b8; margin: 0; line-height: 1.5;">
        Unlock the full potential of your data:
    </p>
    <div style="margin-top: 1rem; display: flex; flex-direction: column; gap: 0.5rem;">
        <div style="display: flex; align-items: center; gap: 0.5rem; font-size: 0.85rem; color: #cbd5e1;">
            <span style="color: #6366f1;">✦</span> Automated EDA
        </div>
        <div style="display: flex; align-items: center; gap: 0.5rem; font-size: 0.85rem; color: #cbd5e1;">
            <span style="color: #a855f7;">✦</span> AI-Powered Insights
        </div>
        <div style="display: flex; align-items: center; gap: 0.5rem; font-size: 0.85rem; color: #cbd5e1;">
            <span style="color: #ec4899;">✦</span> Statistical Engine
        </div>
        <div style="display: flex; align-items: center; gap: 0.5rem; font-size: 0.85rem; color: #cbd5e1;">
            <span style="color: #6366f1;">✦</span> Smart Export
        </div>
    </div>
</div>
""", unsafe_allow_html=True)
    
    # Show dataset info if loaded
    if st.session_state.get('df') is not None:
        st.markdown("---")
        st.markdown("### Dataset Info")
        info = st.session_state.info
        
        st.markdown(f"""
<div class="glass-card" style="padding: 1.2rem; margin-top: 0.5rem; border-left: 4px solid #10B981;">
    <div style="display: flex; justify-content: space-between; margin-bottom: 0.8rem;">
        <span style="color: #94a3b8; font-size: 0.85rem;">Rows</span>
        <span style="color: #f8fafc; font-weight: 700;">{info['rows']:,}</span>
    </div>
    <div style="display: flex; justify-content: space-between; margin-bottom: 0.8rem;">
        <span style="color: #94a3b8; font-size: 0.85rem;">Columns</span>
        <span style="color: #f8fafc; font-weight: 700;">{info['cols']}</span>
    </div>
    <div style="display: flex; justify-content: space-between;">
        <span style="color: #94a3b8; font-size: 0.85rem;">Memory</span>
        <span style="color: #f8fafc; font-weight: 700;">{info['memory_mb']:.2f} MB</span>
    </div>
</div>
""", unsafe_allow_html=True)

# Load data when a file is uploaded or URL is provided
if file:
    with st.spinner("Reading file..."):
        st.session_state.df, st.session_state.info = load_table(file)
        if st.session_state.df is None or st.session_state.info is None:
            st.error("Failed to load the file. Please check the file format and try again.")
            st.stop()
elif url:
    with st.spinner("Reading URL..."):
        st.session_state.df, st.session_state.info = _load_from_url(url)
        if st.session_state.df is None or st.session_state.info is None:
            st.error("Failed to load data from URL. Please check the URL and try again.")
            st.stop()
elif st.session_state.df is None:  # No data loaded yet
    st.markdown("""
<div style="text-align: center; padding: 1rem 2rem;">
    <h2 style="color: #f8fafc; margin-bottom: 1.5rem; font-size: 2.5rem;">Turn Raw Data into Actionable Insights – Instantly</h2>
    <p style="color: #94a3b8; font-size: 1.2rem; max-width: 700px; margin: 0 auto 2.5rem auto; line-height: 1.6;">
        Upload a CSV or Excel file, or paste a URL to get started with 
        <span class="gradient-text">AI-powered data analysis</span>.
    </p>
    <div style="margin-bottom: 3.5rem;">
        <p style="color: #64748b; font-size: 0.9rem; margin-bottom: 1rem;">Don't have a file? Use our sample data from the sidebar.</p>
    </div>
    <div style="display: flex; gap: 2rem; justify-content: center; flex-wrap: wrap;">
        <div class="glass-card" style="width: 240px; text-align: left;">
            <div style="background: var(--primary-gradient); width: 40px; height: 40px; border-radius: 10px; display: flex; align-items: center; justify-content: center; margin-bottom: 1.5rem; font-size: 1.5rem;">
                🧠
            </div>
            <h4 style="color: #f8fafc; margin-bottom: 0.8rem; font-size: 1.2rem;">AI Analysis</h4>
            <p style="color: #94a3b8; font-size: 0.95rem; margin: 0; line-height: 1.5;">
                Deep insights powered by Gemini 2.0 Flash.
            </p>
        </div>
        <div class="glass-card" style="width: 240px; text-align: left;">
            <div style="background: var(--primary-gradient); width: 40px; height: 40px; border-radius: 10px; display: flex; align-items: center; justify-content: center; margin-bottom: 1.5rem; font-size: 1.5rem;">
                📊
            </div>
            <h4 style="color: #f8fafc; margin-bottom: 0.8rem; font-size: 1.2rem;">Smart Viz</h4>
            <p style="color: #94a3b8; font-size: 0.95rem; margin: 0; line-height: 1.5;">
                Beautiful, auto-generated interactive charts.
            </p>
        </div>
        <div class="glass-card" style="width: 240px; text-align: left;">
            <div style="background: var(--primary-gradient); width: 40px; height: 40px; border-radius: 10px; display: flex; align-items: center; justify-content: center; margin-bottom: 1.5rem; font-size: 1.5rem;">
                📝
            </div>
            <h4 style="color: #f8fafc; margin-bottom: 0.8rem; font-size: 1.2rem;">Export</h4>
            <p style="color: #94a3b8; font-size: 0.95rem; margin: 0; line-height: 1.5;">
                Professional PDF & PPT reports in one click.
            </p>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)
    st.stop()

# Create profile and detect types if data is loaded
if st.session_state.df is not None:
    try:
        if st.session_state.profile is None:
            with st.spinner("Profiling dataset..."):
                st.session_state.profile = cached_basic_profile(st.session_state.df)
        if st.session_state.types is None:
            st.session_state.types = detect_types(st.session_state.df)
    except Exception as e:
        st.error(f"Error during data profiling: {e}")
        st.info("Try refreshing the page or uploading a simpler dataset.")

# Use these variables in the rest of the app
df = st.session_state.df
info = st.session_state.info
profile = st.session_state.profile
types = st.session_state.types

# Provide status indicator in sidebar
with st.sidebar:
    if df is not None:
        st.markdown(f"""
            <div style="background: #f0fdf4; 
                        border: 1px solid #10B981; 
                        padding: 0.5rem; 
                        border-radius: 0.25rem;
                        text-align: center;
                        margin-bottom: 1rem;">
                <span style="color: #10B981; font-weight: 700;">Data Ready</span>
            </div>
        """, unsafe_allow_html=True)

# ---------------- Stateful Navigation ----------------
if "active_tab" not in st.session_state:
    st.session_state["active_tab"] = "Overview"

# Custom CSS for the persistent nav bar
st.markdown("""
    <style>
    div[data-testid="stHorizontalBlock"] > div:has(button) {
        display: flex;
        justify-content: center;
    }
    </style>
""", unsafe_allow_html=True)

# Navigation Row
nav_cols = st.columns([1,1,1,1,1,1])
tabs = ["Overview", "EDA Studio", "Hypotheses", "Suggestions", "Q&A", "Export"]

for i, tab_name in enumerate(tabs):
    with nav_cols[i]:
        # Highlight active tab using primary gradient
        is_active = st.session_state["active_tab"] == tab_name
        if st.button(tab_name, key=f"nav_{tab_name}", use_container_width=True, type="primary" if is_active else "secondary"):
            st.session_state["active_tab"] = tab_name
            st.rerun()

st.markdown("---")

# Assign tabs based on selection
active_tab = st.session_state["active_tab"]

# ---------------- Overview ----------------
if active_tab == "Overview":
    try:
        # Success banner with stats
        st.markdown(f"""
            <div class="glass-card" style="border-left: 4px solid #10B981; margin-bottom: 2rem;">
                <div style="display: flex; align-items: center; gap: 1rem;">
                    <div>
                        <h3 style="margin: 0; color: #10B981; font-family: 'Outfit';">Dataset Ready</h3>
                        <p style="margin: 0.5rem 0 0 0; color: #94a3b8; font-size: 1.1rem;">
                            {info['rows']:,} rows • {info['cols']} columns • {info['memory_mb']:.2f} MB
                        </p>
                    </div>
                </div>
            </div>
        """, unsafe_allow_html=True)
        
        # Key Metrics in Cards
        st.markdown("### Dataset Overview")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Rows", f"{info['rows']:,}")
        
        with col2:
            st.metric("Total Columns", f"{info['cols']}")
        
        with col3:
            missing_pct = (profile.get('missing_total', 0) / (info['rows'] * info['cols']) * 100) if info['rows'] * info['cols'] > 0 else 0
            st.metric("Missing Data", f"{missing_pct:.1f}%")
        
        with col4:
            st.metric("Memory Usage", f"{info['memory_mb']:.1f} MB")

        st.markdown("<br>", unsafe_allow_html=True)
        
        # Snapshot + compact tables
        st.markdown("### Dataset Snapshot")
        with st.expander("View First 15 Rows", expanded=True):
            safe_dataframe(df.head(15), use_container_width=True)

        st.markdown("<br>", unsafe_allow_html=True)
        
        # Snapshot + compact tables - continued
        c1, c2 = st.columns([1, 1])
        with c1:
            st.markdown("#### Column Types")
            ct_df = pd.DataFrame({
                "Column": list(types.keys()), 
                "Type": list(types.values())
            })
            safe_dataframe(ct_df, use_container_width=True)
        with c2:
            st.markdown("#### Missing Values")
            safe_dataframe(profile["missing_by_col"].head(10), use_container_width=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("### Quick Visualizations")
        
        num_figs = numeric_histograms(df, max_cols=4)
        cat_figs = categorical_bars(df, max_cols=4, top_k=8)
        if num_figs:
            with st.expander("Numeric Distributions", expanded=True):
                render_figs(num_figs, cols=3)
        if cat_figs:
            with st.expander("Categorical Distributions", expanded=True):
                render_figs(cat_figs, cols=3)
        
        corr_fig = correlation_heatmap(df)
        if corr_fig is not None:
            with st.expander("Correlation Matrix", expanded=True):
                # Centering and reducing size
                c1, c2, c3 = st.columns([1, 2, 1])
                with c2:
                    st.pyplot(corr_fig, use_container_width=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("### Key Insights")
        
        insights = cached_generate_insights(df, profile, top_k=8)
        if insights:
            for i, tip in enumerate(insights, 1):
                st.markdown(f"""
                    <div class="glass-card" style="padding: 1rem; margin: 0.5rem 0; border-left: 4px solid #6366f1;">
                        <span style="color: #6366f1; font-weight: 700; font-family: 'Outfit';">{i}.</span>
                        <span style="color: #cbd5e1; margin-left: 0.5rem;"> {tip}</span>
                    </div>
                """, unsafe_allow_html=True)

        # GPT short summary
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("### AI-Generated Summary")
        with st.spinner("Generating AI insights..."):
            try:
                summary = gpt_insights(df)
                st.markdown(f"""
                    <div class="glass-card" style="border: 1px solid rgba(99, 102, 241, 0.3); line-height: 1.6; color: #cbd5e1;">
                        {summary}
                    </div>
                """, unsafe_allow_html=True)
            except Exception as e:
                st.info("GPT summary unavailable. Check your API configuration.")
    except Exception as e:
        st.error(f"Error in Overview tab: {e}")

# ---------------- EDA Studio ----------------
elif active_tab == "EDA Studio":
    try:
        st.subheader("Exploratory Data Analysis (EDA) — Detailed")

        with st.expander("What EDA covers"):
            st.markdown("""
            EDA steps included:
            - Univariate (distribution, central tendency, spread, outliers)
            - Bivariate (relationships: numeric-numeric, numeric-cat, cat-cat)
            - Multivariate (correlation heatmap)
            - Data quality (missingness, duplicates, outliers)
            - Preprocessing recommendations (encoding, scaling)
            - Preview missing-value imputation (non-destructive)
            """)

        # df.info() as table
        st.markdown("### Dataset Info ")

        info_table = pd.DataFrame({
            "Column": df.columns,
            "Non-Null": df.notna().sum().values,
            "Dtype": df.dtypes.astype(str).values
        })
        safe_dataframe(info_table, use_container_width=True)

        # describe
        st.markdown("### Statistical Summary ")
        safe_dataframe(df.describe(include="all").transpose(), use_container_width=True)

        # Data quality block
        st.markdown("### Data Quality — Missing • Outliers • Duplicates")
        q1, q2 = st.columns(2)
        with q1:
            st.markdown("**Missing by column**")
            safe_dataframe(missing_by_col(df), use_container_width=True)
        with q2:
            st.markdown("**IQR Outlier counts (numeric)**")
            safe_dataframe(iqr_outlier_counts(df), use_container_width=True)
        st.markdown(f"**Duplicate rows:** {duplicates_count(df)}")

        # Missing value preview
        with st.expander("Handling missing values — preview (non-destructive)"):
            st.caption("Choose imputation strategies and preview head() after applying on a copy.")
            num_method = st.selectbox("Numeric imputation (preview)", ["Do nothing", "Mean", "Median", "Mode", "Forward fill", "Backward fill"])
            cat_method = st.selectbox("Categorical imputation (preview)", ["Do nothing", "Mode", "Forward fill", "Backward fill"])
            if st.button("Preview imputation (copy)"):
                imp = df.copy()
                num_cols = imp.select_dtypes(include="number").columns
                cat_cols = imp.select_dtypes(exclude="number").columns
                if num_method != "Do nothing":
                    if num_method == "Mean":
                        imp[num_cols] = imp[num_cols].fillna(imp[num_cols].mean())
                    elif num_method == "Median":
                        imp[num_cols] = imp[num_cols].fillna(imp[num_cols].median())
                    elif num_method == "Mode":
                        imp[num_cols] = imp[num_cols].fillna(imp[num_cols].mode().iloc[0])
                    elif num_method == "Forward fill":
                        imp[num_cols] = imp[num_cols].ffill()
                    elif num_method == "Backward fill":
                        imp[num_cols] = imp[num_cols].bfill()
                if cat_method != "Do nothing":
                    if cat_method == "Mode":
                        imp[cat_cols] = imp[cat_cols].fillna(imp[cat_cols].mode().iloc[0])
                    elif cat_method == "Forward fill":
                        imp[cat_cols] = imp[cat_cols].ffill()
                    elif cat_method == "Backward fill":
                        imp[cat_cols] = imp[cat_cols].bfill()
                safe_dataframe(imp.head(20), use_container_width=True)

        # encoding & scaling recommendations
        st.markdown("### Preprocessing recommendations")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Categorical encoding**")
            enc_df = encoding_recommendations(df)
            if not enc_df.empty:
                safe_dataframe(enc_df, use_container_width=True)
            else:
                st.info("No categorical columns detected.")
        with c2:
            st.markdown("**Numeric scaling**")
            scale_df = scaling_recommendations(df)
            if not scale_df.empty:
                safe_dataframe(scale_df, use_container_width=True)
            else:
                st.info("No numeric columns detected.")

        # Univariate
        st.markdown("### Univariate analysis")
        ucol = st.selectbox("Choose column for univariate", df.columns.tolist(), key="univar_col")
        utype = types.get(ucol, "categorical")
        if utype == "numeric":
            st.markdown("**Numeric summary & plots**")
            safe_dataframe(continuous_summary(df[ucol]).to_frame(name=ucol), use_container_width=True)
            col1, col2 = st.columns(2)
            with col1: 
                st.pyplot(plot_hist_kde(df[ucol], ucol), use_container_width=True)
            with col2: 
                st.pyplot(plot_box(df[ucol], ucol), use_container_width=True)
        elif utype in ("categorical","text"):
            st.markdown("**Categorical summary & counts**")
            summ, counts = categorical_summary(df[ucol])
            safe_dataframe(summ.to_frame(name=ucol), use_container_width=True)
            # Center and constrain categorical bar chart
            c1, c2, c3 = st.columns([1, 3, 1])
            with c2:
                st.pyplot(plot_counts(df[ucol], ucol), use_container_width=True)
        elif utype == "datetime":
            st.info("Datetime column — choose a numeric column in Bivariate to plot over time.")
        else:
            st.warning("Could not classify the column.")

        # Bivariate
        st.markdown("### Bivariate analysis")
        bx = st.selectbox("X", options=df.columns.tolist(), key="bix")
        by = st.selectbox("Y", options=[c for c in df.columns if c != bx], key="biy")
        xk, yk = types.get(bx, "categorical"), types.get(by, "categorical")
        # Centering bivariate plots
        c1, c2, c3 = st.columns([0.5, 3, 0.5])
        with c2:
            if xk == "numeric" and yk == "numeric":
                fig, r = plot_scatter_with_corr(df[bx], df[by], bx, by)
                st.pyplot(fig, use_container_width=True)
                if _HAS_SCIPY:
                    try:
                        xv = pd.to_numeric(df[bx], errors="coerce"); yv = pd.to_numeric(df[by], errors="coerce")
                        m = xv.notna() & yv.notna()
                        if m.sum() >= 3:
                            _, p = spstats.pearsonr(xv[m].values, yv[m].values)
                            st.caption(f"Pearson r≈{r:.2f}, p≈{p:.3g}")
                    except Exception:
                        pass
            elif xk in ("categorical","text") and yk == "numeric":
                st.pyplot(plot_bar_mean_by_cat(df[bx], df[by], bx, by), use_container_width=True)
            elif xk == "numeric" and yk in ("categorical","text"):
                st.pyplot(plot_bar_mean_by_cat(df[by], df[bx], by, bx), use_container_width=True)
            elif xk in ("categorical","text") and yk in ("categorical","text"):
                st.pyplot(plot_stacked_counts(df[bx], df[by], bx, by), use_container_width=True)
            elif xk == "datetime" and yk == "numeric":
                d = pd.to_datetime(df[bx], errors="coerce"); yv = pd.to_numeric(df[by], errors="coerce")
                g = pd.DataFrame({"_d": d, by: yv}).dropna()
                if g.empty:
                    st.warning("No rows after cleaning for line plot.")
                else:
                    # Clean up date labels for trend plot
                    sers = g.groupby(pd.Grouper(key="_d", freq="M"))[by].mean()
                    sers.index = [d.strftime('%Y-%m') for d in sers.index]
                    st_line_from_series(sers, title=f"{by} trend over {bx}")
            else:
                st.info("Pick X and Y with compatible types.")

        # Multivariate
        st.markdown("### Multivariate analysis — correlation heatmap")
        sel = st.multiselect("Select numeric columns (up to 12)", options=df.select_dtypes(include="number").columns.tolist())
        
        # Center and constrain heatmap
        c1, c2, c3 = st.columns([1, 2.5, 1])
        with c2:
            if sel:
                fig = plot_corr_heatmap(df, sel)
                if fig:
                    st.pyplot(fig, use_container_width=True)
                else:
                    st.info("Select at least 2 numeric columns for a heatmap.")
            else:
                fig = plot_corr_heatmap(df)
                if fig:
                    st.pyplot(fig, use_container_width=True)
                else:
                    st.info("No numeric columns found for correlation analysis.")

        # GPT summary for EDA tab
        st.markdown("### GPT summary (EDA)")
        try:
            st.markdown(gpt_insights(df, section="EDA"))
        except Exception:
            st.info("GPT EDA summary unavailable.")
    except Exception as e:
        st.error(f"Error in EDA Studio: {e}")

# ---------------- Hypotheses ----------------
elif active_tab == "Hypotheses":
    try:
        st.markdown("""
            <div style="text-align: center; margin-bottom: 2rem;">
                <h2 style="font-size: 2rem; color: #2563eb; margin-bottom: 0.5rem;">
                    Hypothesis Testing & Validation
                </h2>
                <p style="color: #64748b; font-size: 1.1rem;">
                    Statistical testing with automated hypothesis generation
                </p>
            </div>
        """, unsafe_allow_html=True)

        # Configuration section
        st.markdown("### Test Configuration")
        with st.form("hypo_config"):
            col1, col2, col3 = st.columns(3)
            with col1:
                target = st.selectbox("Target Variable", options=df.columns.tolist(), help="Primary variable for comparisons")
            with col2:
                alpha = st.select_slider("Significance Level (α)", options=[0.01, 0.05, 0.10], value=0.05, help="Risk level for Type I error")
            with col3:
                n = st.slider("Max Discovery Count", 1, 15, 5, help="Number of relationships to analyze")
            
            submit = st.form_submit_button("🚀 Start Statistical Engine", use_container_width=True)

        if submit:
            with st.spinner("Analyzing patterns and formulating hypotheses..."):
                # Pass only relevant columns to speed up filtering
                subset = df[[target] + [c for c in df.columns if c != target]]
                st.session_state["hypo_results"] = v2_cached_generate_hypotheses(subset, n=n, alpha=alpha)

        # Info card
        st.markdown("""
            <div class="glass-card" style="border-left: 4px solid #06B6D4; margin: 1.5rem 0;">
                <h4 style="color: #06B6D4; margin: 0 0 1rem 0; font-family: 'Outfit';">Advanced Engine</h4>
                <ul style="color: #94a3b8; margin: 0; padding-left: 1.5rem; line-height: 1.6;">
                    <li>Forms <strong>Null (H₀)</strong> and <strong>Alternative (H₁)</strong> hypotheses automatically</li>
                    <li>Selects optimal statistical tests (Correlation, T-test, ANOVA, Chi-square)</li>
                    <li>Calculates <strong>test statistics, p-values, and confidence levels</strong></li>
                    <li>Generates deep visual proof for every discovery</li>
                </ul>
            </div>
        """, unsafe_allow_html=True)


        if "hypo_results" not in st.session_state:
            st.markdown("""
                <div class="glass-card" style="text-align: center; padding: 3rem; margin: 2rem 0; border: 1px dashed rgba(255,255,255,0.2);">
                    <div style="font-size: 3rem; margin-bottom: 1.5rem;">🧪</div>
                    <h3 style="color: #f8fafc; margin-bottom: 1rem; font-family: 'Outfit';">Ready for Discovery?</h3>
                    <p style="color: #94a3b8; max-width: 500px; margin: 0 auto 2rem auto; line-height: 1.6;">
                        Our statistical engine will automatically formulate hypotheses, run appropriate tests, and provide deep interpretations of your data relationships.
                    </p>
                    <p style="color: #6366f1; font-weight: 600; font-size: 0.9rem; text-transform: uppercase; letter-spacing: 0.1em;">
                        Click the button above to begin
                    </p>
                </div>
            """, unsafe_allow_html=True)

        if "hypo_results" in st.session_state:
            hypos = st.session_state["hypo_results"]
            if not hypos:
                st.warning("No valid hypotheses could be generated for this dataset.")
            else:
                st.markdown(f"""
                    <div class="glass-card" style="padding: 1rem; margin: 1.5rem 0; border: 1px solid #10B981;">
                        <strong style="color: #10B981;">Discovery Pipeline:</strong> 
                        <span style="color: #cbd5e1;">Generated {len(hypos)} validated hypothesis test(s)</span>
                    </div>
                """, unsafe_allow_html=True)
                
                for i, h in enumerate(hypos, 1):
                    # Safety check for old cache structure
                    if 'steps' not in h:
                        st.info("🔄 Update detected. Please click 'Start Statistical Engine' again to refresh your results.")
                        del st.session_state["hypo_results"]
                        st.stop()
                        
                    is_significant = "Reject" in h['steps']['Step 7: Conclusion']
                    decision_color = "#10B981" if is_significant else "#F59E0B"
                    
                    # 1. Header and Rationale
                    st.markdown(f"""
<div class="glass-card" style="margin: 2rem 0; border: 1px solid rgba(255,255,255,0.1); padding: 1.5rem;">
    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 1.2rem;">
        <h3 style="color: #f8fafc; margin: 0; font-family: 'Outfit'; font-size: 1.4rem;">
            {i}. {h['test_name']}
        </h3>
        <div style="background: {decision_color}; color: #020617; padding: 0.2rem 0.8rem; border-radius: 20px; font-weight: 700; font-size: 0.75rem;">
            { 'SIGNIFICANT' if is_significant else 'NOT SIGNIFICANT' }
        </div>
    </div>
    <div style="margin-bottom: 1.5rem; background: rgba(255,255,255,0.03); padding: 1rem; border-radius: 0.5rem; border-left: 3px solid #6366f1;">
        <p style="color: #cbd5e1; font-size: 0.9rem; margin: 0; line-height: 1.5;">
            <strong>Scientific Rationale:</strong> {h['rationale']}
        </p>
    </div>
    <div style="display: flex; justify-content: center; margin-bottom: 1.5rem;">
        <div style="width: 100%; max-width: 450px; background: rgba(255,255,255,0.02); padding: 1rem; border-radius: 0.8rem; border: 1px solid rgba(255,255,255,0.05);">
""", unsafe_allow_html=True)

                    # 2. Centered Visualization
                    if h['chart'] == "scatter":
                        st.pyplot(plot_scatter_with_corr(df[h['x']], df[h['y']], h['x'], h['y'])[0], use_container_width=True)
                    elif h['chart'] == "boxplot":
                        st.pyplot(plot_box(df[h['y']], h['y']), use_container_width=True)
                    elif h['chart'] == "bar":
                        st.pyplot(plot_counts(df[h['x']], h['x']), use_container_width=True)

                    # 3. Final Conclusion
                    st.markdown(f"""
        </div>
    </div>
    <div style="border-top: 1px solid rgba(255,255,255,0.05); padding-top: 1rem; text-align: center;">
        <p style="color: #94a3b8; font-size: 0.9rem; margin: 0; line-height: 1.5;">
            <strong>Result:</strong> {h['steps']['Step 7: Conclusion']}
        </p>
    </div>
</div>
""", unsafe_allow_html=True)

        # GPT summary
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("### AI-Generated Hypothesis Suggestions")
        with st.spinner("Generating AI suggestions..."):
            try:
                insights = gpt_insights(df, section="Hypotheses")
                if insights.startswith("GPT insights unavailable"):
                    st.info(insights)
                else:
                    st.markdown(f"""
                        <div class="glass-card" style="border: 1px solid rgba(99, 102, 241, 0.3); line-height: 1.6; color: #cbd5e1;">
                            {insights}
                        </div>
                    """, unsafe_allow_html=True)
            except Exception:
                st.info("GPT summary unavailable for Hypotheses tab.")
    except Exception as e:
        st.error(f"Error in Hypotheses tab: {e}")



# ---------------- Suggested Analyses ----------------
elif active_tab == "Suggestions":
    try:
        st.markdown("""
            <div style="text-align: center; margin-bottom: 2rem; padding: 1rem 0;">
                <h2 class="gradient-text" style="font-size: 2.5rem !important; margin-bottom: 0.5rem;">
                    Suggested Analyses
                </h2>
                <p style="color: #94a3b8; font-size: 1.1rem;">
                    AI-driven recommendations tailored to your dataset
                </p>
            </div>
        """, unsafe_allow_html=True)

        # Beginner plain-English starter questions
        qs = beginner_questions(df)
        if qs:
            st.markdown("### Starter Questions")
            st.markdown("""
                <div class="glass-card" style="margin-bottom: 2rem;">
                    <p style="color: #94a3b8; margin-bottom: 1.5rem;">Natural language explorations for your dataset:</p>
            """, unsafe_allow_html=True)
            for i, q in enumerate(qs, 1):
                if st.button(q, key=f"q_{i}"):
                    st.session_state["qa_q"] = q
                    st.session_state["active_tab"] = "Q&A"
                    st.rerun()
            st.markdown("</div>", unsafe_allow_html=True)

        # Recommended analysis pairs
        st.markdown("### Recommended Comparisons")
        st.markdown("""
            <p style="color: #64748b; margin-bottom: 1.5rem;">
                These suggestions are automatically generated based on your dataset's column types and relationships.
                Each includes a rationale to help you interpret the results.
            </p>
        """, unsafe_allow_html=True)
        
        suggestions = recommend_pairs(df, target_hint=None, max_pairs=12)

        if not suggestions:
            st.info("No suggested pairs could be generated for this dataset.")
        else:
            for idx, s in enumerate(suggestions, 1):
                with st.expander(f"{idx}. {s['title']} - {s['chart'].upper()}", expanded=False):
                    st.markdown(f"""
                        <div class="glass-card" style="border-left: 4px solid #06B6D4; padding: 1.2rem; margin-bottom: 1.5rem;">
                            <div style="color: #06B6D4; font-size: 0.75rem; font-weight: 700; margin-bottom: 0.5rem; text-transform: uppercase;">Analysis Rationale</div>
                            <div style="color: #cbd5e1; line-height: 1.5;">{s['why']}</div>
                        </div>
                    """, unsafe_allow_html=True)
                    try:
                        smart_chart(s["x"], s["y"], df, chart_type=s["chart"])
                    except Exception as e:
                        st.error(f"Could not render chart: {e}")

        # GPT-generated extra questions
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("### AI-Powered Additional Ideas")
        with st.spinner("Generating AI suggestions..."):
            try:
                gpt_suggestions = gpt_insights(df, section="Suggested")
                st.markdown(f"""
                    <div style="background: #eff6ff;
                                border: 1px solid #bfdbfe;
                                border-radius: 0.5rem;
                                padding: 1.5rem;">
                        {gpt_suggestions}
                    </div>
                """, unsafe_allow_html=True)
            except Exception:
                st.info("GPT-powered suggestions unavailable.")
    except Exception as e:
        st.error(f"Error in Suggestions tab: {e}")


# ---------------- Ask the Data (Q&A) ----------------
elif active_tab == "Q&A":
    try:
        st.markdown("""
            <div style="text-align: center; margin-bottom: 2rem;">
                <h2 style="font-size: 2rem; color: #10b981; margin-bottom: 0.5rem;">
                    Ask the Data
                </h2>
                <p style="color: #64748b; font-size: 1.1rem;">
                    Get answers to your questions in plain English
                </p>
            </div>
        """, unsafe_allow_html=True)

        # Suggested starter questions
        st.markdown("### Suggested Questions")
        suggested_qs = cached_suggest_questions(df)
        cols = st.columns(2)
        for idx, q in enumerate(suggested_qs):
            with cols[idx % 2]:
                if st.button(q, key=f"sug_{idx}", use_container_width=True):
                    st.session_state["qa_q"] = q

        st.markdown("<br>", unsafe_allow_html=True)
        
        # Combined Input and Mode in a Form to prevent lag
        with st.form("qa_form"):
            st.markdown("### Ask your Question")
            q_input = st.text_input(
                "Enter your question:", 
                value=st.session_state.get("qa_q", ""),
                placeholder="e.g., What is the average value of price?",
                label_visibility="collapsed"
            )
            
            col_m1, col_m2, col_m3 = st.columns([2, 1, 1])
            with col_m1:
                mode = st.radio("Answer Mode", ["Fast (Local)", "Advanced (Gemini AI)"], horizontal=True)
            with col_m2:
                submit_qa = st.form_submit_button("🚀 Get Answer", use_container_width=True)
            with col_m3:
                clear_qa = st.form_submit_button("🗑️ Clear", use_container_width=True)

        if clear_qa:
            st.session_state["qa_q"] = ""
            if "qa_resp" in st.session_state: del st.session_state["qa_resp"]
            if "last_q" in st.session_state: del st.session_state["last_q"]

        if submit_qa or "qa_resp" in st.session_state:
            active_q = q_input if submit_qa else st.session_state.get("last_q", q_input)
            
            if active_q:
                st.markdown(f"""
                    <div class="glass-card" style="border-left: 4px solid #10b981; margin: 1rem 0; padding: 1rem;">
                        <div style="color: #10b981; font-size: 0.7rem; font-weight: 700; margin-bottom: 0.3rem; text-transform: uppercase;">Active Inquiry</div>
                        <div style="color: #f8fafc; font-size: 1.1rem; font-weight: 500;">{active_q}</div>
                    </div>
                """, unsafe_allow_html=True)

                if submit_qa:
                    with st.spinner("Processing..."):
                        if "Fast" in mode:
                            st.session_state["qa_resp"] = qa_answer(active_q, df)
                            st.session_state["qa_mode"] = "local"
                        else:
                            try:
                                dataset_preview = df.head(10).to_dict()
                                st.session_state["qa_resp"] = ask_gemini(f"Analyst: {active_q}\nData: {dataset_preview}")
                                st.session_state["qa_mode"] = "gemini"
                            except Exception as e:
                                st.error(f"Gemini failed: {e}")
                        st.session_state["last_q"] = active_q

                resp = st.session_state.get("qa_resp")
                if resp:
                    try:
                        if st.session_state.get("qa_mode") == "local":
                            st.markdown("### Visualization")
                            chart_type = resp.get("type")
                            if chart_type in ["histogram", "boxplot", "scatter", "bar", "line", "table", "bar_rate"]:
                                smart_chart(
                                    resp.get("x", df.columns[0]),
                                    resp.get("y", df.columns[1] if len(df.columns) > 1 else None),
                                    df,
                                    chart_type
                                )
                            
                            st.markdown("<br>", unsafe_allow_html=True)
                            st.markdown("### Answer")
                            
                            # Technical explanation
                            st.markdown(f"""
                                <div class="glass-card" style="margin-bottom: 1.5rem; border-left: 4px solid #6366F1;">
                                    <div style="color: #6366F1; font-size: 0.75rem; font-weight: 700; margin-bottom: 0.75rem; text-transform: uppercase;">
                                        Technical Logic
                                    </div>
                                    <div style="color: #cbd5e1; line-height: 1.5;">
                                        {resp.get("text", "")}
                                    </div>
                                </div>
                            """, unsafe_allow_html=True)
                            
                            # Plain English insight
                            st.markdown(f"""
                                <div class="glass-card" style="border-left: 4px solid #10B981; border: 1px solid rgba(16, 185, 129, 0.2);">
                                    <div style="color: #10B981; font-size: 0.75rem; font-weight: 700; margin-bottom: 0.75rem; text-transform: uppercase;">
                                        Human Insight
                                    </div>
                                    <div style="color: #f8fafc; font-size: 1.1rem; line-height: 1.5; font-family: 'Outfit';">
                                        {resp.get("explain", "")}
                                    </div>
                                </div>
                            """, unsafe_allow_html=True)
                        else:
                            st.markdown("### Gemini AI Answer")
                            st.markdown(f"""
                                <div class="glass-card" style="border: 1px solid rgba(99, 102, 241, 0.3); line-height: 1.6; color: #cbd5e1;">
                                    {resp}
                                </div>
                            """, unsafe_allow_html=True)
                    except Exception as e:
                        st.error(f"Display Error: {e}")
                else:
                    st.info("Please enter a question and click 'Get Answer'.")

            else:  # Gemini-powered
                if st.button("Ask Gemini", use_container_width=True) or "gemini_resp" in st.session_state:
                    if st.session_state.get("last_gemini_q") != q:
                        with st.spinner("Asking Gemini AI..."):
                            try:
                                dataset_preview = df.head(10).to_dict()
                                st.session_state["gemini_resp"] = ask_gemini(f"""
                                You are a data analyst.
                                Dataset (preview of first 10 rows): {dataset_preview}
                                User question: {q}
                                Answer clearly in plain English, as if explaining to a beginner.
                                """)
                                st.session_state["last_gemini_q"] = q
                            except Exception as e:
                                st.error(f"Gemini Q&A failed: {e}")
                    
                    if "gemini_resp" in st.session_state:
                        st.markdown("### Gemini AI Answer")
                        st.markdown(f"""
                            <div class="glass-card" style="border: 1px solid rgba(99, 102, 241, 0.3); line-height: 1.6; color: #cbd5e1;">
                                {st.session_state["gemini_resp"]}
                            </div>
                        """, unsafe_allow_html=True)
    except Exception as e:
        st.error(f"Error in Q&A tab: {e}")


# ---------------- Visual Builder ----------------


# ---------------- Export ----------------
elif active_tab == "Export":
    try:
        st.markdown("""
            <div style="text-align: center; margin-bottom: 2rem;">
                <h2 style="font-size: 2rem; color: #f59e0b; margin-bottom: 0.5rem;">
                    Export Professional Report
                </h2>
                <p style="color: #64748b; font-size: 1.1rem;">
                    Generate publication-ready reports in PDF or PowerPoint format
                </p>
            </div>
        """, unsafe_allow_html=True)

        # Info card about what's included
        st.markdown("""
            <div class="glass-card" style="border-left: 4px solid #F59E0B; margin-bottom: 2rem;">
                <h4 style="color: #F59E0B; margin: 0 0 1.5rem 0; font-family: 'Outfit';">Report Compilation Strategy</h4>
                <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 1.5rem; color: #94a3b8;">
                    <div>
                        <div style="margin-bottom: 0.8rem; display: flex; align-items: center; gap: 0.5rem;"><span style="color: #F59E0B;">✓</span> Overview & Stats</div>
                        <div style="margin-bottom: 0.8rem; display: flex; align-items: center; gap: 0.5rem;"><span style="color: #F59E0B;">✓</span> Full EDA Suite</div>
                        <div style="margin-bottom: 0.8rem; display: flex; align-items: center; gap: 0.5rem;"><span style="color: #F59E0B;">✓</span> Data Quality Audit</div>
                    </div>
                    <div>
                        <div style="margin-bottom: 0.8rem; display: flex; align-items: center; gap: 0.5rem;"><span style="color: #F59E0B;">✓</span> Hypothesis Validation</div>
                        <div style="margin-bottom: 0.8rem; display: flex; align-items: center; gap: 0.5rem;"><span style="color: #F59E0B;">✓</span> AI Suggestions</div>
                        <div style="margin-bottom: 0.8rem; display: flex; align-items: center; gap: 0.5rem;"><span style="color: #F59E0B;">✓</span> Visual Proofs</div>
                    </div>
                </div>
            </div>
        """, unsafe_allow_html=True)

        # 1. Collect structured report_data
        report_data = {
            "title": "AI Data Insight Assistant Report",
            "overview": {
                "dataset_name": file.name if (file is not None and hasattr(file, 'name')) else "Uploaded Dataset",
                "rows": df.shape[0],
                "cols": df.shape[1],
                "memory": f"{df.memory_usage(deep=True).sum() / (1024**2):.2f} MB",
                "summary": f"The dataset has {df.shape[0]:,} rows × {df.shape[1]} columns. "
                           f"Detected {int(profile.get('missing_total', 0)):,} missing cells "
                           f"and {int(profile.get('duplicates', 0)):,} duplicate rows."
            },
            "eda": {
                "univariate": [f"{c}: mean={df[c].mean():.2f}, std={df[c].std():.2f}" 
                               for c in df.select_dtypes(include='number').columns[:3]],
                "bivariate": ["Example: Numeric vs Numeric relationships visualized via scatter plots.",
                              "Example: Numeric vs Categorical visualized via bar/box plots."],
                "multivariate": ["Correlation heatmap computed across numeric features."]
            },
            "hypotheses": [],
            "suggestions": [],
            "qa": [],
            "figures": []
        }

        # Add Hypotheses (if generated before)
        try:
            from core.hypothesis import generate_hypotheses
            hypos = generate_hypotheses(df, n=3)
            for h in hypos:
                report_data["hypotheses"].append({
                    "title": h.get("title", f"{h['x']} vs {h['y']}"),
                    "test": h.get("test", ""),
                    "result": h.get("decision", ""),
                    "interpretation": h.get("interpretation", "")
                })
        except Exception:
            pass

        # Add Suggested analyses
        try:
            from core.suggester import recommend_pairs
            sug = recommend_pairs(df, max_pairs=5)
            report_data["suggestions"] = [s["title"] + " → " + s["why"] for s in sug]
        except Exception:
            pass

        # Add sample Q&A
        try:
            from core.qa import suggest_questions
            qs = suggest_questions(df, max_q=3)
            report_data["qa"] = [{"q": q, "a": "Generated answer will appear here."} for q in qs]
        except Exception:
            pass

        # Add Figures (save a few plots to temporary PNGs)
        import tempfile
        import matplotlib.pyplot as plt
        figs = numeric_histograms(df, max_cols=2) + categorical_bars(df, max_cols=2, top_k=5)
        for f in figs:
            tmp_img = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
            f.savefig(tmp_img.name, dpi=120, bbox_inches="tight")
            report_data["figures"].append(tmp_img.name)
            plt.close(f)

        # 2. Export Buttons
        st.markdown("### Download Options")
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
                <div class="glass-card" style="text-align: center; margin-bottom: 1rem; border: 1px solid rgba(239, 68, 68, 0.2);">
                    <div style="font-size: 2rem; margin-bottom: 1rem;">📄</div>
                    <h4 style="color: #f8fafc; margin: 0 0 0.5rem 0; font-family: 'Outfit';">PDF Intelligence</h4>
                    <p style="color: #94a3b8; font-size: 0.9rem; margin: 0 0 1.5rem 0; line-height: 1.5;">
                        High-fidelity document for stakeholders and decision-makers.
                    </p>
                </div>
            """, unsafe_allow_html=True)
            if st.button("Generate PDF Report", use_container_width=True):
                with st.spinner("Generating PDF..."):
                    # Assuming build_pdf exists in core.exporter or similar
                    try:
                        from core.exporter import build_pdf
                        pdf_path = build_pdf(report_data)
                        with open(pdf_path, "rb") as f:
                            st.download_button("Download PDF", f, "report.pdf", use_container_width=True)
                    except ImportError:
                        st.error("PDF Export module not found.")
        
        with col2:
            st.markdown("""
                <div style="background: #fffbeb;
                            border: 1px solid #fde68a;
                            border-radius: 0.5rem;
                            padding: 1.5rem;
                            text-align: center;
                            margin-bottom: 1rem;">
                    <h4 style="color: #1e293b; margin: 0 0 0.5rem 0;">PowerPoint Presentation</h4>
                    <p style="color: #64748b; font-size: 0.9rem; margin: 0 0 1rem 0;">
                        Slide deck format, ideal for presentations and meetings
                    </p>
                </div>
            """, unsafe_allow_html=True)
            if st.button("Generate PPT Report", use_container_width=True):
                with st.spinner("Generating PowerPoint..."):
                    try:
                        from core.exporter import build_ppt
                        ppt_path = build_ppt(report_data)
                        with open(ppt_path, "rb") as f:
                            st.download_button("Download PPT", f, "report.pptx", use_container_width=True)
                    except ImportError:
                        st.error("PowerPoint Export module not found.")
    except Exception as e:
        st.error(f"Error in Export tab: {e}")


# End of app.py
