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
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        st.error("❌ GOOGLE_API_KEY not found in environment variables")
        st.stop()
    genai.configure(api_key=api_key)
except Exception as e:
    st.error(f"❌ Error configuring Google AI: {str(e)}")
    st.stop()

# GPT Insights function
def gpt_insights(df: pd.DataFrame, section: str = "Overview") -> str:
    """Generate GPT insights for different sections using Gemini API"""
    try:
        model = genai.GenerativeModel('gemini-2.0-flash-exp')
        
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
        response = model.generate_content(context)
        return response.text
        
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
from core.profiler import basic_profile
from core.charts import (
    numeric_histograms,
    categorical_bars,
    correlation_heatmap,
    render_figs,
    generate_forecast_plot,
)
from core.insights import generate_insights
from core.report import build_html_report
from core.utils import fig_to_base64
from core.suggester import recommend_pairs, beginner_questions
from core.qa import answer as qa_answer

from core.exporter import build_pdf, build_ppt
from core.hypothesis import generate_hypotheses

# ---------------- App setup ----------------
load_dotenv()
st.set_page_config(page_title="InSightGenie", page_icon="📊", layout="wide")
st.title("📊 InSightGenie — AI Data Insight Assistant")
st.caption("Upload CSV/XLSX or paste a URL → automatic EDA, hypothesis testing, suggestions, Q&A and export.")

sns.set(style="whitegrid")
plt.rcParams.update({"figure.autolayout": True})

# ---------------- Helpers: compact plots + altair helpers ----------------
def small_plt_style(fig: plt.Figure):
    """Apply compact fonts to matplotlib figures for a denser UI."""
    try:
        for ax in fig.axes:
            ax.title.set_fontsize(10)
            ax.xaxis.label.set_fontsize(9)
            ax.yaxis.label.set_fontsize(9)
            for lbl in ax.get_xticklabels() + ax.get_yticklabels():
                lbl.set_fontsize(8)
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
    fig, ax = plt.subplots(figsize=(5.6, 3))
    if x.empty:
        ax.text(0.5, 0.5, f"No numeric values in {name}", ha="center", va="center")
        return fig
    sns.histplot(x, bins=20, kde=True, ax=ax)
    ax.set_title(f"{name} — Histogram & KDE")
    ax.set_xlabel(name); ax.set_ylabel("count")
    fig = small_plt_style(fig)
    return fig

def plot_box(s: pd.Series, name: str):
    x = pd.to_numeric(s, errors="coerce").dropna()
    fig, ax = plt.subplots(figsize=(5.2, 2.4))
    if x.empty:
        ax.text(0.5, 0.5, f"No numeric values in {name}", ha="center", va="center")
        return fig
    sns.boxplot(x=x, ax=ax)
    ax.set_title(f"{name} — Boxplot")
    ax.set_xlabel(name)
    fig = small_plt_style(fig)
    return fig

def plot_counts(s: pd.Series, name: str):
    c = s.astype("object").value_counts(dropna=False).head(20)
    fig, ax = plt.subplots(figsize=(5.6, 3))
    c.plot(kind="bar", ax=ax)
    ax.set_title(f"{name} — Top categories")
    ax.set_xlabel(name); ax.set_ylabel("count")
    plt.xticks(rotation=30, ha="right")
    fig.tight_layout()
    fig = small_plt_style(fig)
    return fig

def plot_scatter_with_corr(x: pd.Series, y: pd.Series, xn: str, yn: str):
    xv = pd.to_numeric(x, errors="coerce")
    yv = pd.to_numeric(y, errors="coerce")
    m = xv.notna() & yv.notna()
    fig, ax = plt.subplots(figsize=(5.6, 3))
    if m.sum() < 2:
        ax.text(0.5, 0.5, "Not enough numeric points", ha="center", va="center")
        return fig, np.nan
    sns.scatterplot(x=xv[m], y=yv[m], ax=ax, s=16)
    r = float(np.corrcoef(xv[m], yv[m])[0,1]) if m.sum() >= 2 else np.nan
    ax.set_title(f"{xn} vs {yn} (r≈{r:.2f})")
    ax.set_xlabel(xn); ax.set_ylabel(yn)
    fig = small_plt_style(fig)
    return fig, r

def plot_bar_mean_by_cat(cat: pd.Series, num: pd.Series, catn: str, numn: str):
    g = pd.DataFrame({catn: cat, numn: pd.to_numeric(num, errors="coerce")}).dropna()
    fig, ax = plt.subplots(figsize=(5.6, 3))
    if g.empty:
        ax.text(0.5, 0.5, "No valid rows after cleaning", ha="center", va="center")
        return fig
    m = g.groupby(catn)[numn].mean().sort_values(ascending=False).head(25)
    m.plot(kind="bar", ax=ax)
    ax.set_title(f"Avg {numn} by {catn}")
    ax.set_xlabel(catn); ax.set_ylabel(f"avg {numn}")
    plt.xticks(rotation=30, ha="right")
    fig.tight_layout()
    fig = small_plt_style(fig)
    return fig

def plot_stacked_counts(a: pd.Series, b: pd.Series, an: str, bn: str):
    ct = pd.crosstab(a.astype("object"), b.astype("object"))
    if ct.shape[0] > 25:
        top = ct.sum(axis=1).sort_values(ascending=False).head(25).index
        ct = ct.loc[top]
    fig, ax = plt.subplots(figsize=(6, 3.4))
    ct.plot(kind="bar", stacked=True, ax=ax, legend=False)
    ax.set_title(f"{an} × {bn} — stacked counts")
    ax.set_xlabel(an); ax.set_ylabel("count")
    plt.xticks(rotation=30, ha="right")
    fig.tight_layout()
    fig = small_plt_style(fig)
    return fig

def plot_corr_heatmap(df: pd.DataFrame, cols: Optional[List[str]] = None):
    nums = df[cols] if cols else df.select_dtypes(include="number")
    c = nums.corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(6.8, 4.4))
    sns.heatmap(c, cmap="vlag", annot=False, ax=ax)
    ax.set_title("Correlation Heatmap")
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
    st.header("Upload Data")
    file = st.file_uploader("Upload CSV or Excel", type=["csv", "xlsx", "xls"])
    st.markdown("or paste a CSV / Excel URL")
    url = st.text_input("Data URL (CSV/XLS/XLSX)")

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
elif not st.session_state.df:  # No data loaded yet
    st.info("👈 Upload a CSV/XLSX or paste a URL to begin.")
    st.stop()

# Create profile and detect types if data is loaded
if st.session_state.df is not None:
    st.session_state.profile = basic_profile(st.session_state.df)
    st.session_state.types = detect_types(st.session_state.df)

# Use these variables in the rest of the app
df = st.session_state.df
info = st.session_state.info
profile = st.session_state.profile
types = st.session_state.types

# ---------------- Tabs ----------------
tab_overview, tab_eda, tab_hypotheses, tab_suggested, tab_qa, tab_export = st.tabs(
    ["Overview", "EDA Studio", "Hypotheses", "✨ Suggested Analyses", "💬 Ask the Data", "Export"]
)

# ---------------- Overview ----------------
with tab_overview:
    st.success(f"Loaded {info['rows']:,} rows × {info['cols']} columns; memory ~ {info['memory_mb']:.2f} MB")

    # Snapshot + compact tables
    st.subheader("Dataset Snapshot")
    st.dataframe(df.head(15), use_container_width=True)

    c1, c2 = st.columns([1, 1])
    with c1:
        st.markdown("**Column Types (compact)**")
        ct_df = pd.DataFrame({"column": list(types.keys()), "type": list(types.values())})
        st.dataframe(ct_df, use_container_width=True)
    with c2:
        st.markdown("**Missingness (%)**")
        st.dataframe(profile["missing_by_col"], use_container_width=True)

    st.markdown("### Quick Visuals (compact)")
    num_figs = numeric_histograms(df, max_cols=4)
    cat_figs = categorical_bars(df, max_cols=4, top_k=8)
    if num_figs:
        render_figs(num_figs, cols=2)
    if cat_figs:
        render_figs(cat_figs, cols=2)
    corr_fig = correlation_heatmap(df)
    if corr_fig is not None:
        st.pyplot(corr_fig, use_container_width=True)

    st.markdown("### Key automatic insights (local)")
    for tip in generate_insights(df, profile, top_k=8):
        st.markdown(f"- {tip}")

    # GPT short summary (moved to bottom of overview per your request)
    st.markdown("### 📄 GPT short summary (optional)")
    try:
        # call with short flag if your function supports it; fallback to default
        summary = gpt_insights(df)
        st.markdown(summary)
    except Exception:
        st.info("GPT summary unavailable (OpenRouter misconfigured or not accessible).")

# ---------------- EDA Studio ----------------
with tab_eda:
    st.subheader("Exploratory Data Analysis (EDA) — Detailed")

    with st.expander("ℹ️ What EDA covers"):
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
    st.dataframe(info_table, use_container_width=True)

    # describe
    st.markdown("### Statistical Summary ")
    st.dataframe(df.describe(include="all").transpose(), use_container_width=True)

    # Data quality block
    st.markdown("### Data Quality — Missing • Outliers • Duplicates")
    q1, q2 = st.columns(2)
    with q1:
        st.markdown("**Missing by column**")
        st.dataframe(missing_by_col(df), use_container_width=True)
    with q2:
        st.markdown("**IQR Outlier counts (numeric)**")
        st.dataframe(iqr_outlier_counts(df), use_container_width=True)
    st.markdown(f"**Duplicate rows:** {duplicates_count(df)}")

    # Missing value preview
    with st.expander("🧰 Handling missing values — preview (non-destructive)"):
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
            st.dataframe(imp.head(20), use_container_width=True)

    # encoding & scaling recommendations
    st.markdown("### Preprocessing recommendations")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Categorical encoding**")
        enc_df = encoding_recommendations(df)
        if not enc_df.empty:
            st.dataframe(enc_df, use_container_width=True)
        else:
            st.info("No categorical columns detected.")
    with c2:
        st.markdown("**Numeric scaling**")
        scale_df = scaling_recommendations(df)
        if not scale_df.empty:
            st.dataframe(scale_df, use_container_width=True)
        else:
            st.info("No numeric columns detected.")

    # Univariate
    st.markdown("### Univariate analysis")
    ucol = st.selectbox("Choose column for univariate", df.columns.tolist(), key="univar_col")
    utype = types.get(ucol, "categorical")
    if utype == "numeric":
        st.markdown("**Numeric summary & plots**")
        st.dataframe(continuous_summary(df[ucol]).to_frame(name=ucol), use_container_width=True)
        c1, c2 = st.columns(2)
        with c1: st.pyplot(plot_hist_kde(df[ucol], ucol), use_container_width=True)
        with c2: st.pyplot(plot_box(df[ucol], ucol), use_container_width=True)
    elif utype in ("categorical","text"):
        st.markdown("**Categorical summary & counts**")
        summ, counts = categorical_summary(df[ucol])
        st.dataframe(summ.to_frame(name=ucol), use_container_width=True)
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
            sers = g.groupby(pd.Grouper(key="_d", freq="M"))[by].mean()
            st_line_from_series(sers, title=f"{by} trend over {bx}")
    else:
        st.info("Pick X and Y with compatible types.")

    # Multivariate
    st.markdown("### Multivariate analysis — correlation heatmap")
    sel = st.multiselect("Select numeric columns (up to 12)", options=df.select_dtypes(include="number").columns.tolist())
    if sel:
        st.pyplot(plot_corr_heatmap(df, sel), use_container_width=True)
    else:
        st.pyplot(plot_corr_heatmap(df), use_container_width=True)

    # GPT summary for EDA tab
    st.markdown("### 📄 GPT summary (EDA)")
    try:
        st.markdown(gpt_insights(df, section="EDA"))
    except Exception:
        st.info("GPT EDA summary unavailable.")

# ---------------- Hypotheses ----------------
# ---------------- Hypothesis Tests (detailed output) ----------------
# ---------------- Hypotheses ----------------
with tab_hypotheses:
    st.subheader("Hypothesis Testing & Validation")

    # Let user pick target variable
    target = st.selectbox("🎯 Choose your target variable", options=df.columns.tolist())
    alpha = st.slider("Significance Level (α)", 0.01, 0.10, 0.05, step=0.01)
    n = st.slider("Number of hypotheses to generate", 1, 15, 5)

    st.markdown("""
    🔍 **How this works:**  
    - Automatically formulates **Null (H₀)** and **Alternative (H₁)** hypotheses.  
    - Selects the correct statistical test (Correlation, T-test, ANOVA, Chi-square).  
    - Reports **test statistic, p-value, and decision**.  
    - Visualizes the relationship.  
    """)

    if st.button("Run Hypothesis Tests"):
        from core.hypothesis import generate_hypotheses

        hypos = generate_hypotheses(df[[target] + [c for c in df.columns if c != target]], n=n, alpha=alpha)

        if not hypos:
            st.warning("No valid hypotheses could be generated for this dataset.")
        else:
            for i, h in enumerate(hypos, 1):
                st.markdown(f"### {i}. {h['test']} — {h['x']} vs {h['y']}")
                st.write(f"**H₀:** {h['H0']}")
                st.write(f"**H₁:** {h['H1']}")
                st.write(f"**Test Used:** {h['test']}")
                st.write(f"**Statistic:** {h['statistic']:.3f}")
                st.write(f"**p-value:** {h['p_value']:.4f}")
                st.write(f"**Decision:** {h['decision']}")
                st.success(h["interpretation"])

                try:
                    smart_chart(h["x"], h["y"], df, chart_type=h["chart"])
                except Exception as e:
                    st.warning(f"Chart error: {e}")

    # GPT summary for hypotheses
    st.markdown("### 📄 GPT Summary (Hypotheses)")
    try:
        insights = gpt_insights(df, section="Hypotheses")
        if insights.startswith("GPT insights unavailable"):
            st.info(insights)
        else:
            st.markdown(insights)
    except Exception:
        st.info("GPT summary unavailable for Hypotheses tab.")



# ---------------- Suggested Analyses ----------------
# ---------------- Suggested Analyses ----------------
with tab_suggested:
    st.subheader("✨ Suggested Analyses — AI-driven recommendations")

    st.markdown("These are automatically generated suggestions based on your dataset's column types.")
    st.caption("We recommend comparisons between numeric, categorical, datetime and target columns. "
               "Each suggestion includes the rationale (‘why’) to help you interpret.")

    # Beginner plain-English starter questions
    qs = beginner_questions(df)
    if qs:
        st.markdown("**Starter Questions (natural language):**")
        for q in qs:
            st.markdown(f"- {q}")

    # Recommended analysis pairs
    st.markdown("---")
    st.markdown("### 🔍 Auto-suggested comparisons")
    suggestions = recommend_pairs(df, target_hint=None, max_pairs=12)

    if not suggestions:
        st.info("No suggested pairs could be generated for this dataset.")
    else:
        for s in suggestions:
            with st.expander(f"📊 {s['title']}  ·  ({s['chart']})"):
                st.caption(f"**Why this matters:** {s['why']}")
                try:
                    smart_chart(s["x"], s["y"], df, chart_type=s["chart"])
                except Exception as e:
                    st.error(f"Could not render chart: {e}")

    # GPT-generated extra questions (if enabled)
    st.markdown("---")
    st.markdown("### 💡 GPT-powered additional ideas")
    try:
        gpt_suggestions = gpt_insights(df, section="Suggested")
        st.markdown(gpt_suggestions)
    except Exception:
        st.info("GPT-powered suggestions unavailable.")


# ---------------- Ask the Data (Q&A) ----------------
# ---------------- Ask the Data ----------------
with tab_qa:
    st.subheader("Ask the Data — plain English + AI Explanations")

    # Suggested starter questions
    st.markdown("**Suggested questions (click to copy):**")
    for q in suggest_questions(df):
        if st.button(q, key=q):
            st.session_state["qa_q"] = q

    # Question input
    q = st.text_input("Your Question:", value=st.session_state.get("qa_q", ""))

    # Mode selection
    mode = st.radio(
        "Answer style",
        ["Simple (local)", "Gemini-powered"],
        horizontal=True
    )

    if q:
        st.markdown(f"**Your Question:** {q}**")

        if mode == "Simple (local)":
            resp = qa_answer(q, df)  # from core/qa.py
            if resp:
                try:
                    chart_type = resp.get("type")
                    if chart_type in ["histogram", "boxplot", "scatter", "bar", "line", "table", "bar_rate"]:
                        smart_chart(
                            resp.get("x", df.columns[0]),
                            resp.get("y", df.columns[1] if len(df.columns) > 1 else None),
                            df,
                            chart_type
                        )
                    st.markdown("**Technical Explanation:**")
                    st.write(resp.get("text", ""))
                    st.markdown("**Plain-English Insight:**")
                    st.info(resp.get("explain", ""))
                except Exception as e:
                    st.error(f"Local Q&A error: {e}")
            else:
                st.info("No local answer found for this question.")

        else:  # Gemini-powered
            try:
                from core.qa import ask_gemini  # make sure you add this in qa.py
                dataset_preview = df.head(10).to_dict()
                gpt_resp = ask_gemini(f"""
                You are a data analyst.
                Dataset (preview of first 10 rows): {dataset_preview}
                User question: {q}
                Answer clearly in plain English, as if explaining to a beginner.
                """)
                st.markdown("**Gemini-powered Answer:**")
                st.markdown(gpt_resp)
            except Exception as e:
                st.error(f"Gemini Q&A failed: {e}")


# ---------------- Visual Builder ----------------


# ---------------- Export ----------------
# ---------------- Export ----------------
with tab_export:
    st.subheader("📤 Export Professional Report")

    # 1. Collect structured report_data
    report_data = {
        "title": "AI Data Insight Assistant Report",
        "overview": {
            "dataset_name": file.name if file else "Uploaded Dataset",
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
    col1, col2 = st.columns(2)
    with col1:
        if st.button("📄 Generate PDF Report"):
            pdf_path = build_pdf(report_data)
            with open(pdf_path, "rb") as f:
                st.download_button("⬇ Download PDF", f, "report.pdf")
    with col2:
        if st.button("📊 Generate PPT Report"):
            ppt_path = build_ppt(report_data)
            with open(ppt_path, "rb") as f:
                st.download_button("⬇ Download PPT", f, "report.pptx")


# End of app.py
