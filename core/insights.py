# core/insights.py
import pandas as pd
import numpy as np
from typing import List, Dict


def detect_kpis(df: pd.DataFrame) -> pd.DataFrame:
    """
    Detect KPI-like numeric columns and return a summary table (Total, Avg, Min, Max).
    """
    kpi_keywords = ["revenue", "sales", "income", "profit", "amount", "cost", "price", "gmv", "margin"]
    kpi_cols = [c for c in df.columns if any(k in c.lower() for k in kpi_keywords) 
                and pd.api.types.is_numeric_dtype(df[c])]
    rows = []
    for col in kpi_cols:
        series = pd.to_numeric(df[col], errors="coerce").dropna()
        if series.empty:
            continue
        rows.append({
            "KPI": col,
            "Total": float(series.sum()),
            "Average": float(series.mean()),
            "Min": float(series.min()),
            "Max": float(series.max()),
        })
    return pd.DataFrame(rows)


def _strong_corrs(corr: pd.DataFrame, k: int = 5):
    """
    Extract top k strong correlations.
    """
    pairs = []
    if corr is None or corr.empty:
        return pairs
    c = corr.copy()
    for i in range(min(c.shape)):
        c.iat[i, i] = 0.0
    cm = c.abs().unstack().sort_values(ascending=False)
    seen = set()
    for (a, b), v in cm.items():
        key = tuple(sorted((a, b)))
        if key in seen:
            continue
        seen.add(key)
        pairs.append((a, b, float(v)))
        if len(pairs) >= k:
            break
    return pairs


def generate_insights(df: pd.DataFrame, profile: Dict, top_k: int = 15) -> List[str]:
    """
    Generate plain-English insights: KPIs, missingness, duplicates, outliers, correlations.
    """
    tips: List[str] = []
    n_rows = profile["shape"][0]

    # KPI detection
    kpi_table = detect_kpis(df)
    for _, row in kpi_table.iterrows():
        tips.append(
            f"KPI **{row['KPI']}** → Total={row['Total']:.2f}, Avg={row['Average']:.2f}, "
            f"Min={row['Min']:.2f}, Max={row['Max']:.2f}"
        )

    # Missingness
    miss = profile.get("missing_by_col", pd.DataFrame())
    if not miss.empty:
        high = miss[miss["missing_%"] >= 20].sort_values("missing_%", ascending=False)
        for _, r in high.iterrows():
            tips.append(f"⚠ Column **{r['column']}** has **{r['missing_%']}%** missing values.")

    # Duplicates
    if profile.get("duplicates", 0) > 0:
        tips.append(f"⚠ Found **{profile['duplicates']}** duplicate rows.")

    # Outliers
    num_stats = profile.get("numeric_stats", pd.DataFrame())
    if not num_stats.empty:
        q1 = df[num_stats.index].quantile(0.25)
        q3 = df[num_stats.index].quantile(0.75)
        iqr = q3 - q1
        for col in num_stats.index:
            if iqr[col] > 0:
                outliers = ((df[col] < (q1[col] - 1.5 * iqr[col])) |
                            (df[col] > (q3[col] + 1.5 * iqr[col]))).sum()
                if outliers > 0:
                    tips.append(f"📊 Column **{col}** has ~{outliers} outliers (IQR method).")

    # Strong correlations
    corr = profile.get("corr")
    if corr is not None and not corr.empty:
        pairs = _strong_corrs(corr, k=5)
        for a, b, v in pairs:
            if v >= 0.7:
                tips.append(f"📈 Strong correlation **{a} ↔ {b}** (|r|≈{v:.2f})")

    # Dataset size note
    if n_rows > 200_000:
        tips.append("ℹ Large dataset → consider sampling or chunking for faster analysis.")

    return tips[:top_k]
