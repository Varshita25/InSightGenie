# core/eda_plus.py
from __future__ import annotations
import math
from typing import Dict, List, Optional
import numpy as np
import pandas as pd

# Optional: statistical tests (graceful fallback if not available)
try:
    from scipy import stats
    _HAS_SCIPY = True
except Exception:
    _HAS_SCIPY = False


# ---------- helpers ----------

def _is_binary(series: pd.Series, dropna=True) -> bool:
    s = series.dropna() if dropna else series
    return s.nunique() == 2


def _guess_binary_target(df: pd.DataFrame) -> Optional[str]:
    lower = [c.lower() for c in df.columns]
    keywords = ["outcome", "survived", "target", "label", "churn", "default",
                "is_", "has_", "success", "converted"]
    for kw in keywords:
        for i, c in enumerate(df.columns):
            if kw in lower[i] and _is_binary(df[c]):
                return c
    # fallback: any binary bool/numeric
    for c in df.columns:
        if pd.api.types.is_bool_dtype(df[c]) or (_is_binary(df[c]) and pd.api.types.is_numeric_dtype(df[c])):
            return c
    return None


def _cohens_d(x: np.ndarray, y: np.ndarray) -> float:
    nx, ny = len(x), len(y)
    if nx < 2 or ny < 2:
        return np.nan
    s_pooled = math.sqrt(((nx-1)*np.nanvar(x, ddof=1) + (ny-1)*np.nanvar(y, ddof=1)) / (nx+ny-2))
    if s_pooled == 0:
        return 0.0
    return (np.nanmean(x) - np.nanmean(y)) / s_pooled


def _sig_star(p: Optional[float]) -> str:
    if p is None or np.isnan(p):
        return ""
    return "★★★" if p < 0.001 else "★★" if p < 0.01 else "★" if p < 0.05 else ""


# ---------- main API ----------

def advanced_eda_summary(df: pd.DataFrame, max_cat_levels: int = 30) -> Dict[str, object]:
    """
    Returns dict with:
      - chi_square: DataFrame (binary target vs categorical features)
      - t_tests: DataFrame (binary target vs numeric features)
      - anova: DataFrame (numeric KPI vs categorical features)
      - correlations: DataFrame (numeric vs numeric)
      - business_bullets: List[str] (plain-English insights)
    """
    out: Dict[str, object] = {
        "chi_square": pd.DataFrame(),
        "t_tests": pd.DataFrame(),
        "anova": pd.DataFrame(),
        "correlations": pd.DataFrame(),
        "business_bullets": []
    }

    if df.empty or df.shape[1] < 2:
        return out

    # ----------------- Type detection -----------------
    num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    cat_cols = [c for c in df.columns if (not pd.api.types.is_numeric_dtype(df[c])) 
                and (not pd.api.types.is_datetime64_any_dtype(df[c]))]

    # ----------------- Binary target analysis -----------------
    target = _guess_binary_target(df)
    chi_rows = []
    ttest_rows = []
    if target is not None:
        s_target = pd.to_numeric(df[target], errors="coerce")

        # a) categorical features → chi-square
        for c in cat_cols:
            if c == target:
                continue
            levels = df[c].astype("category")
            if levels.nunique(dropna=False) < 2 or levels.nunique(dropna=False) > max_cat_levels:
                continue
            tab = pd.crosstab(levels, s_target)
            overall = float(s_target.mean(skipna=True))
            for lvl, row in tab.iterrows():
                pos = float(row.get(1, 0.0))
                denom = float(row.sum())
                rate = (pos/denom) if denom else np.nan
                lift = (rate - overall) if pd.notna(rate) else np.nan
                chi_rows.append({
                    "feature": c,
                    "level": str(lvl),
                    "count": int(denom),
                    "rate_%": round(rate*100, 2) if pd.notna(rate) else np.nan,
                    "lift_pts": round(lift*100, 2) if pd.notna(lift) else np.nan,
                    "p_value": None,
                    "sig": ""
                })
            if _HAS_SCIPY and tab.shape[0] > 1 and tab.shape[1] == 2:
                try:
                    _, p, _, _ = stats.chi2_contingency(tab.values)
                    for r in chi_rows[-tab.shape[0]:]:
                        r["p_value"] = float(p)
                        r["sig"] = _sig_star(p)
                except Exception:
                    pass

        # b) numeric features → t-tests
        for c in num_cols:
            if c == target:
                continue
            x = df.loc[s_target == 1, c].astype(float).dropna().values
            y = df.loc[s_target == 0, c].astype(float).dropna().values
            if len(x) >= 5 and len(y) >= 5:
                d = _cohens_d(x, y)
                p = None
                if _HAS_SCIPY:
                    try:
                        _, p = stats.ttest_ind(x, y, equal_var=False, nan_policy="omit")
                        p = float(p)
                    except Exception:
                        p = None
                ttest_rows.append({
                    "feature": c,
                    "mean_target1": float(np.nanmean(x)),
                    "mean_target0": float(np.nanmean(y)),
                    "diff": float(np.nanmean(x) - np.nanmean(y)),
                    "effect_size_d": float(d),
                    "p_value": p,
                    "sig": _sig_star(p) if p is not None else "",
                    "n1": int(len(x)),
                    "n0": int(len(y)),
                })

    out["chi_square"] = pd.DataFrame(chi_rows)
    out["t_tests"] = pd.DataFrame(ttest_rows)

    # ----------------- ANOVA (numeric target vs cat) -----------------
    anova_rows = []
    kpi_keywords = ["revenue", "sales", "amount", "price", "cost", "profit", "income", "gmv", "margin"]
    num_targets = [c for c in num_cols if any(k in c.lower() for k in kpi_keywords)]
    for ycol in num_targets[:3]:
        for xcol in cat_cols:
            g = df[[xcol, ycol]].dropna()
            if g.empty:
                continue
            grouped = [vals.values for _, vals in g.groupby(xcol)[ycol]]
            if len(grouped) < 2 or any(len(arr) < 3 for arr in grouped):
                continue
            p = None
            if _HAS_SCIPY:
                try:
                    _, p = stats.f_oneway(*grouped)
                    p = float(p)
                except Exception:
                    p = None
            means = g.groupby(xcol)[ycol].mean().sort_values(ascending=False)
            top = means.index[0] if not means.empty else None
            anova_rows.append({
                "numeric_target": ycol,
                "by_feature": xcol,
                "top_level": str(top) if top is not None else None,
                "top_mean": float(means.iloc[0]) if not means.empty else np.nan,
                "groups": int(means.size),
                "p_value": p,
                "sig": _sig_star(p) if p is not None else "",
            })
    out["anova"] = pd.DataFrame(anova_rows)

    # ----------------- Correlations -----------------
    corr_rows = []
    for i in range(len(num_cols)):
        for j in range(i+1, len(num_cols)):
            a, b = num_cols[i], num_cols[j]
            sa = pd.to_numeric(df[a], errors="coerce")
            sb = pd.to_numeric(df[b], errors="coerce")
            mask = sa.notna() & sb.notna()
            if mask.sum() < 10:
                continue
            r, p = (np.nan, None)
            if _HAS_SCIPY:
                try:
                    r, p = stats.pearsonr(sa[mask].values, sb[mask].values)
                    r = float(r); p = float(p)
                except Exception:
                    pass
            else:
                r = float(np.corrcoef(sa[mask].values, sb[mask].values)[0,1])
            corr_rows.append({
                "feature_a": a, "feature_b": b,
                "pearson_r": r, "p_value": p,
                "sig": _sig_star(p) if p is not None else ""
            })
    out["correlations"] = pd.DataFrame(corr_rows)

    # ----------------- Business bullets -----------------
    bullets: List[str] = []

    if not out["chi_square"].empty:
        for _, r in out["chi_square"].head(5).iterrows():
            bullets.append(
                f"Segment **{r['feature']}={r['level']}** → {r['rate_%']}% rate "
                f"(lift {r['lift_pts']} pts vs avg). {r['sig']}"
            )

    if not out["t_tests"].empty:
        for _, r in out["t_tests"].head(5).iterrows():
            bullets.append(
                f"Driver **{r['feature']}**: mean1={r['mean_target1']:.2f}, "
                f"mean0={r['mean_target0']:.2f}, diff={r['diff']:.2f} "
                f"(d={r['effect_size_d']:.2f}) {r['sig']}"
            )

    if not out["anova"].empty:
        for _, r in out["anova"].head(3).iterrows():
            bullets.append(
                f"KPI **{r['numeric_target']}** highest in {r['by_feature']}={r['top_level']} "
                f"(avg={r['top_mean']:.2f}) {r['sig']}"
            )

    if not out["correlations"].empty:
        for _, r in out["correlations"].head(3).iterrows():
            bullets.append(
                f"Correlation **{r['feature_a']} ↔ {r['feature_b']}**: r={r['pearson_r']:.2f} {r['sig']}"
            )

    out["business_bullets"] = bullets

    return out


# ---------- anomaly detection ----------

def detect_numeric_anomalies(df: pd.DataFrame, method: str = "zscore", threshold: float = 3.0):
    """
    Detect anomalies (outliers) in numeric columns using Z-score or IQR.
    Returns dict: {column: [row indices with anomalies]}
    """
    anomalies = {}
    numeric_cols = df.select_dtypes(include=[np.number]).columns

    for col in numeric_cols:
        series = df[col].dropna()

        if series.empty:
            continue

        if method == "zscore":
            z_scores = np.abs((series - series.mean()) / series.std(ddof=0))
            mask = z_scores > threshold
        elif method == "iqr":
            Q1, Q3 = np.percentile(series, [25, 75])
            IQR = Q3 - Q1
            mask = (series < (Q1 - 1.5 * IQR)) | (series > (Q3 + 1.5 * IQR))
        else:
            raise ValueError("method must be 'zscore' or 'iqr'")

        anomalies[col] = series[mask].index.tolist()

    return anomalies


def detect_trend_breaks(df: pd.DataFrame, time_col: str, value_col: str, window: int = 3, threshold: float = 2.0):
    """
    Detect trend breaks in a time series using rolling mean deviation.
    Returns DataFrame of anomalies with time, value, rolling_mean, deviation.
    """
    if time_col not in df.columns or value_col not in df.columns:
        return pd.DataFrame()

    d = df.copy()
    d[time_col] = pd.to_datetime(d[time_col], errors="coerce")
    d = d.dropna(subset=[time_col, value_col]).sort_values(time_col)

    if d.empty:
        return pd.DataFrame()

    d["rolling_mean"] = d[value_col].rolling(window=window, min_periods=1).mean()
    d["deviation"] = np.abs(d[value_col] - d["rolling_mean"])

    std_dev = d["deviation"].std()
    anomalies = d[d["deviation"] > threshold * std_dev]

    return anomalies[[time_col, value_col, "rolling_mean", "deviation"]]
