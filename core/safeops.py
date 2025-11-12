import pandas as pd

def to_numeric_safe(s: pd.Series):
    if pd.api.types.is_numeric_dtype(s):
        return s
    # remove commas and spaces, then coerce
    return pd.to_numeric(s.astype(str).str.replace(",", "", regex=False).str.strip(), errors="coerce")

def mean_by_group(df: pd.DataFrame, group: str, metric: str) -> pd.Series:
    g = df[[group, metric]].copy()
    g[metric] = to_numeric_safe(g[metric])
    if g[metric].notna().any():
        return g.dropna().groupby(group)[metric].mean().sort_values(ascending=False)
    # fallback: counts
    return g.groupby(group).size().sort_values(ascending=False).rename("count")

def rate_by_group(df: pd.DataFrame, group: str, binary: str) -> pd.Series:
    g = df[[group, binary]].copy()
    g[binary] = to_numeric_safe(g[binary])
    if g[binary].dropna().nunique() == 2:
        return (g.dropna().groupby(group)[binary].mean() * 100).round(2).rename("rate_%")
    # fallback: counts if not binary
    return g.groupby(group).size().sort_values(ascending=False).rename("count")
