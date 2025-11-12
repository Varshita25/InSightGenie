import pandas as pd
import streamlit as st

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

def safe_dataframe(df: pd.DataFrame, **kwargs):
    """
    Safely display a DataFrame in Streamlit without PyArrow serialization errors.
    Falls back to converting columns to string if Arrow conversion fails.
    """
    try:
        st.dataframe(df, **kwargs)
    except Exception as e:
        # If Arrow conversion fails, convert all columns to string and retry
        if "Arrow" in str(e) or "Conversion failed" in str(e):
            try:
                df_safe = df.copy()
                for col in df_safe.columns:
                    if not pd.api.types.is_numeric_dtype(df_safe[col]):
                        df_safe[col] = df_safe[col].astype(str)
                st.dataframe(df_safe, **kwargs)
            except Exception as e2:
                st.warning(f"Could not display DataFrame: {str(e2)}")
        else:
            raise
