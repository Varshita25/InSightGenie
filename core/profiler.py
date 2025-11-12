import pandas as pd
import numpy as np


def _dtype_map(series: pd.Series) -> str:
    if pd.api.types.is_datetime64_any_dtype(series):
        return "datetime"
    if pd.api.types.is_numeric_dtype(series):
        return "numeric"
    return "categorical"


def basic_profile(df: pd.DataFrame) -> dict:
    shape = df.shape
    missing_total = int(df.isna().sum().sum())
    duplicates = int(df.duplicated().sum())
    dtypes = {c: _dtype_map(df[c]) for c in df.columns}

    miss_by_col = (df.isna().mean() * 100).round(2).rename("missing_%")
    miss_df = miss_by_col.to_frame().reset_index().rename(columns={"index": "column"})

    numeric_cols = [c for c, t in dtypes.items() if t == "numeric"]
    cat_cols = [c for c, t in dtypes.items() if t == "categorical"]

    num_stats = df[numeric_cols].describe().T if numeric_cols else pd.DataFrame()
    if not num_stats.empty:
        num_stats["skew"] = df[numeric_cols].skew(numeric_only=True)
        num_stats["kurt"] = df[numeric_cols].kurt(numeric_only=True)

    top_cats = {}
    for c in cat_cols:
        vc = df[c].astype("category").value_counts(dropna=False).head(10)
        total = len(df)
        top_cats[c] = pd.DataFrame(
            {
                "value": vc.index.astype(str),
                "count": vc.values,
                "pct": (vc.values / total * 100).round(2),
            }
        )

    corr = df[numeric_cols].corr(numeric_only=True) if numeric_cols else pd.DataFrame()

    return {
        "shape": shape,
        "missing_total": missing_total,
        "duplicates": duplicates,
        "dtypes": dtypes,
        "missing_by_col": miss_df,
        "numeric_stats": num_stats,
        "top_categories": top_cats,
        "corr": corr,
    }
