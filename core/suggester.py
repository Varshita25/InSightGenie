# core/suggester.py — rewritten for systematic suggestions
import pandas as pd
from typing import List, Dict

def infer_types(df: pd.DataFrame):
    types = {}
    for c in df.columns:
        s = df[c]
        if pd.api.types.is_datetime64_any_dtype(s):
            types[c] = "datetime"
        elif pd.api.types.is_numeric_dtype(s):
            types[c] = "numeric"
        else:
            nunq = s.astype("object").nunique(dropna=False)
            types[c] = "categorical" if nunq <= max(50, len(df)//20) else "text"
    return types

def recommend_pairs(df: pd.DataFrame, target_hint: str | None = None, max_pairs: int = 12) -> List[Dict]:
    """
    Suggest analysis pairs with chart type and rationale.
    Returns: [{ 'title','chart','x','y','agg','why' }]
    """
    t = infer_types(df)
    cols = list(df.columns)
    n_cols = len(cols)

    # Try to guess binary target
    lower = [c.lower() for c in cols]
    target = None
    if target_hint and target_hint in cols:
        target = target_hint
    else:
        for key in ["survived","outcome","label","target","churn","default"]:
            if key in lower:
                target = cols[lower.index(key)]
                break
        if target is None:
            for c in cols:
                if t[c] == "numeric" and df[c].dropna().nunique() == 2:
                    target = c
                    break

    ideas = []

    # 1) Target rate by categorical
    if target is not None and df[target].dropna().nunique() == 2:
        for c in cols:
            if len(ideas) >= max_pairs * 2: break
            if c == target:
                continue
            if t[c] in ("categorical", "text"):
                ideas.append({
                    "title": f"{target} rate by {c}",
                    "chart": "bar_rate", "x": c, "y": target,
                    "agg": "rate",
                    "why": f"Helps understand how {target} differs across {c} groups."
                })

    # 2) Numeric vs Categorical → box or bar
    # Use a limited search for efficiency if there are many columns
    max_cats = 10
    max_nums = 10
    cats = [c for c in cols if t[c] in ("categorical", "text")][:max_cats]
    nums = [c for c in cols if t[c] == "numeric" and c != target][:max_nums]
    
    for cat in cats:
        if len(ideas) >= max_pairs * 3: break
        for num in nums:
            if len(ideas) >= max_pairs * 3: break
            ideas.append({
                "title": f"Distribution of {num} by {cat}",
                "chart": "boxplot", "x": num, "y": cat,
                "agg": None,
                "why": f"Shows how {num} varies across {cat} groups."
            })
            ideas.append({
                "title": f"Average {num} by {cat}",
                "chart": "bar", "x": cat, "y": num,
                "agg": "mean",
                "why": f"Compare mean {num} across categories of {cat}."
            })

    # 3) Numeric vs Numeric → scatter
    if len(ideas) < max_pairs * 3:
        for i in range(min(len(nums), 10)):
            if len(ideas) >= max_pairs * 4: break
            for j in range(i+1, min(len(nums), 10)):
                if len(ideas) >= max_pairs * 4: break
                a, b = nums[i], nums[j]
                ideas.append({
                    "title": f"{a} vs {b}",
                    "chart": "scatter", "x": a, "y": b,
                    "agg": None,
                    "why": f"Check linear/non-linear relationship between {a} and {b}."
                })

    # 4) Categorical vs Categorical → stacked bar
    if len(ideas) < max_pairs * 3:
        for i in range(min(len(cats), 10)):
            if len(ideas) >= max_pairs * 5: break
            for j in range(i+1, min(len(cats), 10)):
                if len(ideas) >= max_pairs * 5: break
                a, b = cats[i], cats[j]
                ideas.append({
                    "title": f"{a} × {b}",
                    "chart": "bar", "x": a, "y": b,
                    "agg": "count",
                    "why": f"See joint distribution of {a} and {b}."
                })

    # 5) Datetime vs Numeric → monthly trend
    dts = [c for c in cols if t[c] == "datetime"]
    if dts:
        vnum = next((c for c in cols if t[c] == "numeric"), None)
        if vnum:
            ideas.insert(0, {
                "title": f"Monthly trend of {vnum}",
                "chart": "line", "x": dts[0], "y": vnum,
                "agg": "M",
                "why": f"Tracks {vnum} over time to detect seasonality or trends."
            })

    # Deduplicate and cap
    seen, out = set(), []
    for it in ideas:
        key = (it["chart"], it["x"], it["y"], it.get("agg"))
        if key in seen:
            continue
        seen.add(key)
        out.append(it)
        if len(out) >= max_pairs:
            break
    return out


def beginner_questions(df: pd.DataFrame) -> List[str]:
    """
    Generate starter plain-English questions based on dataset types.
    """
    t = infer_types(df)
    qs = []

    qs.append("What is the overall shape of the dataset (rows, columns)?")
    qs.append("Which columns have the most missing values?")

    cat = [c for c in df.columns if t[c] in ("categorical","text")]
    num = [c for c in df.columns if t[c] == "numeric"]

    if cat:
        qs.append(f"What are the most common categories in {cat[0]}?")
    if num:
        qs.append(f"Show histogram of {num[0]}. Are there outliers?")
    if cat and num:
        qs.append(f"Compare average {num[0]} across {cat[0]} categories.")

    if len(num) >= 2:
        qs.append(f"Is there a relationship between {num[0]} and {num[1]}?")

    if len(cat) >= 2:
        qs.append(f"How do {cat[0]} and {cat[1]} interact?")

    return qs[:8]
