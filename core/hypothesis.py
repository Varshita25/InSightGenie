# core/hypothesis.py
import numpy as np
import pandas as pd
from typing import List, Dict
from scipy import stats


# ---------------- Helper Functions ----------------
def _is_numeric(series: pd.Series) -> bool:
    return pd.api.types.is_numeric_dtype(series)


def _is_categorical(series: pd.Series) -> bool:
    return (
        pd.api.types.is_object_dtype(series) or
        pd.api.types.is_categorical_dtype(series) or
        series.nunique(dropna=True) < 20
    )


def _is_id_column(series: pd.Series, name: str) -> bool:
    """Detect ID-like columns (monotonic ints or mostly unique)."""
    if series.nunique(dropna=True) > 0.9 * len(series):
        return True
    if str(name).lower().endswith("id"):
        return True
    return False


# ---------------- Hypothesis Generator ----------------
def generate_hypotheses(
    df: pd.DataFrame, 
    n: int = 5, 
    alpha: float = 0.05
) -> List[Dict]:
    """Generate meaningful hypotheses with appropriate statistical tests."""
    # Speed optimization: sample if dataset is too large
    if len(df) > 10000:
        df_sample = df.sample(10000, random_state=42)
    else:
        df_sample = df

    results = []
    # Identify valid columns (exclude IDs and high-cardinality text)
    cols = [c for c in df.columns if not _is_id_column(df[c], c)]
    if not cols:
        return []

    # Target-centric approach: prioritize tests involving the first column (the target)
    target = cols[0]
    others = cols[1:]
    
    # We will iterate through pairs until we find 'n' results
    # Phase 1: Target vs Others
    for other in others:
        if len(results) >= n:
            break
        
        res = _run_test(df_sample, target, other, alpha)
        if res:
            results.append(res)
            
    # Phase 2: If we still need more, check other pairs (limited search)
    if len(results) < n:
        max_extra_checks = 50
        checks = 0
        for i in range(1, len(cols)):
            if len(results) >= n or checks >= max_extra_checks:
                break
            for j in range(i + 1, len(cols)):
                if len(results) >= n or checks >= max_extra_checks:
                    break
                checks += 1
                res = _run_test(df_sample, cols[i], cols[j], alpha)
                if res:
                    results.append(res)

    return results[:n]


def _run_test(df: pd.DataFrame, x: str, y: str, alpha: float) -> Dict | None:
    """Internal helper to run a statistical test between two columns."""
    series_x, series_y = df[x], df[y]
    
    try:
        # Case 1: Numeric vs Numeric → Correlation Test
        if _is_numeric(series_x) and _is_numeric(series_y):
            clean = df[[x, y]].dropna()
            if clean.shape[0] < 10:
                return None
            r, p = stats.pearsonr(clean[x], clean[y])
            h0 = f"There is no linear correlation between {x} and {y}."
            h1 = f"There is a significant linear correlation between {x} and {y}."
            decision = "Reject H₀" if p <= alpha else "Fail to reject H₀"
            interpretation = f"Pearson r={r:.2f}, p={p:.4f}. {decision}."

            return {
                "x": x, "y": y, "chart": "scatter",
                "H0": h0, "H1": h1,
                "test": "Pearson Correlation",
                "statistic": float(r), "p_value": float(p),
                "decision": decision,
                "interpretation": interpretation
            }

        # Case 2: Numeric vs Categorical → T-Test / ANOVA
        # Ensure we know which one is numeric for the test logic
        num_col, cat_col = (x, y) if _is_numeric(series_x) else (y, x)
        if _is_numeric(df[num_col]) and _is_categorical(df[cat_col]):
            groups = df[[num_col, cat_col]].dropna().groupby(cat_col)[num_col].apply(list)
            if len(groups) < 2:
                return None

            if len(groups) == 2:
                # Independent T-test
                t, p = stats.ttest_ind(groups.iloc[0], groups.iloc[1], equal_var=False)
                h0 = f"The mean of {num_col} is the same across groups of {cat_col}."
                h1 = f"The mean of {num_col} differs between groups of {cat_col}."
                decision = "Reject H₀" if p <= alpha else "Fail to reject H₀"
                interpretation = f"T-test statistic={t:.2f}, p={p:.4f}. {decision}."
                test = "Independent T-Test"
                stat_val = t
            else:
                # ANOVA
                f, p = stats.f_oneway(*groups)
                h0 = f"The mean of {num_col} is the same across all categories of {cat_col}."
                h1 = f"At least one category of {cat_col} has a different mean of {num_col}."
                decision = "Reject H₀" if p <= alpha else "Fail to reject H₀"
                interpretation = f"ANOVA F={f:.2f}, p={p:.4f}. {decision}."
                test = "ANOVA"
                stat_val = f

            return {
                "x": cat_col, "y": num_col, "chart": "boxplot",
                "H0": h0, "H1": h1,
                "test": test,
                "statistic": float(stat_val), "p_value": float(p),
                "decision": decision,
                "interpretation": interpretation
            }

        # Case 3: Categorical vs Categorical → Chi-Square Test
        elif _is_categorical(series_x) and _is_categorical(series_y):
            tab = pd.crosstab(series_x, series_y)
            if tab.shape[0] < 2 or tab.shape[1] < 2:
                return None
            chi2, p, dof, _ = stats.chi2_contingency(tab)
            h0 = f"{x} and {y} are independent."
            h1 = f"{x} and {y} are associated (dependent)."
            decision = "Reject H₀" if p <= alpha else "Fail to reject H₀"
            interpretation = f"Chi²={chi2:.2f}, dof={dof}, p={p:.4f}. {decision}."

            return {
                "x": x, "y": y, "chart": "bar",
                "H0": h0, "H1": h1,
                "test": "Chi-Square Test",
                "statistic": float(chi2), "p_value": float(p),
                "decision": decision,
                "interpretation": interpretation
            }

    except Exception:
        pass
    
    return None

