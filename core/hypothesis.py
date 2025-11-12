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
    results = []
    cols = [c for c in df.columns if not _is_id_column(df[c], c)]

    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            x, y = cols[i], cols[j]
            series_x, series_y = df[x], df[y]

            try:
                # ================================================================
                # Case 1: Numeric vs Numeric → Correlation Test
                # ================================================================
                if _is_numeric(series_x) and _is_numeric(series_y):
                    clean = df[[x, y]].dropna()
                    if clean.shape[0] < 10:
                        continue
                    r, p = stats.pearsonr(clean[x], clean[y])
                    h0 = f"There is no linear correlation between {x} and {y}."
                    h1 = f"There is a significant linear correlation between {x} and {y}."
                    decision = "Reject H₀" if p <= alpha else "Fail to reject H₀"
                    interpretation = f"Pearson r={r:.2f}, p={p:.4f}. {decision}."

                    results.append({
                        "x": x, "y": y, "chart": "scatter",
                        "H0": h0, "H1": h1,
                        "test": "Pearson Correlation",
                        "statistic": r, "p_value": p,
                        "decision": decision,
                        "interpretation": interpretation
                    })

                # ================================================================
                # Case 2: Numeric vs Categorical → T-Test / ANOVA
                # ================================================================
                elif _is_numeric(series_x) and _is_categorical(series_y):
                    groups = df[[x, y]].dropna().groupby(y)[x].apply(list)
                    if len(groups) < 2:
                        continue

                    if len(groups) == 2:
                        # Independent T-test
                        t, p = stats.ttest_ind(groups.iloc[0], groups.iloc[1], equal_var=False)
                        h0 = f"The mean of {x} is the same across groups of {y}."
                        h1 = f"The mean of {x} differs between groups of {y}."
                        decision = "Reject H₀" if p <= alpha else "Fail to reject H₀"
                        interpretation = f"T-test statistic={t:.2f}, p={p:.4f}. {decision}."
                        test = "Independent T-Test"
                        stat_val = t
                    else:
                        # ANOVA
                        f, p = stats.f_oneway(*groups)
                        h0 = f"The mean of {x} is the same across all categories of {y}."
                        h1 = f"At least one category of {y} has a different mean of {x}."
                        decision = "Reject H₀" if p <= alpha else "Fail to reject H₀"
                        interpretation = f"ANOVA F={f:.2f}, p={p:.4f}. {decision}."
                        test = "ANOVA"
                        stat_val = f

                    results.append({
                        "x": y, "y": x, "chart": "boxplot",
                        "H0": h0, "H1": h1,
                        "test": test,
                        "statistic": float(stat_val), "p_value": p,
                        "decision": decision,
                        "interpretation": interpretation
                    })

                # ================================================================
                # Case 3: Categorical vs Categorical → Chi-Square Test
                # ================================================================
                elif _is_categorical(series_x) and _is_categorical(series_y):
                    tab = pd.crosstab(series_x, series_y)
                    if tab.shape[0] < 2 or tab.shape[1] < 2:
                        continue
                    chi2, p, dof, _ = stats.chi2_contingency(tab)
                    h0 = f"{x} and {y} are independent."
                    h1 = f"{x} and {y} are associated (dependent)."
                    decision = "Reject H₀" if p <= alpha else "Fail to reject H₀"
                    interpretation = f"Chi²={chi2:.2f}, dof={dof}, p={p:.4f}. {decision}."

                    results.append({
                        "x": x, "y": y, "chart": "bar",
                        "H0": h0, "H1": h1,
                        "test": "Chi-Square Test",
                        "statistic": chi2, "p_value": p,
                        "decision": decision,
                        "interpretation": interpretation
                    })

            except Exception:
                continue

    # Return only top n hypotheses
    return results[:n]
