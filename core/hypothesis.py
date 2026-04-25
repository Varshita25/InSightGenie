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
    """Internal helper to run a statistical test between two columns following the 7 steps of hypothesis testing."""
    series_x, series_y = df[x], df[y]
    
    try:
        # Step 4 Selection & Rationale
        test_info = {}
        
        # Case 1: Numeric vs Numeric → Correlation Test
        if _is_numeric(series_x) and _is_numeric(series_y):
            clean = df[[x, y]].dropna()
            if clean.shape[0] < 10:
                return None
            
            test_name = "Pearson Correlation"
            r, p = stats.pearsonr(clean[x], clean[y])
            stat_val, p_val = float(r), float(p)
            
            h0 = f"There is no linear relationship between {x} and {y} (Correlation = 0)."
            ha = f"There is a significant linear relationship between {x} and {y}."
            rationale = f"We chose {test_name} because both variables are numeric. This test measures the strength and direction of the linear relationship between two continuous variables."
            stat_desc = f"Correlation Coefficient (r) = {r:.4f}"
            chart_type = "scatter"

        # Case 2: Numeric vs Categorical → T-Test / ANOVA
        elif (_is_numeric(series_x) and _is_categorical(series_y)) or (_is_numeric(series_y) and _is_categorical(series_x)):
            num_col, cat_col = (x, y) if _is_numeric(series_x) else (y, x)
            groups = df[[num_col, cat_col]].dropna().groupby(cat_col)[num_col].apply(list)
            if len(groups) < 2:
                return None

            if len(groups) == 2:
                test_name = "Independent T-Test"
                t, p = stats.ttest_ind(groups.iloc[0], groups.iloc[1], equal_var=False)
                stat_val, p_val = float(t), float(p)
                h0 = f"The mean of {num_col} is the same across both categories of {cat_col}."
                ha = f"The mean of {num_col} is significantly different between the two categories of {cat_col}."
                rationale = f"We chose an {test_name} because we are comparing a numeric value ({num_col}) across exactly two categorical groups in {cat_col}."
                stat_desc = f"T-statistic = {t:.4f}"
            else:
                test_name = "ANOVA (Analysis of Variance)"
                f, p = stats.f_oneway(*groups)
                stat_val, p_val = float(f), float(p)
                h0 = f"The mean of {num_col} is the same across all {len(groups)} categories of {cat_col}."
                ha = f"At least one category of {cat_col} has a significantly different mean for {num_col}."
                rationale = f"We chose {test_name} because we are comparing a numeric value ({num_col}) across more than two categorical groups ({len(groups)}) in {cat_col}."
                stat_desc = f"F-statistic = {f:.4f}"
            chart_type = "boxplot"

        # Case 3: Categorical vs Categorical → Chi-Square Test
        elif _is_categorical(series_x) and _is_categorical(series_y):
            tab = pd.crosstab(series_x, series_y)
            if tab.shape[0] < 2 or tab.shape[1] < 2:
                return None
            chi2, p, dof, _ = stats.chi2_contingency(tab)
            stat_val, p_val = float(chi2), float(p)
            test_name = "Chi-Square Test of Independence"
            h0 = f"{x} and {y} are completely independent of each other."
            ha = f"There is a significant association/dependency between {x} and {y}."
            rationale = f"We chose {test_name} because both variables are categorical. This test determines if there is a significant relationship between two nominal variables."
            stat_desc = f"Chi² Statistic = {chi2:.4f} (Degrees of Freedom = {dof})"
            chart_type = "bar"
        else:
            return None

        # Step 6 & 7: Conclusion
        rejected = p_val <= alpha
        decision = "Reject the Null Hypothesis (H₀)" if rejected else "Fail to Reject the Null Hypothesis (H₀)"
        conclusion = (
            f"Since the p-value ({p_val:.4f}) is {'less than or equal to' if rejected else 'greater than'} "
            f"the significance level (α={alpha}), we {decision.lower()}. "
            f"{'This suggests the effect is statistically significant.' if rejected else 'There is not enough evidence to claim a significant effect.'}"
        )

        return {
            "x": x, "y": y, "chart": chart_type,
            "test_name": test_name,
            "rationale": rationale,
            "alpha": alpha,
            "steps": {
                "Step 1: Null Hypothesis (H₀)": h0,
                "Step 2: Alternative Hypothesis (Hₐ)": ha,
                "Step 3: Significance Level (α)": f"α = {alpha}",
                "Step 4: Selected Test": f"{test_name} (Based on data types: {x} and {y})",
                "Step 5: Results": f"{stat_desc}, p-value = {p_val:.4f}",
                "Step 6: Comparison": f"p-value ({p_val:.4f}) {'≤' if rejected else '>'} α ({alpha})",
                "Step 7: Conclusion": conclusion
            }
        }

    except Exception:
        pass
    
    return None

