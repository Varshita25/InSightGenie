import os
import pandas as pd
import numpy as np
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
import google.generativeai as genai
from dotenv import load_dotenv

# --- Load Google API key ---
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
if api_key:
    genai.configure(api_key=api_key)
else:
    print("⚠️ Warning: GOOGLE_API_KEY not set. Gemini Q&A will not work.")

# ---------------- Helper: Gemini ----------------
def ask_gemini(prompt: str, model="gemini-1.5-flash", temperature=0.3):
    """
    Query Gemini for natural language explanations.
    """
    if not api_key:
        return "⚠️ Gemini API Key missing. Please set GEMINI_API_KEY in your environment."
    try:
        response = genai.GenerativeModel(model).generate_content(
            prompt,
            generation_config={"temperature": temperature}
        )
        return response.text
    except Exception as e:
        return f"⚠️ Gemini API Error: {str(e)}"

# ---------------- Helper: Detect Target Column ----------------
def _detect_target(df: pd.DataFrame) -> str | None:
    """Try to guess a target column (like Survived, Outcome, Target)."""
    lower = [c.lower() for c in df.columns]
    for key in ["survived", "outcome", "label", "target", "churn", "default"]:
        if key in lower:
            return df.columns[lower.index(key)]
    return None

# ---------------- Main Local Answering ----------------
def answer(q: str, df: pd.DataFrame) -> dict:
    """
    Main Q&A logic for Ask-the-Data tab.
    Returns dict with type, x, y, and explanation.
    """
    q_lower = q.lower()

    # --- Feature importance ---
    if "best feature" in q_lower or "important feature" in q_lower or "which feature" in q_lower:
        target = _detect_target(df)
        if target is not None:
            y = df[target].dropna()
            X = df.drop(columns=[target]).fillna("missing")

            # Encode categorical
            X_enc = pd.get_dummies(X, drop_first=True)

            if y.nunique() <= 10:  # classification
                scores = mutual_info_classif(X_enc, y, discrete_features="auto")
            else:  # regression
                scores = mutual_info_regression(X_enc, y)

            ranking = pd.Series(scores, index=X_enc.columns).sort_values(ascending=False)
            top = ranking.head(5).round(3)

            return {
                "type": "bar",
                "x": top.index.tolist(),
                "y": top.values.tolist(),
                "text": "Feature importance ranking.",
                "explain": f"Top features influencing **{target}**: {', '.join(top.index)}."
            }
        else:
            return {"type": "table", "x": None, "y": None,
                    "text": "Could not detect target variable for feature importance.",
                    "explain": "Please specify a target column like Survived, Outcome, or Target."}

    # --- Histogram ---
    if "histogram" in q_lower or "distribution" in q_lower or "kde" in q_lower:
        for c in df.select_dtypes(include="number").columns:
            if c.lower() in q_lower:
                return {"type": "histogram", "x": c, "y": None,
                        "text": f"Histogram of {c} showing distribution.",
                        "explain": f"The histogram shows spread & skewness of {c}."}

    # --- Boxplot ---
    if "box" in q_lower or "outlier" in q_lower or "spread" in q_lower:
        for c in df.select_dtypes(include="number").columns:
            if c.lower() in q_lower:
                return {"type": "boxplot", "x": c, "y": None,
                        "text": f"Boxplot of {c}.",
                        "explain": f"Boxplot shows median, quartiles and outliers of {c}."}

    # --- Scatter ---
    if "scatter" in q_lower or "relationship" in q_lower or "vs" in q_lower:
        nums = df.select_dtypes(include="number").columns
        for c1 in nums:
            for c2 in nums:
                if c1 != c2 and c1.lower() in q_lower and c2.lower() in q_lower:
                    return {"type": "scatter", "x": c1, "y": c2,
                            "text": f"Scatter plot of {c1} vs {c2}.",
                            "explain": f"Scatter plot shows correlation between {c1} and {c2}."}

    # --- Trend ---
    if "trend" in q_lower or "over time" in q_lower or "monthly" in q_lower:
        for c in df.columns:
            if pd.api.types.is_datetime64_any_dtype(df[c]):
                num_col = next((col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])), None)
                if num_col:
                    return {"type": "line", "x": c, "y": num_col,
                            "text": f"Monthly trend of {num_col} over {c}.",
                            "explain": f"Line chart shows how {num_col} evolves over {c}."}

    # --- Bar chart ---
    if "bar" in q_lower or "compare" in q_lower or "average" in q_lower or "by" in q_lower:
        cats = [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])]
        nums = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        if cats and nums:
            return {"type": "bar", "x": cats[0], "y": nums[0],
                    "text": f"Average {nums[0]} across {cats[0]}.",
                    "explain": f"Bar chart compares mean {nums[0]} across categories of {cats[0]}."}

    # --- Correlation heatmap ---
    if "correlation" in q_lower or "heatmap" in q_lower:
        return {"type": "heatmap", "x": None, "y": None,
                "text": "Correlation heatmap of numeric features.",
                "explain": "Shows strength of pairwise linear relationships."}

    # --- Missing / duplicates / describe ---
    if "missing" in q_lower:
        return {"type": "table", "x": None, "y": None,
                "text": "Missing values by column.",
                "explain": "Shows which columns have the most missing data."}

    if "duplicate" in q_lower:
        return {"type": "table", "x": None, "y": None,
                "text": f"Dataset contains {df.duplicated().sum()} duplicate rows.",
                "explain": "Duplicate rows should usually be removed."}

    if "describe" in q_lower or "summary" in q_lower:
        return {"type": "table", "x": None, "y": None,
                "text": "Statistical summary of numeric and categorical columns.",
                "explain": "Shows count, mean, std, min, quartiles, and max values."}

    # --- Default: Gemini Q&A ---
    dataset_preview = df.head(10).to_dict()
    gpt_answer = ask_gemini(f"""
    You are a data analyst. 
    Dataset (preview of first 10 rows): {dataset_preview}
    User question: {q}
    Answer clearly in plain English, as if explaining to a beginner.
    """)
    return {"type": "text", "x": None, "y": None,
            "text": "Gemini-powered Q&A.",
            "explain": gpt_answer}

# ---------------- Suggested Questions ----------------
def suggest_questions(df: pd.DataFrame, max_q: int = 6) -> list[str]:
    """
    Generate dataset-specific starter questions for Ask-the-Data tab.
    """
    qs = []
    cols = df.columns.tolist()

    # Categorical questions
    cats = [c for c in cols if not pd.api.types.is_numeric_dtype(df[c])]
    if cats:
        qs.append(f"What are the top values of {cats[0]}?")
        if len(cats) > 1:
            qs.append(f"Compare {cats[0]} vs {cats[1]}.")

    # Numeric questions
    nums = [c for c in cols if pd.api.types.is_numeric_dtype(df[c])]
    if nums:
        qs.append(f"Show histogram of {nums[0]}.")
        if len(nums) > 1:
            qs.append(f"Show scatter plot of {nums[0]} vs {nums[1]}.")
        qs.append(f"Are there outliers in {nums[0]}? (boxplot)")

    # Time series
    dts = [c for c in cols if pd.api.types.is_datetime64_any_dtype(df[c])]
    if dts and nums:
        qs.append(f"Show monthly trend of {nums[0]} over {dts[0]}.")

    # Feature importance
    target = _detect_target(df)
    if target:
        qs.append(f"Which features are most important for predicting {target}?")

    return qs[:max_q]
