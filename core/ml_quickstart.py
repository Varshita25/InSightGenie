import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report, r2_score, mean_absolute_error, mean_squared_error
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor


def _prep_xy(df: pd.DataFrame, target: str):
    y = df[target]
    X = df.drop(columns=[target])

    # Drop empty columns
    nunique = X.nunique()
    drop_cols = nunique[nunique == 0].index.tolist()
    X = X.drop(columns=drop_cols, errors="ignore")

    # One-hot encode categoricals (limit high-cardinality)
    cat_cols = [c for c in X.columns if X[c].dtype == "object" or str(X[c].dtype).startswith("category")]
    for c in cat_cols:
        if X[c].nunique() > 50:
            X = X.drop(columns=[c])
    X = pd.get_dummies(X, drop_first=True)

    # Determine problem type and encode y if needed
    if y.dtype == "O" or str(y.dtype).startswith("category"):
        problem = "classification"
        y = y.astype("category").cat.codes
    elif pd.api.types.is_integer_dtype(y) or pd.api.types.is_float_dtype(y):
        problem = "regression"
    else:
        problem = "classification"
        y = y.astype("category").cat.codes

    # Remove NA rows
    data = pd.concat([X, y], axis=1).dropna()
    y = data.iloc[:, -1]
    X = data.iloc[:, :-1]

    # Limit width
    if X.shape[1] > 500:
        variances = X.var().sort_values(ascending=False)
        keep = variances.index[:500]
        X = X[keep]

    return X, y, problem


def ml_ui(df: pd.DataFrame):
    if df.shape[1] < 2:
        st.info("Need at least 2 columns to run ML.")
        return

    target = st.selectbox("Select target column", options=df.columns)
    if not target:
        return

    if st.button("Train baseline model"):
        with st.spinner("Training..."):
            X, y, problem = _prep_xy(df, target)
            if len(np.unique(y)) < 2:
                st.warning("Target has <2 unique values after cleaning; cannot train.")
                return

            strat = y if problem == "classification" else None
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=strat
            )

            if problem == "classification":
                model = RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)
                model.fit(X_train, y_train)
                preds = model.predict(X_test)
                acc = accuracy_score(y_test, preds)
                f1 = f1_score(y_test, preds, average="weighted")
                st.write({"accuracy": round(float(acc), 4), "f1_weighted": round(float(f1), 4)})
                st.text("Classification Report:\n" + classification_report(y_test, preds))
            else:
                model = RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1)
                model.fit(X_train, y_train)
                preds = model.predict(X_test)
                r2 = r2_score(y_test, preds)
                mae = mean_absolute_error(y_test, preds)
                rmse = mean_squared_error(y_test, preds, squared=False)
                st.write({"r2": round(float(r2), 4), "mae": round(float(mae), 4), "rmse": round(float(rmse), 4)})

            # Feature importance (top 15)
            importances = getattr(model, "feature_importances_", None)
            if importances is not None:
                imp = pd.Series(importances, index=X.columns).sort_values(ascending=False).head(15)
                st.bar_chart(imp)
            else:
                st.info("Model does not expose feature_importances_.")
