# core/nlq_llm.py
import os
import re
import json
from typing import Dict, Any, Tuple, Optional

import duckdb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sqlparse
from dotenv import load_dotenv
from openai import OpenAI


load_dotenv()


def _summarize_schema(df: pd.DataFrame, max_samples: int = 5) -> str:
    """Return a compact schema description with dtypes and a few sample values per column."""
    parts = []
    for c in df.columns:
        s = df[c]
        if pd.api.types.is_datetime64_any_dtype(s):
            dtype = "datetime"
            samples = s.dropna().astype(str).head(max_samples).tolist()
        elif pd.api.types.is_numeric_dtype(s):
            dtype = "numeric"
            samples = s.dropna().astype(float).round(4).head(max_samples).astype(str).tolist()
        else:
            dtype = "categorical/text"
            samples = s.dropna().astype(str).head(max_samples).tolist()
        parts.append(f"- {c} ({dtype}); samples: {samples}")
    return "\n".join(parts)


def _extract_sql(text: str) -> Optional[str]:
    """Get SQL from ```sql ...``` fences or return None."""
    m = re.search(r"```sql(.*?)```", text, flags=re.S)
    if m:
        return m.group(1).strip()
    # fallback: try plain SELECT
    m = re.search(r"(select\s+.+)", text, flags=re.I | re.S)
    return m.group(1).strip() if m else None


def _is_safe_sql(sql: str) -> bool:
    """Very conservative safety check: only allow a single SELECT; no DDL/DML; limit rows."""
    parsed = sqlparse.parse(sql)
    if len(parsed) != 1:
        return False
    stmt = parsed[0]
    # Disallow any non-SELECT tokens
    if stmt.get_type() != "SELECT":
        return False
    txt = sql.lower()
    banned = [";", " drop ", " delete ", " update ", " insert ", " alter ", " create ", " replace "]
    if any(b in txt for b in banned):
        return False
    return True


def _ensure_limit(sql: str, default_limit: int = 200) -> str:
    """Append LIMIT if user/model didn't add one."""
    txt = sql.strip().rstrip(";")
    if re.search(r"\blimit\b\s+\d+", txt, flags=re.I):
        return txt
    return f"{txt} LIMIT {default_limit}"


def _pick_chart(result: pd.DataFrame) -> str:
    """
    Decide chart type from result shape:
    - 1 numeric -> histogram
    - 2 cols (cat,num) -> bar
    - 2 nums -> scatter
    - time + num -> line
    - else -> table only
    """
    if result.empty:
        return "table"
    cols = result.columns.tolist()
    dtypes = {c: ("date" if pd.api.types.is_datetime64_any_dtype(result[c]) else
                  "num" if pd.api.types.is_numeric_dtype(result[c]) else "cat") for c in cols}
    # time + num (prefer)
    if any(d == "date" for d in dtypes.values()) and any(d == "num" for d in dtypes.values()):
        return "line"
    if len(cols) == 1 and dtypes[cols[0]] == "num":
        return "hist"
    if len(cols) == 2:
        a, b = cols[0], cols[1]
        if dtypes[a] == "cat" and dtypes[b] == "num":
            return "bar"
        if dtypes[a] == "num" and dtypes[b] == "cat":
            return "bar_swap"
        if dtypes[a] == "num" and dtypes[b] == "num":
            return "scatter"
    return "table"


def _plot(result: pd.DataFrame, chart: str):
    fig = None
    if chart == "hist":
        col = result.columns[0]
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(result[col].dropna().values, bins=30)
        ax.set_title(f"Distribution of {col}")
        ax.set_xlabel(col); ax.set_ylabel("Count"); fig.tight_layout()
    elif chart in ("bar", "bar_swap"):
        x, y = result.columns[:2]
        if chart == "bar_swap":
            x, y = y, x
        fig, ax = plt.subplots(figsize=(6, 4))
        # keep top 40 to avoid overcrowding
        tmp = result[[x, y]].copy().dropna().head(40)
        ax.bar([str(v) for v in tmp[x]], tmp[y].values)
        ax.set_title(f"{y} by {x}")
        ax.set_xlabel(x); ax.set_ylabel(y)
        ax.set_xticklabels([str(v)[:16] for v in tmp[x]], rotation=45, ha="right")
        fig.tight_layout()
    elif chart == "scatter":
        a, b = result.columns[:2]
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.scatter(result[a], result[b], s=16)
        ax.set_title(f"{a} vs {b}")
        ax.set_xlabel(a); ax.set_ylabel(b); fig.tight_layout()
    elif chart == "line":
        # choose first date and first numeric columns
        dcol = next(c for c in result.columns if pd.api.types.is_datetime64_any_dtype(result[c]))
        ncol = next(c for c in result.columns if pd.api.types.is_numeric_dtype(result[c]))
        tmp = result[[dcol, ncol]].dropna().sort_values(dcol)
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(tmp[dcol], tmp[ncol])
        ax.set_title(f"{ncol} over {dcol}"); ax.set_xlabel(dcol); ax.set_ylabel(ncol); fig.tight_layout()
    return fig


def _explain_text(client: OpenAI, question: str, df_head: str, sql: str) -> str:
    """Ask LLM to explain the answer in plain English given head of result + SQL."""
    sys = (
        "You are a helpful data analyst. Explain results clearly for students and business users. "
        "Be concise, use percentages where useful, and avoid jargon. If the sample is partial, say so."
    )
    user = (
        f"Question: {question}\n"
        f"SQL used: {sql}\n"
        f"Result head (first rows):\n{df_head}\n\n"
        "Write a short explanation (3–6 bullets or a short paragraph) and mention any caveats."
    )
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "system", "content": sys}, {"role": "user", "content": user}],
        temperature=0.2,
    )
    return resp.choices[0].message.content


def answer_query_llm(df: pd.DataFrame, question: str) -> Dict[str, Any]:
    """
    NL → (safe) SQL via OpenAI → DuckDB execution → auto-chart → LLM explanation.
    Returns: {'text': str, 'figure': fig|None, 'table': pd.DataFrame|None, 'sql': str}
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return {"text": "OpenAI API key not set. Add OPENAI_API_KEY in .env or environment.", "figure": None, "table": None, "sql": ""}

    client = OpenAI(api_key=api_key)

    # 1) Ask LLM for a SELECT SQL over a table named df
    schema = _summarize_schema(df)
    sys = (
        "You write a SINGLE DuckDB SQL SELECT query over a table called df. "
        "Do not mutate data. Prefer grouping & aggregation for summarization. "
        "Use valid DuckDB functions. If date fields exist, you may CAST/DATE_TRUNC. "
        "Return ONLY the SQL inside ```sql fences with LIMIT 200."
    )
    user = f"User question: {question}\nSchema:\n{schema}\nConstraints:\n- Table name: df\n- Return one SELECT with LIMIT 200."
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "system", "content": sys}, {"role": "user", "content": user}],
        temperature=0.1,
    )
    raw = resp.choices[0].message.content or ""
    sql = _extract_sql(raw) or ""
    sql = _ensure_limit(sql)
    if not sql or not _is_safe_sql(sql):
        return {"text": "I couldn't produce a safe SELECT query for that request. Try rephrasing your question.", "figure": None, "table": None, "sql": sql}

    # 2) Execute with DuckDB
    con = duckdb.connect()
    con.register("df", df)
    try:
        result = con.execute(sql).df()
    except Exception as e:
        return {"text": f"SQL failed: {e}", "figure": None, "table": None, "sql": sql}
    finally:
        con.unregister("df")
        con.close()

    # 3) Pick chart + plot
    chart = _pick_chart(result)
    fig = _plot(result, chart)

    # 4) Explanation text from LLM (use only result head to avoid tokens)
    head_txt = result.head(10).to_markdown(index=False)
    explanation = _explain_text(client, question, head_txt, sql)

    return {"text": explanation, "figure": fig, "table": result, "sql": sql}
