# core/report.py
from typing import List, Dict
import pandas as pd
import html

def build_html_report(df: pd.DataFrame, profile: Dict, insights: List[str], figs64: List[str]) -> bytes:
    parts = []
    parts.append("<h1>AI Data Insight Assistant – Report</h1>")
    parts.append(f"<p>Rows: {len(df):,} | Cols: {df.shape[1]}</p>")
    parts.append("<h2>Insights</h2><ul>")
    parts += [f"<li>{html.escape(tip)}</li>" for tip in insights]
    parts.append("</ul>")
    if figs64:
        parts.append("<h2>Figures</h2>")
        for b64 in figs64:
            parts.append(f'<img src="data:image/png;base64,{b64}" style="max-width: 640px; display:block; margin:8px 0;" />')
    return "\n".join(parts).encode("utf-8")
