# core/feedback.py
from __future__ import annotations
import csv, os, time, hashlib
from typing import Optional, Dict

_FEEDBACK_PATH = "feedback.csv"

def _id_from_text(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()[:12]

def save_feedback(kind: str, text: str, meta: Optional[Dict]=None) -> None:
    """
    kind: 'up' | 'down'
    """
    row = {
        "ts": int(time.time()),
        "kind": kind,
        "item_id": _id_from_text(text),
        "text": text,
        "meta": (str(meta) if meta else ""),
    }
    exists = os.path.exists(_FEEDBACK_PATH)
    with open(_FEEDBACK_PATH, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not exists:
            w.writeheader()
        w.writerow(row)

def aggregate() -> Dict[str, Dict[str,int]]:
    out: Dict[str, Dict[str,int]] = {}
    if not os.path.exists(_FEEDBACK_PATH):
        return out
    with open(_FEEDBACK_PATH, "r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            iid = row["item_id"]
            out.setdefault(iid, {"up":0, "down":0})
            out[iid][row["kind"]] += 1
    return out
