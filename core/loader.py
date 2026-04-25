# core/loader.py
from __future__ import annotations
import os, io, sqlite3, tempfile
from typing import Tuple, Dict, Optional
import pandas as pd
import duckdb

try:
    import pdfplumber  # best-effort tables
    _HAS_PDFPLUMBER = True
except Exception:
    _HAS_PDFPLUMBER = False

try:
    from PyPDF2 import PdfReader
    _HAS_PYPDF2 = True
except Exception:
    _HAS_PYPDF2 = False


def _info_from_df(df: pd.DataFrame) -> Dict[str, object]:
    mem_mb = df.memory_usage(deep=True).sum() / (1024**2) if not df.empty else 0.0
    return {
        "rows": int(df.shape[0]),
        "cols": int(df.shape[1]),
        "memory_mb": float(mem_mb),
    }


def _load_csv_excel(file) -> Tuple[pd.DataFrame, Dict[str, object]]:
    if hasattr(file, "seek"):
        file.seek(0)
    name = getattr(file, "name", "uploaded")
    ext = os.path.splitext(name)[1].lower()
    try:
        if ext in [".xlsx", ".xls"]:
            df = pd.read_excel(file)
        else:
            df = pd.read_csv(file)
    except Exception as e:
        # Fallback for weird encodings or formats
        if hasattr(file, "seek"):
            file.seek(0)
        try:
            df = pd.read_csv(file, encoding='latin1')
        except:
            raise e
    return df, _info_from_df(df)


def _save_temp_file(file) -> str:
    # persist uploaded binary to a temp path for SQL/PDF libs
    if hasattr(file, "seek"):
        file.seek(0)
    data = file.read()
    suffix = os.path.splitext(getattr(file, "name", "upload.bin"))[1]
    fd, path = tempfile.mkstemp(suffix=suffix)
    with os.fdopen(fd, "wb") as f:
        f.write(data)
    return path


def _load_sqlite(path: str, table: Optional[str] = None, query: Optional[str] = None) -> Tuple[pd.DataFrame, Dict[str, object]]:
    conn = sqlite3.connect(path)
    try:
        if query:
            df = pd.read_sql_query(query, conn)
        else:
            # pick first table if not given
            tables = pd.read_sql_query("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name;", conn)
            t = table or (tables["name"].iloc[0] if not tables.empty else None)
            if not t:
                return pd.DataFrame(), {"rows":0, "cols":0, "memory_mb":0.0}
            df = pd.read_sql_query(f"SELECT * FROM {t} LIMIT 100000", conn)
    finally:
        conn.close()
    return df, _info_from_df(df)


def _load_duckdb(path: str, table: Optional[str] = None, query: Optional[str] = None) -> Tuple[pd.DataFrame, Dict[str, object]]:
    con = duckdb.connect(database=path, read_only=True)
    try:
        if query:
            df = con.execute(query).df()
        else:
            # show tables → pick first
            tbls = con.execute("SHOW TABLES").df()
            t = table or (tbls["name"].iloc[0] if not tbls.empty else None)
            if not t:
                return pd.DataFrame(), {"rows":0, "cols":0, "memory_mb":0.0}
            df = con.execute(f"SELECT * FROM {t} LIMIT 100000").df()
    finally:
        con.close()
    return df, _info_from_df(df)


def _load_pdf(path: str) -> Tuple[pd.DataFrame, Dict[str, object]]:
    # Best-effort: try to extract tables; else return page_text
    tables: list[pd.DataFrame] = []
    if _HAS_PDFPLUMBER:
        try:
            with pdfplumber.open(path) as pdf:
                for page in pdf.pages:
                    try:
                        tbls = page.extract_tables() or []
                        for t in tbls:
                            df = pd.DataFrame(t)
                            if df.shape[1] > 1:
                                tables.append(df)
                    except Exception:
                        pass
        except Exception:
            pass

    if tables:
        # concat with MultiIndex page-wise
        out = []
        for i, t in enumerate(tables, 1):
            t.columns = [f"col_{j}" for j in range(t.shape[1])]
            t.insert(0, "source_page", i)
            out.append(t)
        pdf_df = pd.concat(out, ignore_index=True)
        return pdf_df, _info_from_df(pdf_df)

    # fallback: text per page
    texts = []
    if _HAS_PYPDF2:
        try:
            reader = PdfReader(path)
            for i, p in enumerate(reader.pages, 1):
                texts.append({"page": i, "page_text": p.extract_text() or ""})
        except Exception:
            pass

    df = pd.DataFrame(texts) if texts else pd.DataFrame({"page_text": ["(Could not extract text)"]})
    return df, _info_from_df(df)


def load_table(
    file,
    source: str = "auto",
    sql_table: Optional[str] = None,
    sql_query: Optional[str] = None,
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    """
    source: 'auto' | 'csv' | 'excel' | 'sqlite' | 'duckdb' | 'pdf'
    """
    name = getattr(file, "name", "uploaded")
    ext = os.path.splitext(name)[1].lower()

    if source == "auto":
        if ext in [".csv", ".xlsx", ".xls"]:
            source = "excel" if ext in [".xlsx", ".xls"] else "csv"
        elif ext in [".db", ".sqlite", ".sqlite3"]:
            source = "sqlite"
        elif ext in [".duckdb"]:
            source = "duckdb"
        elif ext == ".pdf":
            source = "pdf"
        else:
            source = "csv"

    if source in ["csv", "excel"]:
        return _load_csv_excel(file)

    if source in ["sqlite", "duckdb", "pdf"]:
        path = _save_temp_file(file)
        if source == "sqlite":
            return _load_sqlite(path, sql_table, sql_query)
        if source == "duckdb":
            return _load_duckdb(path, sql_table, sql_query)
        return _load_pdf(path)

    # fallback
    df = pd.DataFrame()
    return df, _info_from_df(df)
