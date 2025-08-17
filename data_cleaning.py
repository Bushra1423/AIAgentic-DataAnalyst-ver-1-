# data_cleaning.py
"""
Data cleaning helpers that operate on a pandas DataFrame and return
either a new DataFrame + message or a JSON-friendly report.
"""

from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import pandas as pd


def missing_value_report(df: pd.DataFrame) -> Dict[str, Any]:
    """Return a JSON-friendly report of missing values per column."""
    rep = df.isna().sum().to_dict()
    total = int(df.isna().sum().sum())
    rows, cols = df.shape
    return {
        "rows": rows,
        "cols": cols,
        "total_missing_cells": total,
        "missing_by_column": rep,
    }


def drop_duplicates(df: pd.DataFrame, keep: str = "first") -> Tuple[pd.DataFrame, str]:
    if keep not in {"first", "last"}:
        keep = "first"
    before = len(df)
    new_df = df.drop_duplicates(keep=keep).reset_index(drop=True)
    after = len(new_df)
    return new_df, f"Duplicates dropped. Rows: {before} -> {after} (kept='{keep}')."


def fill_missing(
    df: pd.DataFrame,
    column: str,
    strategy: str = "mean",
    value: Optional[Any] = None,
) -> Tuple[pd.DataFrame, str]:
    """
    Fill missing values in a column using strategy:
    - mean, median, mode (numeric/string OK for mode)
    - constant (requires value)
    """
    if column not in df.columns:
        return df, f"Column '{column}' not found."

    s = df[column]
    strat = strategy.lower()
    if strat == "mean":
        if not np.issubdtype(s.dropna().dtype, np.number):
            return df, f"'mean' only for numeric columns. '{column}' is {s.dtype}."
        fill_val = float(s.mean())
    elif strat == "median":
        if not np.issubdtype(s.dropna().dtype, np.number):
            return df, f"'median' only for numeric columns. '{column}' is {s.dtype}."
        fill_val = float(s.median())
    elif strat == "mode":
        m = s.mode(dropna=True)
        fill_val = None if m.empty else m.iloc[0]
    elif strat == "constant":
        if value is None:
            return df, "Provide value=... for strategy='constant'."
        fill_val = value
    else:
        return df, f"Unknown strategy '{strategy}'. Use mean|median|mode|constant."

    new_df = df.copy()
    new_df[column] = new_df[column].fillna(fill_val)
    return new_df, f"Filled missing in '{column}' using '{strat}' (value={fill_val})."


def convert_best_dtypes(df: pd.DataFrame) -> Tuple[pd.DataFrame, str]:
    new_df = df.convert_dtypes()
    return new_df, "Converted columns to best possible dtypes (pandas convert_dtypes)."


def trim_string_columns(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None,
    lower: bool = False,
) -> Tuple[pd.DataFrame, str]:
    """Trim whitespace for string columns; optionally lower-case."""
    new_df = df.copy()
    object_cols = [c for c in new_df.columns if new_df[c].dtype == "string" or new_df[c].dtype == "object"]
    target_cols = columns if columns else object_cols
    applied = []
    for col in target_cols:
        if col not in new_df.columns:
            continue
        if new_df[col].dtype == "object" or new_df[col].dtype == "string":
            new_df[col] = new_df[col].astype("string").str.strip()
            if lower:
                new_df[col] = new_df[col].str.lower()
            applied.append(col)
    if not applied:
        return df, "No string-like columns to trim."
    return new_df, f"Trimmed whitespace{' and lower-cased' if lower else ''} for columns: {', '.join(applied)}."


def detect_outliers_iqr(df: pd.DataFrame, column: str) -> Dict[str, Any]:
    """Return IQR-based outlier bounds and sample indices."""
    if column not in df.columns:
        return {"error": f"Column '{column}' not found."}
    s = pd.to_numeric(df[column], errors="coerce").dropna()
    if s.empty:
        return {"error": f"Column '{column}' has no numeric values."}

    q1 = float(s.quantile(0.25))
    q3 = float(s.quantile(0.75))
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    mask = (df[column] < lower) | (df[column] > upper)
    idx = df.index[mask].tolist()
    sample = idx[:25]

    return {
        "column": column,
        "q1": q1,
        "q3": q3,
        "iqr": float(iqr),
        "lower_bound": float(lower),
        "upper_bound": float(upper),
        "num_outliers": int(len(idx)),
        "sample_indices": sample,
    }
