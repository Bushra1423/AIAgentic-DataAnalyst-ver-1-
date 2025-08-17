# utils.py
"""
Utility helpers: file loading, Plotly JSON helpers, and a safe filter parser.
"""

from typing import Any, Dict, List, Tuple, Optional
import io
import re
import json
import duckdb
import pandas as pd
import plotly.graph_objects as go


# ---------- Data Loading ----------

def load_data(file_like) -> pd.DataFrame:
    """
    Load CSV or Excel from a file-like object (Streamlit uploader or path-like).
    """
    name = getattr(file_like, "name", "")
    if name.lower().endswith((".xls", ".xlsx")):
        return pd.read_excel(file_like)
    # default CSV
    return pd.read_csv(file_like)


# ---------- Plotly JSON helpers ----------

def fig_to_json(fig: go.Figure) -> str:
    return fig.to_json()


def json_to_fig(s: str) -> go.Figure:
    import plotly.io as pio
    return pio.from_json(s)


# ---------- DuckDB helper ----------

def create_duckdb_conn_and_register(df: pd.DataFrame, table_name: str = "data_table"):
    """
    Create a DuckDB connection in-memory and register the DataFrame as a view.
    """
    con = duckdb.connect(database=":memory:")
    con.register(table_name, df)  # register as view
    return con


# ---------- Safe Filter Parsing ----------

_TOKEN_RE = re.compile(r"""
    (?P<op_and>\band\b)|
    (?P<op_or>\bor\b)|
    (?P<cmp>==|>=|<=|>|<|!=)|
    (?P<in>\bin\b)|
    (?P<lbr>\[)|
    (?P<rbr>\])|
    (?P<num>-?\d+(?:\.\d+)?)|
    (?P<ident>[A-Za-z_]\w*)|
    (?P<comma>,)|
    (?P<ws>\s+)
""", re.VERBOSE | re.IGNORECASE)


def _tokenize(expr: str) -> List[Tuple[str, str]]:
    tokens = []
    for m in _TOKEN_RE.finditer(expr):
        kind = m.lastgroup
        val = m.group(0)
        if kind == "ws":
            continue
        tokens.append((kind, val))
    return tokens


def safe_parse_filter(expr: str, df: pd.DataFrame) -> pd.Series:
    """
    Parse a very small boolean filter language:
      - <col> <cmp> <value>
      - <col> in [a, b, c]
      - combine with 'and' / 'or'
    Values that match column dtype are coerced. Strings need not be quoted.
    """
    tokens = _tokenize(expr)
    i = 0

    def parse_value(col: Optional[str] = None):
        nonlocal i
        if i >= len(tokens):
            raise ValueError("Unexpected end of expression.")
        kind, val = tokens[i]
        # List: [a, b, c]
        if kind == "lbr":
            i += 1
            vals = []
            while i < len(tokens) and tokens[i][0] != "rbr":
                if tokens[i][0] == "comma":
                    i += 1
                    continue
                if tokens[i][0] in ("num", "ident"):
                    vals.append(tokens[i][1])
                    i += 1
                else:
                    raise ValueError("Invalid token in list.")
            if i >= len(tokens) or tokens[i][0] != "rbr":
                raise ValueError("Missing closing bracket ']'.")
            i += 1
            return vals
        # Scalar
        if kind in ("num", "ident"):
            i += 1
            return val
        raise ValueError(f"Unexpected token '{val}'.")

    def coerce_to_col_dtype(col: str, v):
        s = df[col]
        if pd.api.types.is_numeric_dtype(s):
            try:
                return float(v)
            except Exception:
                # allow strings in numeric col => will be NaN comparison (False)
                return v
        else:
            # treat everything as string for object/string columns
            return str(v)

    def parse_atom():
        nonlocal i
        if i >= len(tokens):
            raise ValueError("Empty expression.")
        # <ident> ...
        if tokens[i][0] == "ident":
            col = tokens[i][1]
            i += 1
            if col not in df.columns:
                raise ValueError(f"Unknown column '{col}'.")
            if i < len(tokens) and tokens[i][0] == "in":
                i += 1
                vals = parse_value(col)
                # List membership
                vals = [coerce_to_col_dtype(col, v) for v in vals]
                return df[col].isin(vals)
            elif i < len(tokens) and tokens[i][0] == "cmp":
                op = tokens[i][1]
                i += 1
                val = parse_value(col)
                val = coerce_to_col_dtype(col, val)
                series = df[col]
                if op == "==":
                    return series == val
                if op == "!=":
                    return series != val
                if op == ">":
                    return series > val
                if op == "<":
                    return series < val
                if op == ">=":
                    return series >= val
                if op == "<=":
                    return series <= val
                raise ValueError(f"Unknown comparator '{op}'.")
            else:
                raise ValueError("Expected 'in' or comparator after column.")
        else:
            raise ValueError("Expected a column name at start of condition.")

    def parse_expr():
        nonlocal i
        result = parse_atom()
        while i < len(tokens):
            if tokens[i][0] == "op_and":
                i += 1
                rhs = parse_atom()
                result = result & rhs
            elif tokens[i][0] == "op_or":
                i += 1
                rhs = parse_atom()
                result = result | rhs
            else:
                break
        return result

    if not expr or not expr.strip():
        raise ValueError("Empty filter expression.")
    return parse_expr()
