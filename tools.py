# tools.py
import json
import pandas as pd
import plotly.express as px
import duckdb
from typing import Optional, Any
from pydantic.v1 import BaseModel, Field # Using v1 for broader compatibility

# LangChain's tool decorator
from langchain_core.tools import tool

# Project-local modules
from utils import create_duckdb_conn_and_register, fig_to_json, safe_parse_filter
import data_cleaning as dc

# --- MODULE-LEVEL STATE ---
# These are shared across all tool calls in a session.
_ACTIVE_DF: Optional[pd.DataFrame] = None
_DUCK_CONN: Any = None

# --- HELPER FUNCTIONS ---
def _require_df():
    """Check if a DataFrame is loaded."""
    if _ACTIVE_DF is None:
        return "No active DataFrame. Please upload a dataset first."
    return None

def _df_preview(df: pd.DataFrame, max_rows: int = 20) -> str:
    """Return a markdown-formatted string preview of a DataFrame."""
    if df.empty:
        return "The result is an empty dataset."
    try:
        return df.head(max_rows).to_markdown(index=False)
    except Exception:
        return df.head(max_rows).to_string(index=False)

def _update_df_and_get_msg(result_tuple: tuple[pd.DataFrame, str]) -> str:
    """Helper to update the global DF and re-register the DuckDB view."""
    global _ACTIVE_DF, _DUCK_CONN
    new_df, msg = result_tuple
    _ACTIVE_DF = new_df
    if _DUCK_CONN:
        _DUCK_CONN.unregister("data_table")
        _DUCK_CONN.register("data_table", _ACTIVE_DF)
    return msg

def set_active_df(df: pd.DataFrame) -> str:
    """Sets the active DataFrame for all tools to use."""
    global _ACTIVE_DF, _DUCK_CONN
    _ACTIVE_DF = df.copy()
    _DUCK_CONN = create_duckdb_conn_and_register(_ACTIVE_DF, "data_table")
    return f"Active DataFrame set ({len(df)} rows × {len(df.columns)} cols)."

# ----------------- DATA EXPLORATION TOOLS -----------------

@tool
def preview_head(n: int = 5) -> str:
    """Preview the first n rows of the dataset."""
    err = _require_df()
    if err: return err
    return _df_preview(_ACTIVE_DF, n)

@tool
def dataset_summary(_: str = "") -> str:
    """Summarize the dataset, including shape, column data types, and descriptive statistics for numeric columns."""
    err = _require_df()
    if err: return err
    summary = {
        "shape": _ACTIVE_DF.shape,
        "dtypes": _ACTIVE_DF.dtypes.astype(str).to_dict(),
        "numeric_stats": _ACTIVE_DF.describe(include="number").to_dict()
    }
    return json.dumps(summary, indent=2, default=str)

class GetUniqueInput(BaseModel):
    column: str = Field(description="The name of the column to get unique values from.")

@tool(args_schema=GetUniqueInput)
def get_unique_values(column: str) -> str:
    """Get the unique values from a single specified column."""
    err = _require_df()
    if err: return err
    if column not in _ACTIVE_DF.columns:
        return f"Error: Column '{column}' not found."
    unique_list = _ACTIVE_DF[column].dropna().unique().tolist()
    return json.dumps({"column": column, "unique_values": unique_list}, default=str)

# ----------------- FILTERING & SQL TOOLS -----------------

class FilterRowsInput(BaseModel):
    expression: str = Field(description="A boolean expression to filter rows. Example: 'TotalSaleDollars > 500 and Region == \"West\"'")

@tool(args_schema=FilterRowsInput)
def filter_rows(expression: str) -> str:
    """Filter dataset rows based on a pandas query-like expression."""
    err = _require_df()
    if err: return err
    try:
        mask = safe_parse_filter(expression, _ACTIVE_DF)
        return _df_preview(_ACTIVE_DF[mask])
    except Exception as e:
        return f"Filter error: {e}"

class RunSQLInput(BaseModel):
    query: str = Field(description="A SQL SELECT or WITH query to run against the dataset (named 'data_table').")

@tool(args_schema=RunSQLInput)
def run_sql_query(query: str) -> str:
    """Run an SQL SELECT or WITH query on the dataset. For aggregations, grouping, and complex filtering."""
    err = _require_df()
    if err: return err
    if ";" in query or not query.strip().lower().startswith(("select", "with")):
        return "Error: Only SELECT/WITH queries are allowed."
    try:
        df_result = _DUCK_CONN.execute(query).fetchdf()
        return _df_preview(df_result, max_rows=100)
    except Exception as e:
        return f"SQL execution error: {e}"

# ----------------- VISUALIZATION TOOLS -----------------

class PlotHistogramInput(BaseModel):
    column: str = Field(description="The name of the numeric column to plot.")
    bins: int = Field(30, description="The number of bins for the histogram.")

@tool(args_schema=PlotHistogramInput)
def plot_histogram(column: str, bins: int = 30) -> str:
    """Create and return a histogram for a numeric column."""
    err = _require_df()
    if err: return err
    fig = px.histogram(_ACTIVE_DF, x=column, nbins=bins, title=f"Histogram of {column}")
    return fig_to_json(fig)

class PlotScatterInput(BaseModel):
    x: str = Field(description="The column name for the x-axis.")
    y: str = Field(description="The column name for the y-axis.")
    color: Optional[str] = Field(None, description="Optional column to use for coloring points.")

@tool(args_schema=PlotScatterInput)
def plot_scatter(x: str, y: str, color: Optional[str] = None) -> str:
    """Create and return a scatter plot of two variables."""
    err = _require_df()
    if err: return err
    fig = px.scatter(_ACTIVE_DF, x=x, y=y, color=color, title=f"Scatter Plot: {x} vs {y}")
    return fig_to_json(fig)

class PlotBoxInput(BaseModel):
    x: Optional[str] = Field(None, description="Categorical column for the x-axis.")
    y: str = Field(description="Numeric column for the y-axis.")

@tool(args_schema=PlotBoxInput)
def plot_boxplot(y: str, x: Optional[str] = None) -> str:
    """Create and return a box plot to show distribution."""
    err = _require_df()
    if err: return err
    fig = px.box(_ACTIVE_DF, x=x, y=y, title=f"Box Plot of {y}" + (f" by {x}" if x else ""))
    return fig_to_json(fig)

class PlotBarInput(BaseModel):
    x: str = Field(description="The categorical column for the x-axis (bars).")
    y: str = Field(description="The numeric column for the y-axis (height of bars).")
    color: Optional[str] = Field(None, description="Optional column to color the bars.")

@tool(args_schema=PlotBarInput)
def plot_bar(x: str, y: str, color: Optional[str] = None) -> str:
    """Create and return a bar chart."""
    err = _require_df()
    if err: return err
    # Often, bar charts are used for aggregated data. A simple SQL query can prepare this.
    query = f'SELECT "{x}", SUM("{y}") as "{y}" FROM data_table GROUP BY "{x}" ORDER BY "{y}" DESC LIMIT 25'
    try:
        plot_df = _DUCK_CONN.execute(query).fetchdf()
        fig = px.bar(plot_df, x=x, y=y, color=color, title=f"Bar Chart of {y} by {x}")
    except Exception: # Fallback for non-numeric y, etc.
         fig = px.bar(_ACTIVE_DF, x=x, y=y, color=color, title=f"Bar Chart of {y} by {x}")
    return fig_to_json(fig)

class PlotLineInput(BaseModel):
    x: str = Field(description="The column for the x-axis, often a date or time.")
    y: str = Field(description="The numeric column for the y-axis.")
    color: Optional[str] = Field(None, description="Optional column to create separate lines.")

@tool(args_schema=PlotLineInput)
def plot_line(x: str, y: str, color: Optional[str] = None) -> str:
    """Create and return a line chart, useful for time-series data."""
    err = _require_df()
    if err: return err
    fig = px.line(_ACTIVE_DF, x=x, y=y, color=color, title=f"Line Chart: {y} over {x}")
    return fig_to_json(fig)

class PlotPieInput(BaseModel):
    names: str = Field(description="The column with categories for the pie slices.")
    values: str = Field(description="The numeric column that determines the size of the slices.")

@tool(args_schema=PlotPieInput)
def plot_pie(names: str, values: str) -> str:
    """Create and return a pie chart."""
    err = _require_df()
    if err: return err
    fig = px.pie(_ACTIVE_DF, names=names, values=values, title=f"Pie Chart: {values} by {names}")
    return fig_to_json(fig)

@tool
def plot_heatmap(_: str = "") -> str:
    """Create and return a correlation heatmap of all numeric columns."""
    err = _require_df()
    if err: return err
    numeric_df = _ACTIVE_DF.select_dtypes(include="number")
    if numeric_df.shape[1] < 2:
        return "Error: A heatmap requires at least two numeric columns."
    corr = numeric_df.corr()
    fig = px.imshow(corr, text_auto=True, title="Correlation Heatmap")
    return fig_to_json(fig)

class PlotStackedBarInput(BaseModel):
    x: str = Field(description="The categorical column for the x-axis.")
    y: str = Field(description="The numeric column for the y-axis.")
    color: str = Field(description="The column used to create the stacks in each bar.")

@tool(args_schema=PlotStackedBarInput)
def plot_stacked_bar(x: str, y: str, color: str) -> str:
    """Create and return a stacked bar chart."""
    err = _require_df()
    if err: return err
    fig = px.bar(_ACTIVE_DF, x=x, y=y, color=color, title=f"Stacked Bar: {y} by {x}, stacked by {color}", barmode="stack")
    return fig_to_json(fig)

class PlotAreaInput(BaseModel):
    x: str = Field(description="The column for the x-axis, typically time.")
    y: str = Field(description="The numeric column for the y-axis.")
    color: Optional[str] = Field(None, description="Optional column to create separate areas.")

@tool(args_schema=PlotAreaInput)
def plot_area(x: str, y: str, color: Optional[str] = None) -> str:
    """Create and return an area chart."""
    err = _require_df()
    if err: return err
    fig = px.area(_ACTIVE_DF, x=x, y=y, color=color, title=f"Area Chart: {y} over {x}")
    return fig_to_json(fig)

# ----------------- DATA CLEANING TOOLS -----------------

class DropDuplicatesInput(BaseModel):
    keep: str = Field("first", description="Which duplicate to keep. Can be 'first' or 'last'.")

@tool(args_schema=DropDuplicatesInput)
def drop_duplicates(keep: str = "first") -> str:
    """Remove duplicate rows from the dataset."""
    err = _require_df()
    if err: return err
    return _update_df_and_get_msg(dc.drop_duplicates(_ACTIVE_DF, keep=keep))

class FillMissingInput(BaseModel):
    column: str = Field(description="The column with missing values to fill.")
    strategy: str = Field(description="The method to use: 'mean', 'median', 'mode', or 'constant'.")
    value: Optional[Any] = Field(None, description="The constant value to use if strategy is 'constant'.")

@tool(args_schema=FillMissingInput)
def fill_missing_values(column: str, strategy: str, value: Optional[Any] = None) -> str:
    """Fill missing (NaN) values in a column using a specified strategy."""
    err = _require_df()
    if err: return err
    return _update_df_and_get_msg(dc.fill_missing(_ACTIVE_DF, column=column, strategy=strategy, value=value))

@tool
def convert_column_types(_: str = "") -> str:
    """Automatically convert columns to the best possible data types."""
    err = _require_df()
    if err: return err
    return _update_df_and_get_msg(dc.convert_best_dtypes(_ACTIVE_DF))

class TrimWhitespaceInput(BaseModel):
    lower_case: bool = Field(False, description="Whether to also convert string columns to lower case.")

@tool(args_schema=TrimWhitespaceInput)
def trim_whitespace(lower_case: bool = False) -> str:
    """Trim leading/trailing whitespace from all string columns."""
    err = _require_df()
    if err: return err
    return _update_df_and_get_msg(dc.trim_string_columns(_ACTIVE_DF, lower=lower_case))

class DetectOutliersInput(BaseModel):
    column: str = Field(description="The numeric column to check for outliers.")

@tool(args_schema=DetectOutliersInput)
def detect_outliers(column: str) -> str:
    """Detect outliers in a numeric column using the IQR method and return a report."""
    err = _require_df()
    if err: return err
    report = dc.detect_outliers_iqr(_ACTIVE_DF, column=column)
    return json.dumps(report, indent=2)