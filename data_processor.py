import pandas as pd
import numpy as np
import sqlite3
import io


def load_data(uploaded_file):
    """Load CSV or Excel file into a DataFrame."""
    try:
        filename = uploaded_file.name.lower()
        if filename.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        elif filename.endswith((".xlsx", ".xls")):
            df = pd.read_excel(uploaded_file)
        else:
            return None, "Unsupported file type. Please upload CSV or Excel."

        # Basic cleaning
        df.columns = [str(c).strip().replace(" ", "_").lower() for c in df.columns]
        return df, "Success"
    except Exception as e:
        return None, f"Error loading file: {str(e)}"


def get_summary_stats(df):
    """Return basic summary statistics."""
    return {
        "rows": len(df),
        "cols": len(df.columns),
        "numeric_cols": len(df.select_dtypes(include=np.number).columns),
        "missing": int(df.isnull().sum().sum()),
    }


def run_sql_query(df, query):
    """Run a SQL query on the dataframe using SQLite."""
    try:
        conn = sqlite3.connect(":memory:")
        df.to_sql("data", conn, index=False, if_exists="replace")
        result = pd.read_sql_query(query, conn)
        conn.close()
        return result, None
    except Exception as e:
        return None, str(e)


def get_df_context(df, max_rows=5):
    """Generate a text summary of the dataframe for AI context."""
    context = f"""
Dataset Overview:
- Shape: {df.shape[0]} rows × {df.shape[1]} columns
- Columns: {', '.join(df.columns.tolist())}
- Data Types: {df.dtypes.to_dict()}
- Missing Values: {df.isnull().sum().to_dict()}

Sample Data (first {max_rows} rows):
{df.head(max_rows).to_string()}

Descriptive Statistics:
{df.describe().to_string()}
"""
    return context
