# data_analysis_tools.py

import pandas as pd
from utils.util import get_connection

def data_overview(query: str = "") -> dict:
    """
    Fallback data-analysis tool:
    • Detects whether the user wants order or order-item data.
    • Returns row count, column list, and sample rows (up to 10).
    • Use when the agent cannot pick a more specific analytics tool.
    """
    conn = get_connection()
    q = query.lower()
    if "item" in q or "order_items" in q or "product" in q:
        table = "ne_order_items"
    else:
        table = "ne_order"

    # get total rows
    count = pd.read_sql_query(f"SELECT COUNT(*) AS cnt FROM {table}", conn).iloc[0]["cnt"]
    # sample
    sample = pd.read_sql_query(f"SELECT * FROM {table} LIMIT 10", conn)
    cols = list(sample.columns)
    records = sample.to_dict(orient="records")

    summary = (
        f"Table {table}: {count} rows\n"
        f"Columns: {', '.join(cols)}\n"
        f"Sample (first {len(records)} rows) below."
    )
    return {
        "summary": summary,
        "data": records,
        "chart": None
    }
