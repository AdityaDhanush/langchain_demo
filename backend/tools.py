import os
import pandas as pd
import sqlite3
from dateparser.search import search_dates
from dateparser import parse                 # you can use parse() for single dates
from mlxtend.frequent_patterns import apriori, association_rules

DB_PATH = os.getenv("DB_PATH", "/app/data/ai_ignition_616.db")
DEFAULT_BUS_ID = int(os.getenv("DEFAULT_BUS_ID", 616))

# def parse_date_range(date_text):
#     if not date_text:
#         return None, None
#     dates = dateparser.search.search_dates(date_text)
#     if not dates:
#         return None, None
#     if len(dates) >= 2:
#         return dates[0][1].date(), dates[1][1].date()
#     d = dates[0][1].date()
#     return d, d

def parse_date_range(date_text):
    if not date_text:
        return None, None
    # Use search_dates from dateparser.search
    dates = search_dates(date_text)
    if not dates:
        return None, None
    if len(dates) >= 2:
        # If two or more dates found, treat as a range
        return dates[0][1].date(), dates[1][1].date()
    # Only one date found: use it for both start and end
    d = dates[0][1].date()
    return d, d


def get_connection():
    return sqlite3.connect(DB_PATH)

def segment_customers(query: str = "") -> str:
    start, end = parse_date_range(query)
    conn = get_connection()
    df = pd.read_sql_query(
        "SELECT USER_ID, SERVER_TIMESTAMP, TOTAL FROM ne_order WHERE BUS_ASS_ID = ?",
        conn, params=(DEFAULT_BUS_ID,)
    )
    df['SERVER_TIMESTAMP'] = pd.to_datetime(df['SERVER_TIMESTAMP'])
    if start and end:
        mask = (df['SERVER_TIMESTAMP'].dt.date >= start) & (df['SERVER_TIMESTAMP'].dt.date <= end)
        df = df[mask]
    snapshot = df['SERVER_TIMESTAMP'].max()
    rfm = df.groupby('USER_ID').agg(
        recency=lambda x: (snapshot - x.max()).days,
        frequency=('SERVER_TIMESTAMP', 'count'),
        monetary=('TOTAL', 'sum')
    )
    rfm['r_score'] = pd.qcut(rfm['recency'], 5, labels=range(5, 0, -1))
    rfm['f_score'] = pd.qcut(rfm['frequency'], 5, labels=range(1, 6))
    rfm['m_score'] = pd.qcut(rfm['monetary'], 5, labels=range(1, 6))
    rfm['RFM_Score'] = rfm['r_score'].astype(str) + rfm['f_score'].astype(str) + rfm['m_score'].astype(str)
    summary = rfm['RFM_Score'].value_counts().head(5)
    text = "Top 5 RFM segments:\n"
    for seg, cnt in summary.items():
        text += f"Segment {seg}: {cnt} customers\n"
    return text

def find_bundles(query: str = "") -> str:
    start, end = parse_date_range(query)
    conn = get_connection()
    df = pd.read_sql_query(
        "SELECT ORDER_ID, SKU_CODE FROM ne_order_items WHERE BUS_ASS_ID = ? AND SKU_CODE != 'NOSKU'",
        conn, params=(DEFAULT_BUS_ID,)
    )
    df = df.dropna()
    if start and end:
        orders = pd.read_sql_query(
            "SELECT ORDER_ID, SERVER_TIMESTAMP FROM ne_order WHERE BUS_ASS_ID = ?",
            conn, params=(DEFAULT_BUS_ID,)
        )
        orders['SERVER_TIMESTAMP'] = pd.to_datetime(orders['SERVER_TIMESTAMP'])
        orders = orders[(orders['SERVER_TIMESTAMP'].dt.date >= start) & (orders['SERVER_TIMESTAMP'].dt.date <= end)]
        df = df[df['ORDER_ID'].isin(orders['ORDER_ID'])]
    basket = df.groupby(['ORDER_ID', 'SKU_CODE'])['SKU_CODE']               .count().unstack().fillna(0).astype(int)
    freq_items = apriori(basket, min_support=0.01, use_colnames=True)
    rules = association_rules(freq_items, metric="lift", min_threshold=1.2)
    rules = rules.sort_values('lift', ascending=False).head(5)
    text = "Top 5 product bundles:\n"
    for _, row in rules.iterrows():
        items = list(row['antecedents']) + list(row['consequents'])
        text += f"{items}: support={row['support']:.2f}, confidence={row['confidence']:.2f}, lift={row['lift']:.2f}\n"
    return text

def analyze_marketing(query: str = "") -> str:
    start, end = parse_date_range(query)
    conn = get_connection()
    orders = pd.read_sql_query(
        "SELECT CHANNEL_TYPE, TOTAL, SERVER_TIMESTAMP FROM ne_order WHERE BUS_ASS_ID = ?",
        conn, params=(DEFAULT_BUS_ID,)
    )
    orders['SERVER_TIMESTAMP'] = pd.to_datetime(orders['SERVER_TIMESTAMP'])
    if start and end:
        orders = orders[(orders['SERVER_TIMESTAMP'].dt.date >= start) & (orders['SERVER_TIMESTAMP'].dt.date <= end)]
    summary = orders.groupby('CHANNEL_TYPE').agg(orders=('TOTAL', 'count'), revenue=('TOTAL', 'sum'))
    text = "Marketing performance by channel:\n"
    for idx, row in summary.iterrows():
        text += f"Channel {idx}: {row.orders} orders, ${row.revenue:.2f} revenue\n"
    return text
