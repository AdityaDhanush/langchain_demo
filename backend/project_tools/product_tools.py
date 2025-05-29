import re
import pandas as pd

from utils import util



def get_top_revenue_products(query: str = "") -> dict:
    """
    Returns the top N SKUs by total revenue over a user-specified period.
    • Parses date expressions like “last 3 months” or specific months/years.
    • Filters out any SKU_CODE starting with “NOSKU”.
    • Sums up revenue per SKU and honors “top N” requests (default N=5).
    """
    # 1) Parse date range
    start, end = util.parse_date_range(query)
    today = pd.Timestamp.today().normalize().date()
    if not start or not end:
        # default: last 12 months
        end = today
        start = (pd.Timestamp(today) - pd.DateOffset(months=12)).date()

    # 2) Load item-level revenue and order timestamp in one go
    conn = util.get_connection()
    sql = """
    SELECT
      oi.SKU_CODE,
      oi.TOTAL   AS item_revenue,
      o.SERVER_TIMESTAMP
    FROM ne_order_items AS oi
    JOIN ne_order       AS o
      ON oi.ORDER_ID = o.ORDER_ID
    WHERE oi.BUS_ASS_ID = ?
      AND oi.SKU_CODE NOT LIKE 'NOSKU%'
    """
    df = pd.read_sql_query(sql, conn, params=(util.DEFAULT_BUS_ID,))
    if df.empty:
        return {"summary": "No product data found.", "data": {}, "chart": None}

    # 3) Filter by the parsed date range
    df["SERVER_TIMESTAMP"] = pd.to_datetime(df["SERVER_TIMESTAMP"])
    mask = (df["SERVER_TIMESTAMP"].dt.date >= start) & (df["SERVER_TIMESTAMP"].dt.date <= end)
    df = df.loc[mask]
    if df.empty:
        return {
            "summary": f"No sales for any SKU between {start} and {end}.",
            "data": {},
            "chart": None
        }

    # 4) Aggregate revenue by SKU
    revenue_by_sku = (
        df
        .groupby("SKU_CODE")["item_revenue"]
        .sum()
        .sort_values(ascending=False)
    )

    # 5) Determine N (e.g. “top 10”)
    m = re.search(r"\btop\s+(\d+)", query.lower())
    top_n = int(m.group(1)) if m else 5
    top_series = revenue_by_sku.head(top_n).round(2)
    top_dict = top_series.to_dict()

    # 6) Build summary and chart payload
    summary_lines = [f"- {sku}: ${rev}" for sku, rev in top_dict.items()]
    summary = (
        f"Top {top_n} SKUs by revenue from {start} to {end}:\n"
        + "\n".join(summary_lines)
    )

    return {
        "summary": summary,
        "data": top_dict,
        "chart": { "type": "bar", "data": top_dict }
    }


# def get_pareto_products(query: str = "") -> dict:
#     """
#     Identifies which SKUs (or categories) account for 80% of revenue.
#     Query examples:
#       - "how many SKUs produce 80% of revenue"
#       - "80% of revenue from which categories"
#     """
#     by_cat = 'category' in query.lower()
#     # choose key column
#     keycol = 'CATEGORY' if by_cat else 'SKU_CODE'
#     conn = get_connection()
#     df = pd.read_sql_query(
#         f"SELECT {keycol}, TOTAL FROM ne_order_items oi "
#         "JOIN ne_order o ON oi.ORDER_ID=o.ORDER_ID "
#         "WHERE o.BUS_ASS_ID=? AND SKU_CODE NOT LIKE 'NOSKU%'",
#         conn, params=(DEFAULT_BUS_ID,)
#     )
#     group = df.groupby(keycol)['TOTAL'].sum().sort_values(ascending=False)
#     cumulative = group.cumsum()
#     cutoff = 0.8 * group.sum()
#     pareto = cumulative[cumulative <= cutoff]
#     count = pareto.size
#     total = len(group)
#     perc = (count/total)*100
#     text = (f"{count} {'categories' if by_cat else 'SKUs'} "
#             f"account for 80% of revenue ({count}/{total}, {perc:.1f}%).")
#     return {
#       'summary': text,
#       'data': {k: group[k] for k in pareto.index},
#       'chart': {'type':'pie','data':{k: group[k] for k in pareto.index}}
#     }


def get_pareto_products(query: str = "") -> dict:
    """
    Identifies which SKUs (or categories) account for 80% of total revenue over a period.
    • Parses date ranges like "last 6 months" or specific months/years.
    • Filters out any SKU_CODE starting with "NOSKU".
    • Groups revenue by SKU_CODE or CATEGORY if 'category' appears in the query.
    • Computes the cumulative revenue share and returns the minimal list whose sum ≥80%.
    """
    # 1) Date range
    start, end = util.parse_date_range(query)
    today = pd.Timestamp.today().normalize().date()
    if not start or not end:
        end = today
        start = (pd.Timestamp(today) - pd.DateOffset(months=12)).date()

    # 2) Load data with explicit alias for revenue
    sql = """
    SELECT
      oi.SKU_CODE,
      oi.CATEGORY,
      oi.TOTAL        AS item_revenue,
      o.SERVER_TIMESTAMP
    FROM ne_order_items AS oi
    JOIN ne_order       AS o
      ON oi.ORDER_ID = o.ORDER_ID
    WHERE oi.BUS_ASS_ID = ?
      AND oi.SKU_CODE NOT LIKE 'NOSKU%'
    """
    conn = util.get_connection()
    df = pd.read_sql_query(sql, conn, params=(util.DEFAULT_BUS_ID,))
    if df.empty:
        return {"summary": "No product data found.", "data": {}, "chart": None}

    # 3) Filter by date
    df["SERVER_TIMESTAMP"] = pd.to_datetime(df["SERVER_TIMESTAMP"])
    mask = (df["SERVER_TIMESTAMP"].dt.date >= start) & (df["SERVER_TIMESTAMP"].dt.date <= end)
    df = df.loc[mask]
    if df.empty:
        return {
            "summary": f"No sales between {start} and {end}.",
            "data": {},
            "chart": None
        }

    # 4) Decide grouping
    by_cat = "category" in query.lower()
    group_col = "CATEGORY" if by_cat else "SKU_CODE"

    # 5) Aggregate revenue
    agg = df.groupby(group_col)["item_revenue"].sum().sort_values(ascending=False)

    # 6) Compute Pareto cutoff
    total_rev = agg.sum()
    cumulative = agg.cumsum()
    pareto_mask = cumulative <= 0.8 * total_rev
    pareto_list = agg[pareto_mask]

    # 7) Build output
    count = len(pareto_list)
    total_items = len(agg)
    pct = (count / total_items) * 100

    entity = "categories" if by_cat else "SKUs"
    summary = (
        f"{count} {entity} account for 80% of revenue "
        f"({count} of {total_items}, {pct:.1f}%)."
    )

    data = pareto_list.round(2).to_dict()
    chart = {"type": "pie", "data": data}

    return {"summary": summary, "data": data, "chart": chart}


def get_yoy_revenue_change(query: str = "") -> dict:
    """
    Calculates % change in top SKUs’ revenue from last year to this year.
    Query examples:
      - "YoY change in top 5 products' revenue"
    """
    n = int(re.search(r'top\s+(\d+)', query.lower() or 'top 5').group(1)) if re.search(r'top\s+(\d+)', query.lower()) else 5
    conn = util.get_connection()
    orders = pd.read_sql_query(
        "SELECT oi.SKU_CODE, oi.TOTAL, o.SERVER_TIMESTAMP FROM ne_order_items oi "
        "JOIN ne_order o ON oi.ORDER_ID=o.ORDER_ID "
        "WHERE o.BUS_ASS_ID=? AND SKU_CODE NOT LIKE 'NOSKU%'",
        conn, params=(util.DEFAULT_BUS_ID,)
    )
    orders['date'] = pd.to_datetime(orders['SERVER_TIMESTAMP']).dt.date
    orders['year'] = pd.to_datetime(orders['SERVER_TIMESTAMP']).dt.year
    this, last = orders[orders['year']==orders['year'].max()], orders[orders['year']==orders['year'].max()-1]
    sum_this = this.groupby('SKU_CODE')['TOTAL'].sum()
    sum_last = last.groupby('SKU_CODE')['TOTAL'].sum()
    top_skus = sum_this.sort_values(ascending=False).head(n).index
    change = {}
    for sku in top_skus:
        t = sum_this.get(sku,0)
        l = sum_last.get(sku,0)
        change[sku] = ((t-l)/l *100) if l>0 else None
    text = "YoY % change in revenue for top SKUs:\n" + \
           "\n".join(f"- {sku}: {change[sku]:.2f}%" for sku in change)
    return {'summary':text,'data':change,'chart':{'type':'bar','data':{sku:change[sku] for sku in change}}}


def get_mom_basket_size_change(query: str = "") -> dict:
    """
    Computes the change in average basket size (items per order) month-over-month.
    Query examples:
      - "change in average basket size MoM"
    """
    conn = util.get_connection()
    df = pd.read_sql_query(
        "SELECT ORDER_ID, ITEMS_COUNT, SERVER_TIMESTAMP FROM ne_order "
        "WHERE BUS_ASS_ID=?", conn, params=(util.DEFAULT_BUS_ID,)
    )
    df['month'] = pd.to_datetime(df['SERVER_TIMESTAMP']).dt.to_period('M').astype(str)
    avg_size = df.groupby('month')['ITEMS_COUNT'].mean()
    pct_change = avg_size.pct_change().fillna(0)*100
    data = pct_change.round(2).to_dict()
    text = "MoM % change in average basket size:\n" + \
           "\n".join(f"- {m}: {v:.2f}%" for m,v in data.items())
    return {'summary': text, 'data': data, 'chart': {'type':'line','data':data}}



def product_overview(query: str = "") -> dict:
    """
    Catch-all product analytics:
    - Filters out any SKU_CODE starting with 'NOSKU'
    - Parses date ranges ("last 3 months", "April 2025", etc.)
    - Groups by SKU_CODE+PRODUCT_NAME (default) or by CATEGORY if 'category' is in the query
    - Aggregates total_revenue and total_units sold
    - Recognizes 'top N' or 'bottom N' to limit results (default N=10)
    - Sorts by revenue (default) or by quantity if 'quantity' or 'units' in the query
    """
    # 1) Date range
    start, end = util.parse_date_range(query)
    today = pd.Timestamp.today().date()
    if not start or not end:
        # default to last 12 months
        end = today
        start = (pd.Timestamp(today) - pd.DateOffset(months=12)).date()

    # 2) Load raw data
    conn = util.get_connection()
    df = pd.read_sql_query(
        """
        SELECT oi.SKU_CODE, oi.PRODUCT_NAME, oi.CATEGORY,
               oi.QUANTITY AS total_units, oi.TOTAL AS total_revenue,
               o.SERVER_TIMESTAMP
        FROM ne_order_items oi
        JOIN ne_order o ON oi.ORDER_ID = o.ORDER_ID
        WHERE o.BUS_ASS_ID = ?
          AND oi.SKU_CODE NOT LIKE 'NOSKU%'
        """,
        conn, params=(util.DEFAULT_BUS_ID,)
    )
    df['SERVER_TIMESTAMP'] = pd.to_datetime(df['SERVER_TIMESTAMP'])
    # 3) Filter by date
    mask = (df['SERVER_TIMESTAMP'].dt.date >= start) & \
           (df['SERVER_TIMESTAMP'].dt.date <= end)
    df = df.loc[mask]
    if df.empty:
        return {'summary': 'No product data in that period.', 'data': {}, 'chart': None}

    # 4) Choose grouping
    by_category = 'category' in query.lower()
    if by_category:
        group_cols = ['CATEGORY']
        label_col = 'CATEGORY'
    else:
        group_cols = ['SKU_CODE','PRODUCT_NAME']
        label_col = 'SKU_CODE'

    # 5) Aggregate
    agg = df.groupby(group_cols).agg(
        total_units  = ('total_units',  'sum'),
        total_revenue= ('total_revenue','sum')
    ).reset_index()

    # 6) Determine sort key
    sort_by_qty = bool(re.search(r'\b(quantity|units)\b', query.lower()))
    sort_col   = 'total_units' if sort_by_qty else 'total_revenue'

    # 7) Determine N
    m = re.search(r'\b(top|bottom)\s+(\d+)', query.lower())
    if m:
        direction, n = m.group(1), int(m.group(2))
    else:
        direction, n = 'top', 10

    # 8) Sort & slice
    ascending = (direction == 'bottom')
    agg = agg.sort_values(sort_col, ascending=ascending)
    subset = agg.head(n)

    # 9) Build output dict
    data = {}
    for _, row in subset.iterrows():
        if by_category:
            key = row['CATEGORY']
        else:
            key = f"{row['SKU_CODE']} ({row['PRODUCT_NAME']})"
        data[key] = {
            'revenue': round(row['total_revenue'], 2),
            'units':   int(row['total_units'])
        }

    # 10) Build text summary
    summary_lines = []
    for k, v in data.items():
        summary_lines.append(f"- {k}: ${v['revenue']} revenue, {v['units']} units")
    summary = (
        f"{direction.title()} {n} "
        f"{'categories' if by_category else 'products'} "
        f"by {'quantity' if sort_by_qty else 'revenue'} "
        f"from {start} to {end}:\n" +
        "\n".join(summary_lines)
    )

    # 11) Chart data: separate series for revenue and units
    chart = {
        'type': 'multi-bar',
        'data': {
            'revenue': {k: v['revenue'] for k, v in data.items()},
            'units':   {k: v['units']   for k, v in data.items()}
        }
    }

    return {'summary': summary, 'data': data, 'chart': chart}

