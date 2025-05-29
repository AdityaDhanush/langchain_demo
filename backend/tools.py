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

# def segment_customers(query: str = "") -> str:
#     start, end = parse_date_range(query)
#     conn = get_connection()
#     df = pd.read_sql_query(
#         "SELECT USER_ID, SERVER_TIMESTAMP, TOTAL FROM ne_order WHERE BUS_ASS_ID = ?",
#         conn, params=(DEFAULT_BUS_ID,)
#     )
#     df['SERVER_TIMESTAMP'] = pd.to_datetime(df['SERVER_TIMESTAMP'])
#     if start and end:
#         mask = (df['SERVER_TIMESTAMP'].dt.date >= start) & (df['SERVER_TIMESTAMP'].dt.date <= end)
#         df = df[mask]
#     snapshot = df['SERVER_TIMESTAMP'].max()
#     # rfm = df.groupby('USER_ID').agg(
#     #     recency=lambda x: (snapshot - x.max()).days,
#     #     frequency=('SERVER_TIMESTAMP', 'count'),
#     #     monetary=('TOTAL', 'sum')
#     # )
#     rfm = df.groupby('USER_ID').agg(
#         recency=('SERVER_TIMESTAMP', lambda x: (snapshot - x.max()).days),
#         frequency=('SERVER_TIMESTAMP', 'count'),
#         monetary=('TOTAL', 'sum')
#     )
#     rfm['r_score'] = pd.qcut(rfm['recency'], 5, labels=range(5, 0, -1))
#     rfm['f_score'] = pd.qcut(rfm['frequency'], 5, labels=range(1, 6))
#     rfm['m_score'] = pd.qcut(rfm['monetary'], 5, labels=range(1, 6))
#     rfm['RFM_Score'] = rfm['r_score'].astype(str) + rfm['f_score'].astype(str) + rfm['m_score'].astype(str)
#     summary = rfm['RFM_Score'].value_counts().head(5)
#     text = "Top 5 RFM segments:\n"
#     for seg, cnt in summary.items():
#         text += f"Segment {seg}: {cnt} customers\n"
#     return text

def safe_qcut(series: pd.Series, q: int, ascending: bool = True) -> pd.Series:
    """
    Bucket `series` into up to `q` quantiles (dropping duplicate edges),
    then return integer codes 1..n_bins. If ascending=False, invert codes.
    """
    # If constant series, everything gets bin 1
    if series.nunique() < 2:
        return pd.Series(1, index=series.index)

    # Let pandas determine the bins, dropping duplicates
    cats = pd.qcut(series, q, duplicates="drop")

    # Extract integer codes (0..n_bins-1) then shift to 1..n_bins
    codes = cats.cat.codes + 1  # <— use .cat.codes, not .codes

    if not ascending:
        max_code = codes.max()
        codes = (max_code + 1) - codes

    # Ensure same index as input
    codes.index = series.index
    return codes


# def segment_customers(query: str = "") -> str:
#     start, end = parse_date_range(query)
#     conn = get_connection()
#     df = pd.read_sql_query(
#         "SELECT USER_ID, SERVER_TIMESTAMP, TOTAL FROM ne_order WHERE BUS_ASS_ID = ?",
#         conn, params=(DEFAULT_BUS_ID,)
#     )
#     df['SERVER_TIMESTAMP'] = pd.to_datetime(df['SERVER_TIMESTAMP'])
#     if start and end:
#         df = df[
#             (df['SERVER_TIMESTAMP'].dt.date >= start) &
#             (df['SERVER_TIMESTAMP'].dt.date <= end)
#         ]
#     if df.empty:
#         return "No orders found for the given period."
#
#     snapshot = df['SERVER_TIMESTAMP'].max()
#     rfm = df.groupby('USER_ID').agg(
#         recency=('SERVER_TIMESTAMP', lambda x: (snapshot - x.max()).days),
#         frequency=('SERVER_TIMESTAMP', 'count'),
#         monetary=('TOTAL', 'sum')
#     )
#
#     # Use safe_qcut instead of raw qcut
#     rfm['r_score'] = safe_qcut(rfm['recency'], q=5, ascending=False)
#     rfm['f_score'] = safe_qcut(rfm['frequency'], q=5, ascending=True)
#     rfm['m_score'] = safe_qcut(rfm['monetary'], q=5, ascending=True)
#
#     rfm['RFM_Score'] = (
#         rfm['r_score'].astype(str) +
#         rfm['f_score'].astype(str) +
#         rfm['m_score'].astype(str)
#     )
#
#     # Compute per-customer AOV and then per-segment stats
#     rfm['aov'] = rfm['monetary'] / rfm['frequency']
#     summary = rfm.groupby('RFM_Score').agg(
#         num_customers=('aov', 'count'),
#         avg_aov=('aov', 'mean')
#     ).sort_values('avg_aov', ascending=False)
#
#     text = "Customer segments and their AOV:\n"
#     for seg, row in summary.iterrows():
#         text += (
#             f"- Segment {seg}: {row['num_customers']} customers, "
#             f"avg AOV = ${row['avg_aov']:.2f}\n"
#         )
#     return text


def segment_customers(query: str = "") -> str:
    # 1) Fetch & filter orders
    start, end = parse_date_range(query)
    conn = get_connection()
    df = pd.read_sql_query(
        "SELECT USER_ID, SERVER_TIMESTAMP, TOTAL FROM ne_order WHERE BUS_ASS_ID = ?",
        conn, params=(DEFAULT_BUS_ID,)
    )
    df['SERVER_TIMESTAMP'] = pd.to_datetime(df['SERVER_TIMESTAMP'])
    if start and end:
        df = df[
            (df['SERVER_TIMESTAMP'].dt.date >= start) &
            (df['SERVER_TIMESTAMP'].dt.date <= end)
        ]
    if df.empty:
        return "No orders found for that period."

    # 2) Build RFM metrics per customer
    snapshot = df['SERVER_TIMESTAMP'].max()
    rfm = df.groupby('USER_ID').agg(
        recency=('SERVER_TIMESTAMP', lambda x: (snapshot - x.max()).days),
        frequency=('SERVER_TIMESTAMP', 'count'),
        monetary=('TOTAL', 'sum')
    )

    # 3) Score each dimension into quintiles
    rfm['r_score'] = safe_qcut(rfm['recency'], q=5, ascending=False)
    rfm['f_score'] = safe_qcut(rfm['frequency'], q=5, ascending=True)
    rfm['m_score'] = safe_qcut(rfm['monetary'],  q=5, ascending=True)

    # 4) Compute combined RFM_total
    rfm['RFM_total'] = rfm['r_score'] + rfm['f_score'] + rfm['m_score']

    # 5) Compute per-customer AOV
    rfm['aov'] = rfm['monetary'] / rfm['frequency']

    # 6) Bucket into tertiles: Low / Mid / High
    #    Use pandas qcut to split RFM_total into 3 groups
    #    labels: ['Low', 'Mid', 'High']
    rfm['segment'] = pd.qcut(
        rfm['RFM_total'],
        q=3,
        labels=['Low Value', 'Mid Value', 'High Value'],
        duplicates='drop'
    )

    # 7) Aggregate counts and AOV by segment
    summary = rfm.groupby('segment').agg(
        num_customers=('aov', 'count'),
        avg_aov=('aov', 'mean')
    ).loc[['High Value', 'Mid Value', 'Low Value']]  # order

    # 8) Build the output
    summary_text = "Customer Segments (RFM) by Value:\n"
    segments = {}
    chart_data = {}
    for seg, row in summary.iterrows():
        summary_text += f"- {seg}: {int(row['num_customers'])} customers, avg AOV=${row['avg_aov']:.2f}\n"
        segments[seg] = {'num_customers': int(row['num_customers']),
                         'avg_aov': round(row['avg_aov'], 2)}
        chart_data[seg] = int(row['num_customers'])

    return {
        'summary': summary_text,
        'segments': segments,
        'chart': {'type': 'bar', 'data': chart_data}
    }

    # for seg in ['High Value', 'Mid Value', 'Low Value']:
    #     if seg in summary.index:
    #         row = summary.loc[seg]
    #         text += (
    #             f"- {seg}: {row['num_customers']} customers, "
    #             f"Average Order Value (AOV) = ${row['avg_aov']:.2f}\n"
    #         )
    # return text



# def segment_customers(query: str = "") -> str:
#     start, end = parse_date_range(query)
#     conn = get_connection()
#     df = pd.read_sql_query(
#         "SELECT USER_ID, SERVER_TIMESTAMP, TOTAL FROM ne_order WHERE BUS_ASS_ID = ?",
#         conn, params=(DEFAULT_BUS_ID,)
#     )
#     df['SERVER_TIMESTAMP'] = pd.to_datetime(df['SERVER_TIMESTAMP'])
#     if start and end:
#         mask = (
#             (df['SERVER_TIMESTAMP'].dt.date >= start) &
#             (df['SERVER_TIMESTAMP'].dt.date <= end)
#         )
#         df = df.loc[mask]
#
#     if df.empty:
#         return "No orders found for the given period."
#
#     # Snapshot date for recency calculation
#     snapshot = df['SERVER_TIMESTAMP'].max()
#
#     # Build RFM table
#     rfm = df.groupby('USER_ID').agg(
#         recency=('SERVER_TIMESTAMP', lambda x: (snapshot - x.max()).days),
#         frequency=('SERVER_TIMESTAMP', 'count'),
#         monetary=('TOTAL', 'sum')
#     )
#
#     # Score each dimension with qcut; drop duplicate bins if necessary
#     rfm['r_score'] = pd.qcut(rfm['recency'],   5, labels=range(5, 0, -1), duplicates='drop')
#     rfm['f_score'] = pd.qcut(rfm['frequency'], 5, labels=range(1, 6),     duplicates='drop')
#     rfm['m_score'] = pd.qcut(rfm['monetary'],  5, labels=range(1, 6),     duplicates='drop')
#
#     # Combine to RFM segment code
#     rfm['RFM_Score'] = (
#         rfm['r_score'].astype(str) +
#         rfm['f_score'].astype(str) +
#         rfm['m_score'].astype(str)
#     )
#
#     # Compute AOV per customer, then segment‐level stats
#     rfm['aov'] = rfm['monetary'] / rfm['frequency']
#     summary = rfm.groupby('RFM_Score').agg(
#         num_customers=('aov', 'count'),
#         avg_aov=('aov', 'mean')
#     ).sort_values('avg_aov', ascending=False)
#
#     # Build output text
#     text = "Customer segments and their AOV:\n"
#     for seg, row in summary.iterrows():
#         text += (
#             f"- Segment {seg}: {row['num_customers']} customers, "
#             f"avg AOV = ${row['avg_aov']:.2f}\n"
#         )
#     return text


# def find_bundles(query: str = "") -> str:
#     start, end = parse_date_range(query)
#     conn = get_connection()
#     df = pd.read_sql_query(
#         "SELECT ORDER_ID, SKU_CODE FROM ne_order_items WHERE BUS_ASS_ID = ? AND SKU_CODE != 'NOSKU'",
#         conn, params=(DEFAULT_BUS_ID,)
#     )
#     df = df.dropna()
#     if start and end:
#         orders = pd.read_sql_query(
#             "SELECT ORDER_ID, SERVER_TIMESTAMP FROM ne_order WHERE BUS_ASS_ID = ?",
#             conn, params=(DEFAULT_BUS_ID,)
#         )
#         orders['SERVER_TIMESTAMP'] = pd.to_datetime(orders['SERVER_TIMESTAMP'])
#         orders = orders[(orders['SERVER_TIMESTAMP'].dt.date >= start) & (orders['SERVER_TIMESTAMP'].dt.date <= end)]
#         df = df[df['ORDER_ID'].isin(orders['ORDER_ID'])]
#     basket = df.groupby(['ORDER_ID', 'SKU_CODE'])['SKU_CODE']               .count().unstack().fillna(0).astype(int)
#     freq_items = apriori(basket, min_support=0.01, use_colnames=True)
#     rules = association_rules(freq_items, metric="lift", min_threshold=1.2)
#     rules = rules.sort_values('lift', ascending=False).head(5)
#     text = "Top 5 product bundles:\n"
#     for _, row in rules.iterrows():
#         items = list(row['antecedents']) + list(row['consequents'])
#         text += f"{items}: support={row['support']:.2f}, confidence={row['confidence']:.2f}, lift={row['lift']:.2f}\n"
#     return text


def find_bundles(query: str = "") -> str:
    start, end = parse_date_range(query)
    conn = get_connection()
    # 1) Grab order items with their product names
    df = pd.read_sql_query(
        """
        SELECT ORDER_ID, SKU_CODE, PRODUCT_NAME
        FROM ne_order_items
        WHERE BUS_ASS_ID = ?
        """,
        conn,
        params=(DEFAULT_BUS_ID,)
    ).dropna(subset=['SKU_CODE', 'PRODUCT_NAME'])

    # 2) Exclude any SKU codes that begin with "NOSKU"
    # df = df[~df['SKU_CODE'].str.startswith('NOSKU')]

    # 3) Date-filter the orders if requested
    if start and end:
        orders = pd.read_sql_query(
            "SELECT ORDER_ID, SERVER_TIMESTAMP FROM ne_order WHERE BUS_ASS_ID = ?",
            conn,
            params=(DEFAULT_BUS_ID,)
        )
        orders['SERVER_TIMESTAMP'] = pd.to_datetime(orders['SERVER_TIMESTAMP'])
        valid = orders[
            (orders['SERVER_TIMESTAMP'].dt.date >= start) &
            (orders['SERVER_TIMESTAMP'].dt.date <= end)
        ]['ORDER_ID']
        df = df[df['ORDER_ID'].isin(valid)]

    if df.empty:
        return "No valid order-item data found for that period."

    # 4) Compute minimum support count
    n_orders = df['ORDER_ID'].nunique()
    min_support = 0.02
    support_count = max(1, int(min_support * n_orders))

    # 5) Filter SKUs by support so the pivot stays small
    sku_counts = df['SKU_CODE'].value_counts()
    freq_skus = sku_counts[sku_counts >= support_count].index
    df = df[df['SKU_CODE'].isin(freq_skus)]

    if len(freq_skus) < 2:
        return "Not enough frequent SKUs at 1% support to form bundles."

    # 6) Pivot into a basket matrix
    basket = (
        df.groupby(['ORDER_ID','SKU_CODE'])['SKU_CODE']
          .count()
          .unstack(fill_value=0)
    )

    # 7) Apriori + association rules
    freq_items = apriori(basket, min_support=min_support, use_colnames=True)
    if freq_items.empty:
        return "No frequent itemsets found at 1% support."

    try:
        rules = association_rules(freq_items, metric="lift", min_threshold=1.2)
    except ValueError:
        return "Frequent itemsets found but couldn't generate rules (lift threshold)."

    if rules.empty:
        return "Frequent itemsets found, but no strong rules (lift ≥ 1.2)."

    # 8) Map each SKU code in the rules to its PRODUCT_NAME
    sku_to_name = dict(
        df.drop_duplicates('SKU_CODE')
          .set_index('SKU_CODE')['PRODUCT_NAME']
    )

    # 9) Build the top-5 bundles text using product names
    top5 = rules.sort_values('lift', ascending=False).head(5)
    bundles = []
    summary_text = "Top 5 product bundles (by lift):\n"
    for _, row in top5.iterrows():
        codes = list(row['antecedents']) + list(row['consequents'])
        # names = [sku_to_name.get(c, c) for c in codes]
        names = codes
        bundles.append({
            'items': names,
            'support': round(row['support'], 2),
            'confidence': round(row['confidence'], 2),
            'lift': round(row['lift'], 2)
        })
        summary_text += (
            f"- {names}: support={row['support']:.2f}, "
            f"confidence={row['confidence']:.2f}, lift={row['lift']:.2f}\n"
        )
    return {'summary':summary_text, 'bundles': bundles, 'chart': None}


# def find_bundles(query: str = "") -> str:
#     start, end = parse_date_range(query)
#     conn = get_connection()
#     df = pd.read_sql_query(
#         "SELECT ORDER_ID, SKU_CODE FROM ne_order_items "
#         "WHERE BUS_ASS_ID = ? AND SKU_CODE != 'NOSKU'",
#         conn, params=(DEFAULT_BUS_ID,)
#     ).dropna()
#
#     # Date‐filter orders if needed
#     if start and end:
#         orders = pd.read_sql_query(
#             "SELECT ORDER_ID, SERVER_TIMESTAMP FROM ne_order "
#             "WHERE BUS_ASS_ID = ?",
#             conn, params=(DEFAULT_BUS_ID,)
#         )
#         orders['SERVER_TIMESTAMP'] = pd.to_datetime(orders['SERVER_TIMESTAMP'])
#         mask = (
#             (orders['SERVER_TIMESTAMP'].dt.date >= start) &
#             (orders['SERVER_TIMESTAMP'].dt.date <= end)
#         )
#         valid = set(orders.loc[mask, 'ORDER_ID'])
#         df = df[df['ORDER_ID'].isin(valid)]
#
#     if df.empty:
#         return "No order-item data found for the given period."
#
#     # Determine how many orders we have, and compute support threshold
#     n_orders = df['ORDER_ID'].nunique()
#     min_support = 0.01  # 1%
#     support_count = max(1, int(min_support * n_orders))
#
#     # Count SKU frequency and keep only those above threshold
#     sku_counts = df['SKU_CODE'].value_counts()
#     frequent_skus = sku_counts[sku_counts >= support_count].index
#     df = df[df['SKU_CODE'].isin(frequent_skus)]
#
#     if df.empty or len(frequent_skus) < 2:
#         return "Not enough frequent SKUs at 1% support to generate bundles."
#
#     # Build a smaller basket matrix
#     basket = (
#         df.groupby(['ORDER_ID', 'SKU_CODE'])['SKU_CODE']
#           .count()
#           .unstack(fill_value=0)
#     )
#
#     # Run Apriori on the reduced matrix
#     freq_items = apriori(basket, min_support=min_support, use_colnames=True)
#     if freq_items.empty:
#         return "No frequent itemsets found at 1% support."
#
#     # Generate rules (lift ≥1.2)
#     try:
#         rules = association_rules(freq_items, metric="lift", min_threshold=1.2)
#     except ValueError:
#         return "No association rules could be generated (perhaps no itemsets met lift ≥1.2)."
#
#     if rules.empty:
#         return "Found frequent itemsets but no strong rules at lift ≥1.2."
#
#     # Pick top 5 by lift
#     top5 = rules.sort_values('lift', ascending=False).head(5)
#     text = "Top 5 product bundles:\n"
#     for _, row in top5.iterrows():
#         items = list(row['antecedents']) + list(row['consequents'])
#         text += (
#             f"- {items}: support={row['support']:.2f}, "
#             f"confidence={row['confidence']:.2f}, lift={row['lift']:.2f}\n"
#         )
#     return text



# def analyze_marketing(query: str = "") -> str:
#     start, end = parse_date_range(query)
#     conn = get_connection()
#     orders = pd.read_sql_query(
#         "SELECT CHANNEL_TYPE, TOTAL, SERVER_TIMESTAMP FROM ne_order WHERE BUS_ASS_ID = ?",
#         conn, params=(DEFAULT_BUS_ID,)
#     )
#     orders['SERVER_TIMESTAMP'] = pd.to_datetime(orders['SERVER_TIMESTAMP'])
#     if start and end:
#         orders = orders[(orders['SERVER_TIMESTAMP'].dt.date >= start) & (orders['SERVER_TIMESTAMP'].dt.date <= end)]
#     summary = orders.groupby('CHANNEL_TYPE').agg(orders=('TOTAL', 'count'), revenue=('TOTAL', 'sum'))
#     text = "Marketing performance by channel:\n"
#     for idx, row in summary.iterrows():
#         text += f"Channel {idx}: {row.orders} orders, ${row.revenue:.2f} revenue\n"
#     return text

""" commented code """
# def analyze_financial_metrics(query: str = "") -> tuple[str, dict]:
#     """
#     Calculate month-by-month financial KPIs:
#     ROI, ROAS, CAC, CLTV, AOV, ARPU, CAGR, LTV:CAC ratio.
#     Expects optional CSVs under /app/data/ad_spend.csv and /app/data/marketing_cost.csv:
#       - ad_spend.csv with columns: month (YYYY-MM), ad_spend
#       - marketing_cost.csv with columns: month (YYYY-MM), marketing_cost
#     """
#     # 1) Date range
#     start, end = parse_date_range(query)
#     # default: last 12 months
#     today = pd.Timestamp.today().normalize().date()
#     if not start or not end:
#         end = today
#         start = (today - pd.DateOffset(months=12)).date()
#
#     # 2) Load orders
#     conn = get_connection()
#     df = pd.read_sql_query(
#         "SELECT TOTAL, USER_ID, SERVER_TIMESTAMP FROM ne_order "
#         "WHERE BUS_ASS_ID = ?",
#         conn, params=(DEFAULT_BUS_ID,)
#     )
#     df["SERVER_TIMESTAMP"] = pd.to_datetime(df["SERVER_TIMESTAMP"])
#     df = df[
#         (df["SERVER_TIMESTAMP"].dt.date >= start) &
#         (df["SERVER_TIMESTAMP"].dt.date <= end)
#     ]
#     if df.empty:
#         return "No orders in that period.", {}
#
#     # 3) Prepare monthly buckets
#     df["month"] = df["SERVER_TIMESTAMP"].dt.to_period("M").astype(str)
#     monthly = df.groupby("month").agg(
#         revenue=("TOTAL", "sum"),
#         orders=("TOTAL", "count"),
#         users=("USER_ID", "nunique"),
#     ).reset_index()
#
#     # New customers per month (first purchase)
#     first_purchase = (
#         df.groupby("USER_ID")["SERVER_TIMESTAMP"]
#           .min()
#           .reset_index()
#     )
#     first_purchase["month"] = first_purchase["SERVER_TIMESTAMP"].dt.to_period("M").astype(str)
#     new_cust = first_purchase.groupby("month").size().rename("new_customers")
#     monthly = monthly.join(new_cust, on="month").fillna({"new_customers": 0})
#
#     # 4) Read ad spend & marketing cost CSVs if present
#     spend_path = "/app/data/ad_spend.csv"
#     mcost_path = "/app/data/marketing_cost.csv"
#     if os.path.exists(spend_path):
#         ad = pd.read_csv(spend_path, dtype={"month":str})
#         ad = ad.set_index("month")["ad_spend"]
#         monthly["ad_spend"] = monthly["month"].map(ad).fillna(0)
#     else:
#         monthly["ad_spend"] = 0
#
#     if os.path.exists(mcost_path):
#         mc = pd.read_csv(mcost_path, dtype={"month":str})
#         mc = mc.set_index("month")["marketing_cost"]
#         monthly["marketing_cost"] = monthly["month"].map(mc).fillna(0)
#     else:
#         monthly["marketing_cost"] = monthly["ad_spend"]
#
#     # 5) Compute KPIs
#     monthly["AOV"] = monthly["revenue"] / monthly["orders"]
#     monthly["ARPU"] = monthly["revenue"] / monthly["users"]
#     # CAC = marketing_cost / new_customers
#     monthly["CAC"] = monthly.apply(
#         lambda row: row["marketing_cost"] / row["new_customers"]
#         if row["new_customers"] > 0 else 0,
#         axis=1
#     )
#     # CLTV = AOV * (orders/users) * lifespan(1yr)
#     monthly["freq_per_user"] = monthly["orders"] / monthly["users"]
#     monthly["CLTV"] = monthly["AOV"] * monthly["freq_per_user"] * 1
#
#     # ROI = (revenue - marketing_cost) / marketing_cost
#     monthly["ROI_pct"] = ((monthly["revenue"] - monthly["marketing_cost"])
#                           / monthly["marketing_cost"].replace(0, pd.NA)) * 100
#     monthly["ROI_pct"] = monthly["ROI_pct"].fillna(0)
#
#     # ROAS = revenue / marketing_cost
#     monthly["ROAS"] = monthly.apply(
#         lambda row: row["revenue"] / row["marketing_cost"]
#         if row["marketing_cost"] > 0 else 0,
#         axis=1
#     )
#
#     # LTV:CAC ratio
#     monthly["LTV_CAC"] = monthly.apply(
#         lambda row: row["CLTV"] / row["CAC"] if row["CAC"] > 0 else 0,
#         axis=1
#     )
#
#     # CAGR on revenue across whole period:
#     first_rev = monthly["revenue"].iloc[0]
#     last_rev  = monthly["revenue"].iloc[-1]
#     n_years = (pd.Period(monthly["month"].iloc[-1], "M")
#                - pd.Period(monthly["month"].iloc[0], "M")).n / 12
#     monthly["CAGR_pct"] = (
#         (last_rev / first_rev) ** (1 / n_years) - 1
#     ) * 100 if first_rev > 0 and n_years > 0 else 0
#
#     # 6) Build text summary
#     summary_text = "📊 Month-by-Month Financial Metrics:\n"
#     for _, row in monthly.iterrows():
#         summary_text += (
#             f"{row['month']}: AOV=${row['AOV']:.2f}, ARPU=${row['ARPU']:.2f}, "
#             f"CAC=${row['CAC']:.2f}, CLTV=${row['CLTV']:.2f}, "
#             f"ROI={row['ROI_pct']:.1f}%, ROAS={row['ROAS']:.2f}, "
#             f"LTV:CAC={row['LTV_CAC']:.2f}, CAGR={row['CAGR_pct']:.1f}%\n"
#         )
#
#     # 7) Prepare chart data (you can pick which series to plot)
#     chart_data = {
#         "AOV": monthly.set_index("month")["AOV"].round(2).to_dict(),
#         "ARPU": monthly.set_index("month")["ARPU"].round(2).to_dict(),
#         "CAC": monthly.set_index("month")["CAC"].round(2).to_dict(),
#         "CLTV": monthly.set_index("month")["CLTV"].round(2).to_dict(),
#         # Add others if desired...
#     }
#
#     # return txt, chart_data
#     return {
#         'summary': summary_text,
#         'chart': {'type': 'multi-line',
#                   'data': chart_data}
#     }

def analyze_marketing(query: str = "") -> str:
    # 1) Parse dates
    start, end = parse_date_range(query)
    conn = get_connection()
    # 2) Load orders
    df = pd.read_sql_query(
        "SELECT TOTAL, CHANNEL_TYPE, SERVER_TIMESTAMP FROM ne_order WHERE BUS_ASS_ID = ?",
        conn, params=(DEFAULT_BUS_ID,)
    )
    df['SERVER_TIMESTAMP'] = pd.to_datetime(df['SERVER_TIMESTAMP'])
    if start and end:
        mask = (df['SERVER_TIMESTAMP'].dt.date >= start) & (df['SERVER_TIMESTAMP'].dt.date <= end)
        df = df.loc[mask]
    # 3) Prepare for analysis
    q = query.lower()
    # ----- Monthly AOV Trend -----
    if ("aov" in q or "average order value" in q) and "monthly" in q:
        # bucket by month
        df['month'] = df['SERVER_TIMESTAMP'].dt.to_period('M').astype(str)
        summary = (
            df.groupby('month')
              .agg(orders=('TOTAL', 'count'), revenue=('TOTAL', 'sum'))
        )
        summary['aov'] = summary['revenue'] / summary['orders']
        # build human-readable text
        text = "📊 Monthly AOV Trend:\n"
        for month, row in summary.iterrows():
            text += f"- {month}: AOV = ${row['aov']:.2f} (Revenue: ${row['revenue']:.0f}, Orders: {row['orders']})\n"
        return text

    # ----- Channel Aggregates (fallback) -----
    summary = df.groupby('CHANNEL_TYPE').agg(
        orders=('TOTAL', 'count'),
        revenue=('TOTAL', 'sum')
    )
    summary_text = "Marketing performance by channel:\n"
    chart_data = {}
    channels = {}
    for channel, row in summary.iterrows():
        summary_text += f"- Channel {channel}: {row.orders} orders, ${row.revenue:.2f} revenue\n"
        channels[int(channel)] = {'orders': int(row.orders), 'revenue': round(row.revenue, 2)}
        chart_data[f'Channel {channel}'] = round(row.revenue, 2)
    return {'summary': summary_text, 'channels': channels, 'chart': {'type':'bar','data': chart_data}}


def analyze_trends(query: str = "") -> dict:
    # Generic trend analyzer; defaults to monthly growth
    return analyze_monthly_growth(query)

def analyze_monthly_growth(query: str = "") -> tuple[str, dict]:
    """
    Calculate month-over-month growth percentages for revenue.
    Returns (text_summary, chart_data) where chart_data maps month -> growth%.
    """
    # 1. Determine date range: default to last 12 months if none given
    start, end = parse_date_range(query)
    if not start or not end:
        # default: one year back from today
        today = pd.Timestamp.today().normalize().date()
        start = (today - pd.DateOffset(months=12)).date()
        end = today

    conn = get_connection()
    df = pd.read_sql_query(
        "SELECT TOTAL, SERVER_TIMESTAMP FROM ne_order WHERE BUS_ASS_ID = ?",
        conn, params=(DEFAULT_BUS_ID,)
    )
    df["SERVER_TIMESTAMP"] = pd.to_datetime(df["SERVER_TIMESTAMP"])
    mask = (df["SERVER_TIMESTAMP"].dt.date >= start) & (df["SERVER_TIMESTAMP"].dt.date <= end)
    df = df.loc[mask]

    # 2. Group by month
    df["month"] = df["SERVER_TIMESTAMP"].dt.to_period("M").astype(str)
    monthly = df.groupby("month").agg(revenue=("TOTAL", "sum")).reset_index()

    # 3. Compute month-over-month growth %
    monthly["prev_revenue"] = monthly["revenue"].shift(1)
    monthly["growth_pct"] = ((monthly["revenue"] / monthly["prev_revenue"]) - 1) * 100
    monthly = monthly.dropna(subset=["growth_pct"])

    # 4. Build text summary
    summary = "📈 Monthly Growth Rate (%):\n"
    for _, row in monthly.iterrows():
        summary += f"- {row['month']}: {row['growth_pct']:.2f}%\n"

    # 5. Chart data: { "2024-06": 5.23, "2024-07": -2.15, ... }
    growth = monthly.set_index("month")["growth_pct"].round(2).to_dict()
    return {
        'summary': summary,
        'monthly_growth': growth,
        # 'cagr': round(cagr,2),
        'chart': {'type':'line','data':growth}
    }






def analyze_financial_metrics(query: str = "") -> dict:
    """
    Calculate month-by-month financial KPIs as requested in the query.
    Supports: ROI, ROAS, CAC, CLTV, AOV, ARPU, CAGR, LTV:CAC.
    """
    # 0) Figure out which metrics the user wants
    q = query.lower()
    # map simple keywords → dataframe column names
    metric_map = {
        'aov':      'AOV',
        'arpu':     'ARPU',
        # 'cac':      'CAC',
        'cltv':     'CLTV',
        # 'roi':      'ROI_pct',
        # 'roas':     'ROAS',
        # 'ltv:cac':  'LTV_CAC',
    }
    # always track CAGR separately
    want_cagr = 'cagr' in q
    # pick only those metrics mentioned
    requested = [metric_map[k] for k in metric_map if k in q]
    # if user asked generically for "financial metrics" or none found, show all
    if 'financial metrics' in q or not requested:
        requested = list(metric_map.values())

    # 1) Date range
    start, end = parse_date_range(query)
    today = pd.Timestamp.today().normalize().date()
    if not start or not end:
        end = today
        start = (today - pd.DateOffset(months=12)).date()

    # 2) Load orders & filter
    conn = get_connection()
    df = pd.read_sql_query(
        "SELECT TOTAL, USER_ID, SERVER_TIMESTAMP FROM ne_order WHERE BUS_ASS_ID = ?",
        conn, params=(DEFAULT_BUS_ID,)
    )
    df['SERVER_TIMESTAMP'] = pd.to_datetime(df['SERVER_TIMESTAMP'])
    df = df[(df['SERVER_TIMESTAMP'].dt.date >= start) &
            (df['SERVER_TIMESTAMP'].dt.date <= end)]
    if df.empty:
        return {'summary': 'No orders in that period.', 'metrics': {}, 'cagr': 0, 'chart': None}

    # 3) Build monthly summary
    df['month'] = df['SERVER_TIMESTAMP'].dt.to_period('M').astype(str)
    monthly = df.groupby('month').agg(
        revenue=('TOTAL','sum'),
        orders=('TOTAL','count'),
        users=('USER_ID','nunique'),
    ).reset_index()

    # new customers
    fp = df.groupby('USER_ID')['SERVER_TIMESTAMP'].min().reset_index()
    fp['month'] = fp['SERVER_TIMESTAMP'].dt.to_period('M').astype(str)
    newcust = fp.groupby('month').size().rename('new_customers')
    monthly = monthly.join(newcust, on='month').fillna({'new_customers':0})

    # optional spend files (same as before)...
    # [snip in this example for brevity—but keep your ad_spend/marketing_cost logic here]

    # 4) Compute **all** KPIs into the DataFrame
    monthly['AOV']      = monthly['revenue']/monthly['orders']
    monthly['ARPU']     = monthly['revenue']/monthly['users']
    # monthly['CAC']      = monthly.apply(lambda r: r['marketing_cost']/r['new_customers'] if r['new_customers']>0 else 0, axis=1)
    monthly['freq']     = monthly['orders']/monthly['users']
    monthly['CLTV']     = monthly['AOV'] * monthly['freq'] * 1
    # monthly['ROI_pct']  = ((monthly['revenue']-monthly['marketing_cost'])/monthly['marketing_cost'].replace(0,pd.NA)*100).fillna(0)
    # monthly['ROAS']     = monthly.apply(lambda r: r['revenue']/r['marketing_cost'] if r['marketing_cost']>0 else 0, axis=1)
    # monthly['LTV_CAC']  = monthly.apply(lambda r: r['CLTV']/r['CAC'] if r['CAC']>0 else 0, axis=1)

    # 5) Compute CAGR on revenue if needed
    first_rev,last_rev = monthly['revenue'].iloc[0], monthly['revenue'].iloc[-1]
    n_years = ((pd.Period(monthly['month'].iloc[-1],'M')-pd.Period(monthly['month'].iloc[0],'M')).n)/12
    cagr_pct = ((last_rev/first_rev)**(1/n_years)-1)*100 if first_rev>0 and n_years>0 else 0

    # 6) Build your output structures **only** with requested metrics
    metrics = {}
    for _, row in monthly.iterrows():
        entry = {}
        for col in requested:
            entry[col] = round(row[col], 2) if col != 'ROI_pct' else round(row[col], 1)
        metrics[row['month']] = entry

    # 7) Summary text: only mention the requested KPIs
    summary = "Financial Metrics by Month:\n"
    for mo, entry in metrics.items():
        parts = [f"{k}={entry[k]}" + ("%" if k in ['ROI_pct'] else "") for k in entry]
        summary += f"{mo}: " + ", ".join(parts) + "\n"
    if want_cagr:
        summary += f"\nOverall CAGR: {cagr_pct:.2f}%\n"

    # 8) Chart: multi-line of each requested metric
    chart_data = {col: {mo: entry[col] for mo,entry in metrics.items()} for col in requested}
    if want_cagr:
        chart_data['CAGR'] = cagr_pct

    return {
        'summary': summary,
        'metrics': metrics,
        # 'cagr': round(cagr_pct,2),
        'chart': {'type': 'multi-line', 'data': chart_data}
    }

#
# def analyze_product_performance(query: str = "") -> dict:
#     """
#     Answers:
#       - Top SKU revenue in period
#       - Number of SKUs making 80% revenue (Pareto)
#       - Year-over-year % change for top SKUs
#       - Month-over-month change in average basket size
#     Uses SKU_CODE, ignoring any with 'NOSKU'.
#     Returns dict with summary, data, chart.
#     """
#     q = query.lower()
#     # Determine date range
#     start, end = parse_date_range(query)
#     today = pd.Timestamp.today().date()
#     if not start or not end:
#         # default last 3 months
#         end = today
#         start = (pd.Timestamp(today) - pd.DateOffset(months=3)).date()
#
#     conn = get_connection()
#     # Load orders
#     orders = pd.read_sql_query(
#         "SELECT ORDER_ID, SERVER_TIMESTAMP, ITEMS_COUNT FROM ne_order WHERE BUS_ASS_ID = ?", conn,
#         params=(DEFAULT_BUS_ID,)
#     )
#     orders['SERVER_TIMESTAMP'] = pd.to_datetime(orders['SERVER_TIMESTAMP'])
#     # Filter orders by date
#     mask = (orders['SERVER_TIMESTAMP'].dt.date >= start) & (orders['SERVER_TIMESTAMP'].dt.date <= end)
#     orders = orders.loc[mask]
#
#     # Load order items
#     items = pd.read_sql_query(
#         "SELECT ORDER_ITEM_ID, ORDER_ID, SKU_CODE, QUANTITY, TOTAL FROM ne_order_items WHERE BUS_ASS_ID = ?", conn,
#         params=(DEFAULT_BUS_ID,)
#     )
#     # Filter items to matched orders and valid SKU codes
#     df_items = items[items['ORDER_ID'].isin(orders['ORDER_ID'])]
#     df_items = df_items[~df_items['SKU_CODE'].str.contains('NOSKU', na=False)]
#
#     result = {'summary':'','data':{},'chart':None}
#
#     # 1) Top revenue SKUs
#     if 'top revenue' in q or 'top skus' in q or 'top sku' in q:
#         sku_rev = df_items.groupby('SKU_CODE')['TOTAL'].sum().sort_values(ascending=False)
#         top = sku_rev.head(10)
#         text = 'Top 10 revenue SKUs:'
#         for sku, rev in top.items():
#             text += f"\n- {sku}: ${rev:.2f}"
#         result['summary'] = text
#         result['data'] = top.round(2).to_dict()
#         result['chart'] = {'type':'bar','data':result['data']}
#         return result
#
#     # 2) Pareto: SKUs making 80% of revenue
#     if '80%' in q or 'pareto' in q or 'producing 80%' in q:
#         sku_rev = df_items.groupby('SKU_CODE')['TOTAL'].sum().sort_values(ascending=False)
#         cum = sku_rev.cumsum()
#         total = sku_rev.sum()
#         pareto = cum[cum <= 0.8 * total]
#         count = pareto.size
#         total_skus = sku_rev.size
#         text = f"{count} SKUs generate 80% of revenue out of {total_skus} SKUs."
#         result['summary'] = text
#         result['data'] = {'skus_80pct': count, 'total_skus': total_skus}
#         result['chart'] = None
#         return result
#
#     # 3) YoY % change for top SKUs
#     if ('last year' in q or 'year to this year' in q) and ('top' in q):
#         # Determine years
#         this_year = end.year
#         last_year = this_year - 1
#         # Merge items with order timestamps
#         df = df_items.merge(
#             orders[['ORDER_ID','SERVER_TIMESTAMP']], on='ORDER_ID'
#         )
#         df['YEAR'] = df['SERVER_TIMESTAMP'].dt.year
#         rev_y = df.groupby(['YEAR','SKU_CODE'])['TOTAL'].sum().unstack(fill_value=0)
#         if last_year in rev_y.index and this_year in rev_y.index:
#             top_skus = rev_y.loc[this_year].nlargest(5).index
#             change = ((rev_y.loc[this_year, top_skus] - rev_y.loc[last_year, top_skus])
#                       / rev_y.loc[last_year, top_skus] * 100).round(2)
#             text = 'YoY % change for top 5 SKUs:'
#             for sku, pct in change.items():
#                 text += f"\n- {sku}: {pct:.2f}%"
#             result['summary'] = text
#             result['data'] = change.to_dict()
#             result['chart'] = {'type':'bar','data':result['data']}
#             return result
#
#     # 4) MoM average basket size change
#     if 'basket size' in q or 'average basket' in q:
#         orders['month'] = orders['SERVER_TIMESTAMP'].dt.to_period('M').astype(str)
#         avg_size = orders.groupby('month')['ITEMS_COUNT'].mean().round(2)
#         mom_change = avg_size.diff().round(2).fillna(0)
#         text = 'MoM change in avg basket size:'
#         for m, v in mom_change.items():
#             text += f"\n- {m}: {v:.2f}"
#         result['summary'] = text
#         result['data'] = mom_change.to_dict()
#         result['chart'] = {'type':'line','data':result['data']}
#         return result
#
#     # Fallback
#     result['summary'] = "Sorry, couldn't parse the product performance query."
#     return result




"""Product bundle code"""

# itemsets = apriori(basket, min_support=min_support, use_colnames=True)
# if itemsets.empty:
#     return {"summary": "No frequent itemsets at 1% support.", "bundles": [], "chart": None}
#
# # 6) Generate rules
# try:
#     rules = association_rules(itemsets, metric="lift", min_threshold=1.0)
# except ValueError:
#     return {"summary": "No strong association rules (lift ≥1.2).", "bundles": [], "chart": None}
# if rules.empty:
#     return {"summary": "No strong association rules (lift ≥1.2).", "bundles": [], "chart": None}
#
# # 7) Determine N: parse “top N” or default to 10
# m = re.search(r"\btop\s+(\d+)", query.lower())
# top_n = int(m.group(1)) if m else 10
#
# # 8) Select top N by lift
# top_rules = (
#     rules.sort_values("lift", ascending=False)
#     .head(top_n)
#     .reset_index(drop=True)
# )
# bundles = []
# for _, row in top_rules.iterrows():
#     bundles.append({
#         "antecedents": sorted(list(row["antecedents"])),
#         "consequents": sorted(list(row["consequents"])),
#         "support": round(row["support"], 3),
#         "confidence": round(row["confidence"], 3),
#         "lift": round(row["lift"], 3)
#     })
#
# # 9) Build summary text and chart data
# summary_lines = [
#     f"{i + 1}. {b['antecedents']} → {b['consequents']} "
#     f"(support={b['support']}, conf={b['confidence']}, lift={b['lift']})"
#     for i, b in enumerate(bundles)
# ]
# summary = (
#         f"Top {len(bundles)} product bundles by lift "
#         f"from {start} to {end}:\n" + "\n".join(summary_lines)
# )
# chart = {
#     "type": "bar",
#     "data": {
#         f"{b['antecedents']}→{b['consequents']}": b["lift"]
#         for b in bundles
#     }
# }
#
# return {"summary": summary, "bundles": bundles, "chart": chart}

"""commented code from agent app"""
# @app.post("/query")
# async def query(request: QueryRequest):
#     try:
#         result = agent.run(request.query)
#         return {"result": result}
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))

# @app.post("/query")
# async def query(request: QueryRequest):
#     try:
#         # If you want to silence the deprecation warning, you could later switch to agent.invoke()
#         agent_response = agent.invoke(request.query)
#         # result = agent_response.output if hasattr(agent_response, "output") else agent_response
#
#         # If it has an `.output` attribute, use that; otherwise stringify
#         if hasattr(agent_response, "output"):
#             final = agent_response.output
#         elif isinstance(agent_response, dict) and "output" in agent_response:
#             final = agent_response["output"]
#         else:
#             final = str(agent_response)
#         return {"result": final}
#
#     except Exception as e:
#         # Log full traceback to container logs
#         tb = traceback.format_exc()
#         logger.error(f"Error while running agent:\n{tb}")
#         # Return the error message (and optionally traceback) in the HTTP response
#         return JSONResponse(
#             status_code=500,
#             content={
#                 "error": str(e),
#                 # "traceback": tb  # you can include this if you want it on the client
#             }
#         )
#q = request.query.lower()
# if any(k in q for k in
#        ["roi", "roas", "cac", "cltv", "aov", "arpu", "cagr", "ltv:cac", "financial metrics"]):
#     response = analyze_financial_metrics(request.query)
#     # return {
#     #     "result": text,
#     #     "chart": {"type": "multi-line", "data": chart_data}
#     # }
#     return response
#
# # Monthly AOV
# if "monthly" in q and ("aov" in q or "average order value" in q):
#     response = analyze_marketing(request.query)
#     # return {"result": text, "chart": {"type": "bar", "data": chart_data}}
#     return response
#
# # Monthly Growth / "CAGR" requests
# if "monthly" in q and ("growth" in q):
#     response = analyze_monthly_growth(request.query)
#     # return {"result": text, "chart": {"type": "line", "data": chart_data}}
#     return response
# Fallback to agent


