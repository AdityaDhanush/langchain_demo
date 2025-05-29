# bundle_tools.py

import re
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
from utils import util


def get_frequent_itemsets(query: str = "") -> dict:
    """
    List frequent itemsets of SKUs (min_support=1%) over a period.
    • Parses natural-language dates (e.g. “last 6 months”, “Apr 2025”).
    • Joins ne_order_items → ne_order, filters BUS_ASS_ID and SKU_CODE NOT LIKE 'NOSKU%'.
    • Filters by the date range.
    • Builds a basket matrix after dropping SKUs below the 1% support count.
    • Returns all itemsets with their support values.
    """
    start, end = util.parse_date_range(query)
    today = pd.Timestamp.today().normalize().date()
    if not start or not end:
        end   = today
        start = (pd.Timestamp(today) - pd.DateOffset(months=12)).date()

    sql = """
    SELECT oi.ORDER_ID, oi.SKU_CODE
    FROM ne_order_items oi
    JOIN ne_order o ON oi.ORDER_ID = o.ORDER_ID
    WHERE oi.BUS_ASS_ID = ?
      AND oi.SKU_CODE NOT LIKE 'NOSKU%'
    """
    conn = util.get_connection()
    df = pd.read_sql_query(sql, conn, params=(util.DEFAULT_BUS_ID,))
    df['SERVER_TIMESTAMP'] = pd.to_datetime(
        pd.read_sql_query(
            "SELECT ORDER_ID, SERVER_TIMESTAMP FROM ne_order WHERE BUS_ASS_ID=?",
            conn, params=(util.DEFAULT_BUS_ID,)
        ).set_index('ORDER_ID').loc[df['ORDER_ID']]['SERVER_TIMESTAMP']
    )
    mask = (df['SERVER_TIMESTAMP'].dt.date >= start) & (df['SERVER_TIMESTAMP'].dt.date <= end)
    df = df.loc[mask]
    if df.empty:
        return {"summary": "No order items in that period.", "itemsets": [], "chart": None}

    # compute support threshold
    n_orders = df['ORDER_ID'].nunique()
    min_support = 0.01
    support_count = max(1, int(min_support * n_orders))

    # filter SKUs by frequency
    freqs = df['SKU_CODE'].value_counts()
    keep = freqs[freqs >= support_count].index
    df = df[df['SKU_CODE'].isin(keep)]

    # basket
    basket = df.groupby(['ORDER_ID','SKU_CODE'])['SKU_CODE']\
               .count().unstack(fill_value=0)
    freq_items = apriori(basket, min_support=min_support, use_colnames=True)
    if freq_items.empty:
        return {"summary": "No frequent itemsets at 1% support.", "itemsets": [], "chart": None}

    # format
    itemsets = freq_items.sort_values('support', ascending=False)\
                         .to_dict(orient='records')
    summary = (
        f"Found {len(itemsets)} frequent itemsets (support ≥1%) "
        f"from {start} to {end}."
    )
    chart = {
        "type": "bar",
        "data": {str(row['itemsets']): round(row['support'], 2) for row in itemsets[:10]}
    }
    return {"summary": summary, "itemsets": itemsets, "chart": chart}


def get_association_rules(query: str = "") -> dict:
    """
    Generate strong association rules (lift ≥1.2).
    • Uses the same filtered basket from get_frequent_itemsets.
    • Runs association_rules(metric='lift', min_threshold=1.2).
    • Returns rules with antecedents, consequents, support, confidence, lift.
    """
    # reuse frequent itemsets routine to get basket
    fi_res = get_frequent_itemsets(query)
    if not fi_res.get('itemsets'):
        return {"summary": fi_res['summary'], "rules": [], "chart": None}

    # Rebuild basket for rules
    start, end = util.parse_date_range(query)
    today = pd.Timestamp.today().normalize().date()
    if not start or not end:
        end   = today
        start = (pd.Timestamp(today) - pd.DateOffset(months=12)).date()
    conn = util.get_connection()
    df = pd.read_sql_query("""
        SELECT oi.ORDER_ID, oi.SKU_CODE
        FROM ne_order_items oi
        JOIN ne_order o ON oi.ORDER_ID = o.ORDER_ID
        WHERE oi.BUS_ASS_ID=? AND oi.SKU_CODE NOT LIKE 'NOSKU%'
    """, conn, params=(util.DEFAULT_BUS_ID,))
    df['SERVER_TIMESTAMP'] = pd.to_datetime(
        pd.read_sql_query(
            "SELECT ORDER_ID, SERVER_TIMESTAMP FROM ne_order WHERE BUS_ASS_ID=?",
            conn, params=(util.DEFAULT_BUS_ID,)
        ).set_index('ORDER_ID').loc[df['ORDER_ID']]['SERVER_TIMESTAMP']
    )
    mask = (df['SERVER_TIMESTAMP'].dt.date >= start) & (df['SERVER_TIMESTAMP'].dt.date <= end)
    df = df.loc[mask]
    # re-filter SKUs by support count
    n_orders = df['ORDER_ID'].nunique()
    support_count = max(1, int(0.01 * n_orders))
    keep = df['SKU_CODE'].value_counts()[lambda x: x>=support_count].index
    df = df[df['SKU_CODE'].isin(keep)]
    basket = df.groupby(['ORDER_ID','SKU_CODE'])['SKU_CODE']\
               .count().unstack(fill_value=0)
    fi = apriori(basket, min_support=0.01, use_colnames=True)
    try:
        rules = association_rules(fi, metric="lift", min_threshold=1.2)
    except ValueError:
        return {"summary": "Frequent itemsets found, but no rules at lift ≥1.2.", "rules": [], "chart": None}

    if rules.empty:
        return {"summary": "No strong association rules (lift ≥1.2).", "rules": [], "chart": None}

    # format
    recs = []
    for _, r in rules.sort_values('lift', ascending=False).head(10).iterrows():
        recs.append({
            "antecedents": list(r['antecedents']),
            "consequents": list(r['consequents']),
            "support": round(r['support'],2),
            "confidence": round(r['confidence'],2),
            "lift": round(r['lift'],2)
        })
    summary = f"Top {len(recs)} association rules (lift ≥1.2)."
    chart = {
        "type": "bar",
        "data": {f"{r['antecedents']}→{r['consequents']}": r['lift'] for r in recs}
    }
    return {"summary": summary, "rules": recs, "chart": chart}


def get_product_bundles(query: str = "") -> dict:
    """
    Generates the top N product bundles by lift over a specified period.

    • Parses date ranges (e.g. "last 6 months", "Jan 2025"); defaults to last 365 days.
    • Joins ne_order_items → ne_order, filters BUS_ASS_ID and SKU_CODE NOT LIKE 'NOSKU%'.
    • Drops infrequent SKUs (support <1%) before pivoting to control memory.
    • Runs Apriori (min_support=1%) + association_rules(lift>=1.2).
    • Returns the top N bundles (antecedents→consequents) with support, confidence, and lift.
      Default N=10, or parsed from "top N" in the query.
    """
    # 1) Determine date range (default = last 365 days)
    start, end = util.parse_date_range(query)
    today = pd.Timestamp.today().normalize().date()
    if not start or not end:
        end = today
        start = (pd.Timestamp(today) - pd.DateOffset(days=365)).date()

    # 2) Extract raw order-item data once
    conn = util.get_connection()
    df = pd.read_sql_query(
        """
        SELECT oi.ORDER_ID, oi.SKU_CODE
        FROM ne_order_items oi
        JOIN ne_order       o  ON oi.ORDER_ID = o.ORDER_ID
        WHERE oi.BUS_ASS_ID = ?
          
        """,
        conn, params=(util.DEFAULT_BUS_ID,)
    )
    if df.empty:
        return {"summary": "No order-item data found.", "bundles": [], "chart": None}

    # 3) Attach timestamps & filter by date
    ts = pd.read_sql_query(
        "SELECT ORDER_ID, SERVER_TIMESTAMP FROM ne_order WHERE BUS_ASS_ID = ?",
        conn, params=(util.DEFAULT_BUS_ID,)
    )
    ts["SERVER_TIMESTAMP"] = pd.to_datetime(ts["SERVER_TIMESTAMP"])
    df = df.merge(ts, on="ORDER_ID")
    mask = (df["SERVER_TIMESTAMP"].dt.date >= start) & (df["SERVER_TIMESTAMP"].dt.date <= end)
    df = df.loc[mask]
    if df.empty:
        return {
            "summary": f"No order-items between {start} and {end}.",
            "bundles": [],
            "chart": None
        }

    # 4) Pre-filter SKUs by 1% support
    total_orders = df["ORDER_ID"].nunique()
    min_support = 0.01
    support_count = max(1, int(total_orders * min_support))
    sku_counts = df["SKU_CODE"].value_counts()
    frequent_skus = sku_counts[sku_counts >= support_count].index
    df = df[df["SKU_CODE"].isin(frequent_skus)]
    if df["SKU_CODE"].nunique() < 2:
        return {"summary": "Not enough frequent SKUs to generate bundles.", "bundles": [],
                "chart": None}

    # 5) Build basket & find itemsets
    basket = df.groupby(["ORDER_ID", "SKU_CODE"])["SKU_CODE"] \
        .count().unstack(fill_value=0)

    # 6) Apriori with a slightly lower support floor (0.5%)
    min_support = 0.005
    itemsets = apriori(basket, min_support=min_support, use_colnames=True)
    if itemsets.empty:
        return {"summary": "Not enough frequent SKUs to form bundles.", "bundles": [],
                "chart": None}

    # 7) Try association rules with relaxed lift threshold (≥1.0)
    rules = association_rules(itemsets, metric="lift", min_threshold=1.0)
    # If still no rules, fall back to top itemsets by support
    if rules.empty:
        # pick top 10 itemsets (size ≥2) by support
        freq = itemsets[itemsets["itemsets"].map(len) >= 2]
        top_itemsets = freq.sort_values("support", ascending=False).head(10)
        bundles = []
        for _, row in top_itemsets.iterrows():
            bundles.append({
                "items": sorted(list(row["itemsets"])),
                "support": round(row["support"], 3)
            })
        summary = (
            f"No association rules at lift ≥1.0; "
            f"falling back to the top {len(bundles)} frequent itemsets (support ≥{min_support * 100:.1f}%):"
        )
        chart_data = {str(b["items"]): b["support"] for b in bundles}
        return {"summary": summary, "bundles": bundles,
                "chart": {"type": "bar", "data": chart_data}}

    # 8) Otherwise, select top N rules by lift (default N=10 or from query)
    top_n = int(re.search(r"\btop\s+(\d+)", query.lower()).group(1)) if re.search(r"\btop\s+(\d+)",
                                                                                  query.lower()) else 10
    top_rules = rules.sort_values("lift", ascending=False).head(top_n)
    bundles = []
    for _, r in top_rules.iterrows():
        bundles.append({
            "antecedents": sorted(list(r["antecedents"])),
            "consequents": sorted(list(r["consequents"])),
            "support": round(r["support"], 3),
            "confidence": round(r["confidence"], 3),
            "lift": round(r["lift"], 3)
        })
    summary = (
        f"Top {len(bundles)} product bundles by lift (≥1.0) "
        f"from {start} to {end}:"
    )
    chart_data = {
        f"{b['antecedents']}→{b['consequents']}": b["lift"]
        for b in bundles
    }
    return {"summary": summary, "bundles": bundles, "chart": {"type": "bar", "data": chart_data}}


def get_bundle_overview(query: str = "") -> dict:
    """
    Fallback bundle tool:
    Returns both frequent itemsets and association rules.
    Use when agent is uncertain which specific bundle tool to call.
    """
    fi = get_frequent_itemsets(query)
    ar = get_association_rules(query)
    summary = "Bundle Overview:\n" + fi['summary'] + "\n" + ar['summary']
    return {
        "summary": summary,
        "itemsets": fi.get('itemsets', []),
        "rules": ar.get('rules', []),
        "chart": None
    }
