import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from utils import util

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


def get_customer_segment_counts(query: str = "") -> dict:
    """
    Returns the number of customers in each RFM segment (High, Mid, Low) over a specified period.
    Steps:
      1. Parse date range from the query (e.g. "last 3 months", "January 2025").
      2. Load orders (USER_ID, SERVER_TIMESTAMP, TOTAL) for the default business.
      3. Filter orders to that date range.
      4. Compute RFM metrics per customer:
         - recency: days since last purchase
         - frequency: number of orders
         - monetary: total spend
      5. Score each metric into quintiles using safe_qcut (recency reversed).
      6. Sum the three scores into RFM_total, bucket into tertiles labeled
         ["Low Value", "Mid Value", "High Value"].
      7. Count customers in each segment and return a bar‐chart payload.
    """
    start, end = util.parse_date_range(query)
    conn = util.get_connection()
    df = pd.read_sql_query(
        "SELECT USER_ID, SERVER_TIMESTAMP, TOTAL FROM ne_order WHERE BUS_ASS_ID = ?",
        conn, params=(util.DEFAULT_BUS_ID,)
    )
    df['SERVER_TIMESTAMP'] = pd.to_datetime(df['SERVER_TIMESTAMP'])
    if start and end:
        df = df[
            (df['SERVER_TIMESTAMP'].dt.date >= start) &
            (df['SERVER_TIMESTAMP'].dt.date <= end)
        ]
    if df.empty:
        return {"summary": "No orders in that period.", "data": {}, "chart": None}

    snapshot = df['SERVER_TIMESTAMP'].max()
    rfm = df.groupby('USER_ID').agg(
        recency=('SERVER_TIMESTAMP', lambda x: (snapshot - x.max()).days),
        frequency=('SERVER_TIMESTAMP', 'count'),
        monetary=('TOTAL', 'sum')
    )

    rfm['r_score'] = safe_qcut(rfm['recency'],   q=5, ascending=False)
    rfm['f_score'] = safe_qcut(rfm['frequency'], q=5, ascending=True)
    rfm['m_score'] = safe_qcut(rfm['monetary'],  q=5, ascending=True)
    rfm['RFM_total'] = rfm['r_score'] + rfm['f_score'] + rfm['m_score']
    rfm['segment'] = pd.qcut(
        rfm['RFM_total'], q=3,
        labels=['Low Value','Mid Value','High Value'],
        duplicates='drop'
    )

    counts = (
        rfm['segment']
        .value_counts()
        .reindex(['High Value','Mid Value','Low Value'], fill_value=0)
        .astype(int)
    )
    data = counts.to_dict()
    summary = "Customer count per segment:\n" + "\n".join(
        f"- {seg}: {cnt} customers" for seg, cnt in data.items()
    )
    return {
        "summary": summary,
        "data": data,
        "chart": {"type": "bar", "data": data}
    }


def get_customer_segment_aov(query: str = "") -> dict:
    """
    Returns the average order value (AOV) for each RFM segment (High, Mid, Low)
    over a specified period.
    Steps:
      • Parses date range from the query.
      • Loads and filters orders as in get_customer_segment_counts.
      • Builds the same RFM segmentation.
      • Computes per-customer AOV = monetary / frequency.
      • Averages that AOV within each segment.
      • Returns a bar‐chart payload of segment → avg AOV.
    """
    start, end = util.parse_date_range(query)
    conn = util.get_connection()
    df = pd.read_sql_query(
        "SELECT USER_ID, SERVER_TIMESTAMP, TOTAL FROM ne_order WHERE BUS_ASS_ID = ?",
        conn, params=(util.DEFAULT_BUS_ID,)
    )
    df['SERVER_TIMESTAMP'] = pd.to_datetime(df['SERVER_TIMESTAMP'])
    if start and end:
        df = df[
            (df['SERVER_TIMESTAMP'].dt.date >= start) &
            (df['SERVER_TIMESTAMP'].dt.date <= end)
        ]
    if df.empty:
        return {"summary": "No orders in that period.", "data": {}, "chart": None}

    snapshot = df['SERVER_TIMESTAMP'].max()
    rfm = df.groupby('USER_ID').agg(
        recency=('SERVER_TIMESTAMP', lambda x: (snapshot - x.max()).days),
        frequency=('SERVER_TIMESTAMP', 'count'),
        monetary=('TOTAL', 'sum')
    )
    rfm['r_score'] = safe_qcut(rfm['recency'],   q=5, ascending=False)
    rfm['f_score'] = safe_qcut(rfm['frequency'], q=5, ascending=True)
    rfm['m_score'] = safe_qcut(rfm['monetary'],  q=5, ascending=True)
    rfm['RFM_total'] = rfm['r_score'] + rfm['f_score'] + rfm['m_score']
    rfm['segment'] = pd.qcut(
        rfm['RFM_total'], q=3,
        labels=['Low Value','Mid Value','High Value'],
        duplicates='drop'
    )
    rfm['aov'] = rfm['monetary'] / rfm['frequency']

    avg_aov = (
        rfm.groupby('segment')['aov']
        .mean()
        .reindex(['High Value','Mid Value','Low Value'], fill_value=0)
        .round(2)
    )
    data = avg_aov.to_dict()
    summary = "Average Order Value (AOV) per segment:\n" + "\n".join(
        f"- {seg}: ${val:.2f}" for seg, val in data.items()
    )
    return {
        "summary": summary,
        "data": data,
        "chart": {"type": "bar", "data": data}
    }


def get_customer_segment_overview(query: str = "") -> dict:
    """
    Fallback segmentation tool: returns both customer counts and avg AOV per segment.
    Use when the agent cannot decide which specific segmentation metric was requested.
    Steps:
      • Performs the same segmentation pipeline.
      • Returns two series: segment→count and segment→avg AOV.
      • Provides a multi-bar chart with side‐by‐side bars for count vs AOV.
    """
    # reuse the count tool and aov tool internally
    counts_res = get_customer_segment_counts(query)
    aov_res    = get_customer_segment_aov(query)

    # merge their data dicts
    counts = counts_res["data"]
    aov    = aov_res["data"]
    data = {
        seg: {"count": counts.get(seg, 0), "avg_aov": aov.get(seg, 0.0)}
        for seg in ["High Value","Mid Value","Low Value"]
    }

    summary = "Customer Segmentation Overview:\n"
    for seg, vals in data.items():
        summary += f"- {seg}: {vals['count']} customers, avg AOV=${vals['avg_aov']:.2f}\n"

    return {
        "summary": summary,
        "data": data,
        "chart": {
            "type": "multi-bar",
            "data": {
                "count": {seg: vals["count"] for seg, vals in data.items()},
                "avg_aov": {seg: vals["avg_aov"] for seg, vals in data.items()},
            }
        }
    }

def kmeans_customer_segmentation(query: str = "") -> dict:
    """
    Segment customers into High/Mid/Low value groups using K-Means clustering.
    • Parses date ranges (e.g. “last 6 months”).
    • Builds RFM features: recency (days), frequency (count), monetary (sum).
    • Derives avg_order_value = monetary / frequency.
    • Scales features for KMeans.
    • Clusters into 3 groups, orders them by average monetary value.
    Returns:
      - summary: human-readable counts per segment
      - segments: { "High Value": {num_customers, avg_rfm…}, … }
      - chart: { type:"bar", data: { segment: num_customers } }
    """
    # 1) Date range and data fetch
    start, end = util.parse_date_range(query)
    conn = util.get_connection()
    df = pd.read_sql_query(
        "SELECT USER_ID, SERVER_TIMESTAMP, TOTAL "
        "FROM ne_order WHERE BUS_ASS_ID = ?",
        conn, params=(util.DEFAULT_BUS_ID,)
    )
    df["SERVER_TIMESTAMP"] = pd.to_datetime(df["SERVER_TIMESTAMP"])
    if start and end:
        mask = (df["SERVER_TIMESTAMP"].dt.date >= start) & \
               (df["SERVER_TIMESTAMP"].dt.date <= end)
        df = df.loc[mask]
    if df.empty:
        return {"summary": "No orders in that period.", "segments": {}, "chart": None}

    # 2) Build RFM table
    snapshot = df["SERVER_TIMESTAMP"].max()
    rfm = df.groupby("USER_ID").agg(
        recency=("SERVER_TIMESTAMP", lambda x: (snapshot - x.max()).days),
        frequency=("SERVER_TIMESTAMP", "count"),
        monetary=("TOTAL", "sum")
    ).reset_index()

    # 3) Derive avg order value
    rfm["avg_order_value"] = rfm["monetary"] / rfm["frequency"]

    # 4) Prepare feature matrix
    features = rfm[["recency", "frequency", "monetary", "avg_order_value"]].values
    scaler = StandardScaler()
    X = scaler.fit_transform(features)

    # 5) K-Means clustering
    kmeans = KMeans(n_clusters=3, random_state=42, n_init="auto")
    labels = kmeans.fit_predict(X)
    rfm["cluster"] = labels

    # 6) Determine cluster ordering by centroid monetary
    centroids = scaler.inverse_transform(kmeans.cluster_centers_)
    # centroids[:,2] is monetary mean for each cluster
    order = sorted(
        range(3),
        key=lambda i: centroids[i, 2],  # sort by monetary
        reverse=True
    )
    label_map = {
        order[0]: "High Value",
        order[1]: "Mid Value",
        order[2]: "Low Value"
    }
    rfm["segment"] = rfm["cluster"].map(label_map)

    # 7) Aggregate results
    agg = rfm.groupby("segment").agg(
        num_customers = ("USER_ID", "count"),
        recency_mean  = ("recency", "mean"),
        freq_mean     = ("frequency", "mean"),
        monetary_mean = ("monetary", "mean"),
        aov_mean      = ("avg_order_value", "mean")
    ).loc[["High Value", "Mid Value", "Low Value"]]

    # 8) Build output
    summary = "K-Means Customer Segments:\n"
    segments = {}
    chart = {}
    for seg, row in agg.iterrows():
        summary += (
            f"- {seg}: {int(row['num_customers'])} customers, "
            f"avg recency={row['recency_mean']:.1f}d, "
            f"freq={row['freq_mean']:.1f}, "
            f"monetary=${row['monetary_mean']:.2f}, "
            f"AOV=${row['aov_mean']:.2f}\n"
        )
        segments[seg] = {
            "num_customers": int(row["num_customers"]),
            "recency": round(row["recency_mean"], 1),
            "frequency": round(row["freq_mean"], 1),
            "monetary": round(row["monetary_mean"], 2),
            "aov": round(row["aov_mean"], 2)
        }
        chart[seg] = int(row["num_customers"])

    return {
        "summary": summary,
        "segments": segments,
        "chart": {"type": "bar", "data": chart}
    }
