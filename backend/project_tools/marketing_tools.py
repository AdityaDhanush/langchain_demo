# marketing_tools.py

import pandas as pd
from utils import util

def get_monthly_aov_trend(query: str = "") -> dict:
    """
    Calculate the month-by-month Average Order Value (AOV) trend.
    • Parses date ranges like “last 6 months” or explicit dates.
    • Queries ne_order for TOTAL and SERVER_TIMESTAMP filtered by BUS_ASS_ID.
    • Buckets by month, computes AOV = revenue/orders.
    • Returns a dict with:
        - summary: human-readable lines per month
        - data: { month: aov_value, … }
        - chart: { type: 'line', data }
    """
    start, end = util.parse_date_range(query)
    today = pd.Timestamp.today().date()
    if not start or not end:
        end = today
        start = (pd.Timestamp(today) - pd.DateOffset(months=12)).date()

    conn = util.get_connection()
    df = pd.read_sql_query(
        "SELECT TOTAL, SERVER_TIMESTAMP FROM ne_order WHERE BUS_ASS_ID = ?",
        conn, params=(util.DEFAULT_BUS_ID,)
    )
    df["SERVER_TIMESTAMP"] = pd.to_datetime(df["SERVER_TIMESTAMP"])
    df = df[(df["SERVER_TIMESTAMP"].dt.date >= start) &
            (df["SERVER_TIMESTAMP"].dt.date <= end)]

    if df.empty:
        return {"summary": "No orders in that period.", "data": {}, "chart": None}

    df["month"] = df["SERVER_TIMESTAMP"].dt.to_period("M").astype(str)
    agg = df.groupby("month").agg(
        revenue=("TOTAL", "sum"),
        orders =("TOTAL", "count")
    )
    agg["aov"] = agg["revenue"] / agg["orders"]

    data = agg["aov"].round(2).to_dict()
    summary = "Monthly AOV Trend:\n" + "\n".join(
        f"- {m}: ${v:.2f}" for m, v in data.items()
    )
    return {
        "summary": summary,
        "data": data,
        "chart": {"type": "line", "data": data}
    }


def get_channel_performance(query: str = "") -> dict:
    """
    Analyze performance per marketing channel.
    • Parses date ranges.
    • Queries ne_order for TOTAL, CHANNEL_TYPE, SERVER_TIMESTAMP.
    • Filters by BUS_ASS_ID and date.
    • Aggregates orders & revenue per CHANNEL_TYPE.
    • Returns:
        - summary: lines per channel
        - data: { channel: {orders, revenue}, … }
        - chart: { type: 'bar', data: { channel: revenue } }
    """
    start, end = util.parse_date_range(query)
    conn = util.get_connection()
    df = pd.read_sql_query(
        "SELECT TOTAL, CHANNEL_TYPE, SERVER_TIMESTAMP FROM ne_order WHERE BUS_ASS_ID = ?",
        conn, params=(util.DEFAULT_BUS_ID,)
    )
    df["SERVER_TIMESTAMP"] = pd.to_datetime(df["SERVER_TIMESTAMP"])
    if start and end:
        df = df[(df["SERVER_TIMESTAMP"].dt.date >= start) &
                (df["SERVER_TIMESTAMP"].dt.date <= end)]

    if df.empty:
        return {"summary": "No orders in that period.", "data": {}, "chart": None}

    agg = df.groupby("CHANNEL_TYPE").agg(
        orders = ("TOTAL", "count"),
        revenue =("TOTAL", "sum")
    )
    data = {
        int(ch): {"orders": int(r.orders), "revenue": round(r.revenue,2)}
        for ch, r in agg.iterrows()
    }
    chart = {f"Channel {ch}": round(r.revenue,2) for ch, r in agg.iterrows()}
    summary = "Marketing Performance by Channel:\n" + "\n".join(
        f"- Channel {ch}: {vals['orders']} orders, ${vals['revenue']:.2f} revenue"
        for ch, vals in data.items()
    )
    return {
        "summary": summary,
        "data": data,
        "chart": {"type": "bar", "data": chart}
    }


def get_monthly_revenue_growth(query: str = "") -> dict:
    """
    Compute month-over-month revenue growth percentages.
    • Parses date ranges (default last 12 months).
    • Queries ne_order for TOTAL, SERVER_TIMESTAMP.
    • Buckets by month, sums revenue, computes growth_pct.
    • Returns:
        - summary: lines per month with %
        - monthly_growth: { month: pct, … }
        - chart: { type:'line', data: monthly_growth }
    """
    start, end = util.parse_date_range(query)
    today = pd.Timestamp.today().date()
    if not start or not end:
        end = today
        start = (pd.Timestamp(today) - pd.DateOffset(months=12)).date()

    conn = util.get_connection()
    df = pd.read_sql_query(
        "SELECT TOTAL, SERVER_TIMESTAMP FROM ne_order WHERE BUS_ASS_ID = ?",
        conn, params=(util.DEFAULT_BUS_ID,)
    )
    df["SERVER_TIMESTAMP"] = pd.to_datetime(df["SERVER_TIMESTAMP"])
    df = df[(df["SERVER_TIMESTAMP"].dt.date >= start) &
            (df["SERVER_TIMESTAMP"].dt.date <= end)]
    if df.empty:
        return {"summary": "No orders in that period.", "monthly_growth": {}, "chart": None}

    df["month"] = df["SERVER_TIMESTAMP"].dt.to_period("M").astype(str)
    monthly = df.groupby("month").agg(revenue=("TOTAL","sum")).reset_index()
    monthly["prev"] = monthly["revenue"].shift(1)
    monthly["growth_pct"] = ((monthly["revenue"]/monthly["prev"] - 1)*100).fillna(0)
    growth = monthly.set_index("month")["growth_pct"].round(2).to_dict()

    summary = "Monthly Revenue Growth (%):\n" + "\n".join(
        f"- {m}: {v:.2f}%" for m, v in growth.items()
    )
    return {
        "summary": summary,
        "monthly_growth": growth,
        "chart": {"type": "line", "data": growth}
    }


def marketing_overview(query: str = "") -> dict:
    """
    Fallback marketing tool:
    Combines AOV trend, channel performance, and revenue growth.
    • Calls get_monthly_aov_trend, get_channel_performance, get_monthly_revenue_growth.
    • Returns:
        - summary: concatenated summaries
        - data: { aov:…, channels:…, growth:… }
        - chart: None (frontend can pick individual charts)
    """
    aov = get_monthly_aov_trend(query)
    chn = get_channel_performance(query)
    grw = get_monthly_revenue_growth(query)
    data = {
        "monthly_aov": aov["data"],
        "channels":    chn["data"],
        "growth":      grw["monthly_growth"]
    }
    summary = "\n\n".join([aov["summary"], chn["summary"], grw["summary"]])
    return {"summary": summary, "data": data, "chart": None}
