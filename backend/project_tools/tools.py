from langchain.agents import Tool
from .product_tools import *
from .segment_tools import *
from .bundle_tools import *
from .marketing_tools import *
from .data_analysis_tools import *

product_tool = [
    Tool(
        name="top_revenue",
        func=get_top_revenue_products,
        description=(
            "Returns the top N SKUs by total revenue over a user-specified period. "
            "It parses date expressions like “last 3 months” or specific months/years, "
            "filters out any SKU_CODE starting with “NOSKU”, then sums up revenue per SKU. "
            "By default returns the top 5, but honors requests such as “top 10 products this year.” "
            "Use this when the user asks “What are the top revenue generating products...” or "
            "“Which products generated the most revenue in April 2025?”."
        )
    ),
    Tool(
        name="pareto_products",
        func=get_pareto_products,
        description=(
            "Identifies which SKUs—or categories, if the user mentions “category”—account for "
            "80% of total revenue in a given period. It filters out placeholder SKUs (NOSKU), "
            "groups revenue by SKU or CATEGORY, computes the cumulative sum, and returns the count "
            "and list of items that together make up 80% of revenue. Use for queries like "
            "“How many products produce 80% of revenue?” or “80% of revenue comes from which categories?”."
        )
    ),
    Tool(
        name="yoy_revenue_change",
        func=get_yoy_revenue_change,
        description=(
            "Calculates the year-over-year percentage change in revenue for the top N SKUs. "
            "It determines this year and last year’s revenues per SKU (filtering out NOSKU), "
            "automatically picks the top 5 if no N is specified, and returns the % growth for each. "
            "Intended for questions like “What is the percentage change in top products’ revenue from last year to this year?”"
        )
    ),
    Tool(
        name="mom_basket_change",
        func=get_mom_basket_size_change,
        description=(
            "Computes the month-over-month percentage change in average basket size (items per order). "
            "It buckets orders by month, averages the ITEMS_COUNT per order, calculates the % change "
            "from each month to the next, and returns both the raw percentages and a line-chart series. "
            "Use for queries like “What is the change in the average basket size month on month?” or "
            "“MoM basket size change for the last 6 months.”"
        )
    ),
    Tool(
            name="product_overview",
            func=product_overview,
            description=(
                "General product analytics fallback tool. "
                "Use when the user asks any product-related question not covered by the other tools. "
                "It filters out SKU_CODEs starting with 'NOSKU', parses date ranges (e.g. 'last 3 months', 'Apr 2025'), "
                "groups by product or category, aggregates total revenue and units sold, "
                "honors 'top N' or 'bottom N' requests, and can sort by revenue or quantity."
            )
        ),
]

segment_tool = [
    Tool(
        name="customer_segment_counts",
        func=get_customer_segment_counts,
        description=(
            "Count customers in each RFM segment (High/Mid/Low) over a period. "
            "Parses natural-language dates (e.g. 'last 6 months', 'Jan 2025'), "
            "computes recency, frequency, monetary, assigns each customer to High/Mid/Low value, "
            "and returns a bar chart of counts per segment."
        )
    ),
    Tool(
        name="customer_segment_aov",
        func=get_customer_segment_aov,
        description=(
            "Calculate the Average Order Value (AOV) for each RFM segment (High/Mid/Low) over a period. "
            "Parses dates from the query, segments customers into High/Mid/Low, "
            "and returns a bar chart of segment→avg AOV."
        )
    ),
    Tool(
        name="customer_segment_overview",
        func=get_customer_segment_overview,
        description=(
            "General customer segmentation overview. "
            "If the agent cannot determine counts vs AOV, this fallback returns both: "
            "customer count and avg AOV per segment (High/Mid/Low), "
            "with a multi-bar chart for side-by-side visualization."
        )
    ),
    Tool(
        name="kmeans_customer_segmentation",
        func=kmeans_customer_segmentation,
        description=(
            "Use K-Means clustering to segment customers into High, Mid, and Low value groups. "
            "Parses date ranges, computes Recency, Frequency, Monetary, and Average Order Value features, "
            "scales them, runs KMeans(n_clusters=3), "
            "and labels clusters based on average monetary value. "
            "Returns summary, per-segment stats, and a bar chart of customer counts."
        )
    ),
]

bundle_tool = [
    Tool(
        name="frequent_itemsets",
        func=get_frequent_itemsets,
        description=(
            "List all frequent SKU itemsets (support ≥1%) over a given period. "
            "Parses date ranges, excludes any SKU_CODE starting with 'NOSKU', "
            "builds a basket matrix, and returns itemsets with their support values. "
            "Use for queries like 'Frequent itemsets last 3 months' or "
            "'Which product combinations occur most often?'"
        )
    ),
    Tool(
        name="association_rules",
        func=get_association_rules,
        description=(
            "Generate association rules with lift ≥1.2 from frequent itemsets. "
            "Filters out 'NOSKU' SKUs, parses dates, runs Apriori then association_rules, "
            "and returns top rules with support, confidence, and lift. "
            "Use for queries like 'Show me association rules' or "
            "'Which products are frequently bought together with high lift?'"
        )
    ),
    Tool(
        name="product_bundles",
        func=get_product_bundles,
        description=(
            "Generate the top N product bundles based on market-basket analysis "
            "for a user-specified period. Parses date ranges like “last 6 months” or explicit dates, "
            "filters out any SKU_CODE starting with 'NOSKU%', pre-filters to SKUs with ≥1% support to control memory, "
            "runs Apriori (min_support=1%) and association_rules (lift ≥1.2), "
            "and returns the top 10 bundles (or N if the user asks 'top N'). "
            "Each bundle includes antecedents, consequents, support, confidence, and lift."
        )
    ),
    Tool(
        name="bundle_overview",
        func=get_bundle_overview,
        description=(
            "Fallback tool that provides both frequent itemsets and association rules. "
            "Use when the agent cannot determine whether to call frequent_itemsets, "
            "association_rules, or product_bundles."
        )
    ),
    # …other tools…
]

marketing_tool = [
    Tool(
        name="monthly_aov_trend",
        func=get_monthly_aov_trend,
        description=(
            "Compute month-by-month Average Order Value (AOV). "
            "Parses natural-language dates, filters orders by BUS_ASS_ID and date, "
            "aggregates revenue and orders per month, and returns a line-chart series. "
            "Use for queries like “monthly AOV trend for the last year.”"
        )
    ),
    Tool(
        name="channel_performance",
        func=get_channel_performance,
        description=(
            "Analyze marketing performance by channel. "
            "Parses dates, filters orders by BUS_ASS_ID and date, "
            "aggregates orders and revenue per CHANNEL_TYPE, and returns a bar-chart. "
            "Use for queries like “revenue by channel this quarter.”"
        )
    ),
    Tool(
        name="monthly_revenue_growth",
        func=get_monthly_revenue_growth,
        description=(
            "Calculate month-over-month revenue growth percentages. "
            "Parses date ranges, sums monthly revenue, computes growth %, "
            "and returns a line-chart. "
            "Use for queries like “monthly revenue growth %.”"
        )
    ),
    Tool(
        name="marketing_overview",
        func=marketing_overview,
        description=(
            "Fallback marketing analysis tool. "
            "When the agent is unsure which specific metric to use, this provides "
            "monthly AOV trend, channel performance, and monthly revenue growth "
            "in a consolidated summary and data dict."
        )
    ),
    # … other tools …
]

data_analysis_tool = [
    Tool(
            name="data_overview",
            func=data_overview,
            description=(
                "General data-analysis fallback tool. "
                "When the agent cannot identify a specific analytic tool, "
                "this returns row counts, column names, and a 10-row sample "
                "for either the ne_order or ne_order_items table, based on the query."
            )
        )
]


all_tools = product_tool + segment_tool + bundle_tool + marketing_tool + data_analysis_tool