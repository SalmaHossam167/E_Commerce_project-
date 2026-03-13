"""
05_KPIs_RFM.py
--------------
Business KPIs calculation and RFM customer segmentation
for the Olist Brazilian E-Commerce dataset.

Sections:
    1. Business KPIs  - ARPU, Product Margin, Gross Adds, Churn Rate
    2. RFM Analysis   - Customer segmentation based on Recency, Frequency, Monetary
    3. Recommendations - Actionable insights per customer segment

Outputs:
    outputs/Analysis_RFM_KPIs.png
    outputs/rfm_analysis.csv

Run order:
    01_EDA.py -> 02_Feature_Engineering.py -> 03_ML_Model.py
    -> 04_Advanced_Analysis.py -> 05_KPIs_RFM.py
"""

import os
import warnings

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DATA = "data/"
OUT  = "outputs/"
os.makedirs(OUT, exist_ok=True)

DARK  = "#0f172a"
CARD  = "#1e293b"
TEXT  = "#e2e8f0"
SUB   = "#94a3b8"
PURP  = "#6366f1"
GREEN = "#10b981"
AMBER = "#f59e0b"
RED   = "#ef4444"
BLUE  = "#3b82f6"
PINK  = "#ec4899"

SEGMENT_COLORS = {
    "Champions":           AMBER,
    "Loyal Customers":     GREEN,
    "Potential Loyalists": BLUE,
    "At Risk":             PINK,
    "Lost Customers":      RED,
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def style_axis(ax, title, xlabel="", ylabel=""):
    """Apply consistent dark-theme styling to a matplotlib axis."""
    ax.set_facecolor(CARD)
    ax.set_title(title, color=TEXT, fontsize=11, fontweight="bold", pad=10)
    ax.tick_params(colors=SUB, labelsize=8)
    ax.set_xlabel(xlabel, color=SUB, fontsize=8)
    ax.set_ylabel(ylabel, color=SUB, fontsize=8)
    for spine in ax.spines.values():
        spine.set_edgecolor("#334155")
    ax.grid(axis="y", color="#334155", linewidth=0.5, alpha=0.5)


def assign_segment(score):
    """Map RFM total score to a customer segment label."""
    if score >= 13:
        return "Champions"
    elif score >= 10:
        return "Loyal Customers"
    elif score >= 7:
        return "Potential Loyalists"
    elif score >= 5:
        return "At Risk"
    else:
        return "Lost Customers"


# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
print("=" * 55)
print("  STEP 5: KPIs & RFM Analysis")
print("=" * 55)
print("\nLoading datasets...")

orders    = pd.read_csv(OUT  + "final_orders_complete.csv")
items     = pd.read_csv(DATA + "olist_order_items_dataset.csv")
payments  = pd.read_csv(DATA + "cleaned_payments.csv")
customers = pd.read_csv(DATA + "olist_customers_dataset.csv")

print("Datasets loaded successfully.")

orders["order_purchase_timestamp"] = pd.to_datetime(orders["order_purchase_timestamp"])
orders["month"] = orders["order_purchase_timestamp"].dt.to_period("M")

cust_orders = orders.merge(
    customers[["customer_id", "customer_unique_id"]], on="customer_id", how="left"
)
order_payments = cust_orders.merge(
    payments[["order_id", "payment_value"]], on="order_id", how="left"
)

# ---------------------------------------------------------------------------
# Section 1: Business KPIs
# ---------------------------------------------------------------------------
print("\n" + "-" * 45)
print("  Section 1: Business KPIs")
print("-" * 45)

total_revenue    = payments["payment_value"].sum()
unique_customers = customers["customer_unique_id"].nunique()
arpu             = total_revenue / unique_customers

avg_margin = (
    (items["price"] - items["freight_value"]) / items["price"]
).mean() * 100

first_purchase = (
    cust_orders
    .groupby("customer_unique_id")["month"]
    .min()
    .reset_index()
    .rename(columns={"month": "first_month"})
)
gross_adds = first_purchase.groupby("first_month").size()
avg_gross  = gross_adds.mean()
peak_month = gross_adds.idxmax()
peak_adds  = gross_adds.max()

repeat         = cust_orders.groupby("customer_unique_id").size()
churn_rate     = (repeat == 1).mean() * 100
retention_rate = 100 - churn_rate

returning_monthly = (
    cust_orders[cust_orders["customer_unique_id"].isin(repeat[repeat > 1].index)]
    .groupby("month")["customer_unique_id"]
    .nunique()
)
net_adds = gross_adds.subtract(returning_monthly, fill_value=0)

print(f"\n  ARPU:              R${arpu:.2f}")
print(f"  Product Margin:    {avg_margin:.1f}%")
print(f"  Avg Gross Adds:    {avg_gross:.0f} new customers / month")
print(f"  Peak Gross Adds:   {peak_adds:,} in {peak_month}")
print(f"  Avg Net Adds:      {net_adds.mean():.0f} customers / month")
print(f"  Churn Rate:        {churn_rate:.1f}%")
print(f"  Retention Rate:    {retention_rate:.1f}%")

# ---------------------------------------------------------------------------
# Section 2: RFM Analysis
# ---------------------------------------------------------------------------
print("\n" + "-" * 45)
print("  Section 2: RFM Analysis")
print("-" * 45)

snapshot_date = orders["order_purchase_timestamp"].max() + pd.Timedelta(days=1)

rfm = order_payments.groupby("customer_unique_id").agg(
    Recency   = ("order_purchase_timestamp", lambda x: (snapshot_date - x.max()).days),
    Frequency = ("order_id",                 "nunique"),
    Monetary  = ("payment_value",            "sum"),
).reset_index()

rfm["R_Score"] = pd.qcut(rfm["Recency"],                        5, labels=[5, 4, 3, 2, 1])
rfm["F_Score"] = pd.qcut(rfm["Frequency"].rank(method="first"), 5, labels=[1, 2, 3, 4, 5])
rfm["M_Score"] = pd.qcut(rfm["Monetary"],                       5, labels=[1, 2, 3, 4, 5])

rfm["RFM_Score"] = (
    rfm["R_Score"].astype(int)
    + rfm["F_Score"].astype(int)
    + rfm["M_Score"].astype(int)
)

rfm["Segment"] = rfm["RFM_Score"].apply(assign_segment)

seg_summary = (
    rfm.groupby("Segment")
    .agg(
        Count       = ("customer_unique_id", "count"),
        Avg_Revenue = ("Monetary",           "mean"),
        Avg_Recency = ("Recency",            "mean"),
    )
    .round(2)
    .sort_values("Avg_Revenue", ascending=False)
)

print("\n  Customer Segments:\n")
print(seg_summary.to_string())

# ---------------------------------------------------------------------------
# Section 3: Business Recommendations
# ---------------------------------------------------------------------------
print("\n" + "-" * 45)
print("  Section 3: Business Recommendations")
print("-" * 45)

champions = rfm[rfm["Segment"] == "Champions"]
loyal     = rfm[rfm["Segment"] == "Loyal Customers"]
potential = rfm[rfm["Segment"] == "Potential Loyalists"]
at_risk   = rfm[rfm["Segment"] == "At Risk"]
lost      = rfm[rfm["Segment"] == "Lost Customers"]

print(f"""
  CHAMPIONS ({len(champions):,} customers | Avg Revenue: R${champions['Monetary'].mean():.0f})
    - Launch a VIP program with early access to new products
    - Spend {champions['Monetary'].mean() / arpu:.1f}x more than the average customer

  LOYAL CUSTOMERS ({len(loyal):,} customers | Avg Revenue: R${loyal['Monetary'].mean():.0f})
    - Introduce a loyalty points system
    - Offer bundle discounts to increase average order value

  POTENTIAL LOYALISTS ({len(potential):,} customers | Avg Revenue: R${potential['Monetary'].mean():.0f})
    - Send a personalised follow-up email after the first purchase
    - Offer 10% discount on second order to convert to loyal

  AT RISK ({len(at_risk):,} customers | Avg days since last purchase: {at_risk['Recency'].mean():.0f})
    - Run a win-back campaign with a limited-time offer
    - Provide a free-shipping coupon to encourage return

  LOST CUSTOMERS ({len(lost):,} customers)
    - Last resort: offer 20-30% discount or a free gift
    - Remove from active marketing list if no response

  PRIORITY ACTION:
    - Churn rate of {churn_rate:.1f}% is the most critical metric to improve
    - Converting {len(potential):,} Potential Loyalists to Loyal
      is estimated to increase revenue by ~25%
""")

# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------
print("Generating charts...")

fig = plt.figure(figsize=(20, 14), facecolor=DARK)
gs  = gridspec.GridSpec(
    2, 2, figure=fig,
    hspace=0.45, wspace=0.35,
    top=0.91, bottom=0.06, left=0.06, right=0.97,
)
fig.text(
    0.5, 0.96,
    "RFM Customer Segmentation & Business KPIs",
    ha="center", color=TEXT, fontsize=15, fontweight="bold",
)

# Chart 1: Segment distribution (donut)
ax1 = fig.add_subplot(gs[0, 0])
seg_counts = rfm["Segment"].value_counts()
pie_colors = [SEGMENT_COLORS.get(s, BLUE) for s in seg_counts.index]
wedges, texts, autotexts = ax1.pie(
    seg_counts.values,
    labels=[f"{s}\n{v:,}" for s, v in zip(seg_counts.index, seg_counts.values)],
    autopct="%1.1f%%",
    colors=pie_colors,
    startangle=90,
    wedgeprops={"width": 0.55},
    textprops={"color": TEXT, "fontsize": 8},
)
for at in autotexts:
    at.set_color(DARK)
    at.set_fontsize(8)
    at.set_fontweight("bold")
ax1.set_title("Customer Segment Distribution", color=TEXT,
              fontsize=11, fontweight="bold", pad=10)
ax1.set_facecolor(CARD)

# Chart 2: Average revenue per segment
ax2 = fig.add_subplot(gs[0, 1])
seg_rev    = rfm.groupby("Segment")["Monetary"].mean().sort_values(ascending=True)
bar_colors = [SEGMENT_COLORS.get(s, BLUE) for s in seg_rev.index]
ax2.barh(range(len(seg_rev)), seg_rev.values, color=bar_colors, edgecolor="none")
ax2.set_yticks(range(len(seg_rev)))
ax2.set_yticklabels([s[:28] for s in seg_rev.index], color=SUB, fontsize=8)
for i, val in enumerate(seg_rev.values):
    ax2.text(val + 3, i, f"R${val:.0f}", va="center", color=TEXT, fontsize=9)
style_axis(ax2, "Average Revenue per Customer Segment", "Avg Revenue (R$)", "")
ax2.grid(axis="x", color="#334155", linewidth=0.5, alpha=0.5)
ax2.grid(axis="y", color="none")

# Chart 3: KPI summary bars
ax3 = fig.add_subplot(gs[1, 0])
kpi_labels = ["ARPU\n(R$)", "Avg Margin\n(%)", "Gross Adds\n(/month / 10)", "Churn Rate\n(%)"]
kpi_values = [arpu, avg_margin, avg_gross / 10, churn_rate]
kpi_colors = [GREEN, BLUE, AMBER, RED]
bars = ax3.bar(range(4), kpi_values, color=kpi_colors, edgecolor="none", width=0.5)
ax3.set_xticks(range(4))
ax3.set_xticklabels(kpi_labels, color=SUB, fontsize=9)
display_labels = [f"R${arpu:.0f}", f"{avg_margin:.1f}%",
                  f"{avg_gross:.0f}", f"{churn_rate:.1f}%"]
for bar, lbl in zip(bars, display_labels):
    ax3.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 1,
        lbl, ha="center", color=TEXT, fontsize=10, fontweight="bold",
    )
style_axis(ax3, "Business KPIs Overview", "", "Value")

# Chart 4: Monthly gross adds (area chart)
ax4 = fig.add_subplot(gs[1, 1])
gross_plot    = gross_adds.tail(20)
tick_positions = range(0, len(gross_plot), 3)
ax4.fill_between(range(len(gross_plot)), gross_plot.values, color=PURP, alpha=0.4)
ax4.plot(range(len(gross_plot)), gross_plot.values, color=PURP, linewidth=2.5)
ax4.set_xticks(list(tick_positions))
ax4.set_xticklabels(
    [str(gross_plot.index[i]) for i in tick_positions],
    rotation=30, color=SUB, fontsize=7,
)
style_axis(ax4, "Monthly New Customer Acquisition (Gross Adds)", "Month", "New Customers")

plt.savefig(OUT + "Analysis_RFM_KPIs.png", dpi=150, bbox_inches="tight", facecolor=DARK)
print(f"  Saved: {OUT}Analysis_RFM_KPIs.png")

# ---------------------------------------------------------------------------
# Export RFM table
# ---------------------------------------------------------------------------
rfm.to_csv(OUT + "rfm_analysis.csv", index=False)
print(f"  Saved: {OUT}rfm_analysis.csv")

print("\n" + "=" * 55)
print("  STEP 5 complete.")
print("=" * 55)
