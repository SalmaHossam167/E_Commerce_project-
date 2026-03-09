# E-Commerce Project - Step 1: EDA (Exploratory Data Analysis)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings
warnings.filterwarnings('ignore')

# ── File paths ──
DATA = 'data/'       # فولدر الـ CSV
OUT  = 'outputs/'    # فولدر الصور والنتايج

import os
os.makedirs(OUT, exist_ok=True)

print("=" * 55)
print("  STEP 1: EDA - Exploratory Data Analysis")
print("=" * 55)

# ── Download data ──
print("\n the data is being dawnloaded...")
orders    = pd.read_csv(DATA + 'cleaned_orders_dataset.csv')
customers = pd.read_csv(DATA + 'cleaned_customers.csv')
payments  = pd.read_csv(DATA + 'cleaned_payments.csv')
products  = pd.read_csv(DATA + 'cleaned_products_english.csv')
reviews   = pd.read_csv(DATA + 'olist_order_reviews_dataset.csv')
items     = pd.read_csv(DATA + 'olist_order_items_dataset.csv')
print("All files have been uploaded")

# ── Data processing──
for col in ['order_purchase_timestamp', 'order_approved_at',
            'order_delivered_carrier_date', 'order_delivered_customer_date',
            'order_estimated_delivery_date']:
    orders[col] = pd.to_datetime(orders[col])

orders['delivery_days'] = (
    orders['order_delivered_customer_date'] - orders['order_purchase_timestamp']
).dt.days

orders['is_late'] = (
    orders['order_delivered_customer_date'] > orders['order_estimated_delivery_date']
).astype(float)

orders['purchase_month'] = orders['order_purchase_timestamp'].dt.to_period('M')
orders['purchase_hour']  = orders['order_purchase_timestamp'].dt.hour

delivered = orders[orders['order_status'] == 'delivered'].copy()
monthly   = orders.groupby('purchase_month').size().reset_index(name='count')
monthly['purchase_month'] = monthly['purchase_month'].astype(str)

items_prod = items.merge(
    products[['product_id', 'product_category_name_english']], on='product_id', how='left'
)
top_cats   = items_prod.groupby('product_category_name_english')['price'].sum().sort_values(ascending=False).head(10)
pay_type   = payments['payment_type'].value_counts()
rev_dist   = reviews['review_score'].value_counts().sort_index()
top_states = customers['customer_state'].value_counts().head(10)
hour_dist  = orders.groupby('purchase_hour').size()

# ── Colors ──
DARK  = '#0f172a'
CARD  = '#1e293b'
TEXT  = '#e2e8f0'
SUB   = '#94a3b8'
PURP  = '#6366f1'
GREEN = '#10b981'
AMBER = '#f59e0b'
RED   = '#ef4444'
BLUE  = '#3b82f6'
PINK  = '#ec4899'

def style(ax, title, xlabel='', ylabel=''):
    ax.set_facecolor(CARD)
    ax.set_title(title, color=TEXT, fontsize=11, fontweight='bold', pad=10)
    ax.tick_params(colors=SUB, labelsize=8)
    ax.set_xlabel(xlabel, color=SUB, fontsize=8)
    ax.set_ylabel(ylabel, color=SUB, fontsize=8)
    for spine in ax.spines.values():
        spine.set_edgecolor('#334155')
    ax.grid(axis='y', color='#334155', linewidth=0.5, alpha=0.5)

print("\n The dashboard is being drawn")

fig = plt.figure(figsize=(20, 24), facecolor=DARK)
gs  = gridspec.GridSpec(4, 3, figure=fig, hspace=0.5, wspace=0.35,
                         top=0.93, bottom=0.04, left=0.06, right=0.97)

fig.text(0.5, 0.965, ' Olist E-Commerce — EDA Dashboard',
         ha='center', color=TEXT, fontsize=16, fontweight='bold')
fig.text(0.5, 0.950,
         f"99,441 Customers  |  98,666 Orders  |  R$ 16.0M Revenue  |  8.1% Late Deliveries",
         ha='center', color=SUB, fontsize=11)

# 1. Monthly Trend
ax1 = fig.add_subplot(gs[0, :])
ax1.set_facecolor(CARD)
ax1.plot(range(len(monthly)), monthly['count'], color=PURP, linewidth=2.5, marker='o', markersize=4)
ax1.fill_between(range(len(monthly)), monthly['count'], alpha=0.15, color=PURP)
ax1.set_xticks(range(len(monthly)))
ax1.set_xticklabels(monthly['purchase_month'], rotation=45, ha='right', color=SUB, fontsize=8)
ax1.tick_params(axis='y', colors=SUB)
for spine in ax1.spines.values(): spine.set_edgecolor('#334155')
ax1.set_title('📈 Monthly Orders Trend', color=TEXT, fontsize=12, fontweight='bold', pad=10)
ax1.grid(axis='y', color='#334155', linewidth=0.5, alpha=0.5)
peak_idx = monthly['count'].idxmax()
ax1.annotate(f"Peak: {monthly.loc[peak_idx,'count']:,}",
             xy=(peak_idx, monthly.loc[peak_idx, 'count']),
             xytext=(peak_idx - 4, monthly.loc[peak_idx, 'count'] + 400),
             color=AMBER, fontsize=9, fontweight='bold',
             arrowprops=dict(arrowstyle='->', color=AMBER))

# 2. Payment Types
ax2 = fig.add_subplot(gs[1, 0])
ax2.set_facecolor(CARD)
ax2.pie(pay_type.values, labels=pay_type.index, autopct='%1.1f%%',
        colors=[PURP, GREEN, AMBER, RED, BLUE],
        startangle=90, textprops={'color': TEXT, 'fontsize': 8})
ax2.set_title('💳 Payment Methods', color=TEXT, fontsize=11, fontweight='bold')

# 3. Review Scores
ax3 = fig.add_subplot(gs[1, 1])
bars = ax3.bar(rev_dist.index, rev_dist.values,
               color=[RED, '#f97316', AMBER, BLUE, GREEN], edgecolor='none', width=0.6)
for bar, val in zip(bars, rev_dist.values):
    ax3.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 300,
             f'{val:,}', ha='center', color=TEXT, fontsize=7)
style(ax3, f'⭐ Review Scores (Avg: {reviews["review_score"].mean():.2f})', 'Score', 'Count')

# 4. Top States
ax4 = fig.add_subplot(gs[1, 2])
ax4.barh(top_states.index[::-1], top_states.values[::-1], color=PURP, edgecolor='none')
style(ax4, '🗺️ Top 10 States by Customers', 'Customers', '')

# 5. Top Categories by Revenue
ax5 = fig.add_subplot(gs[2, :2])
ax5.barh(range(len(top_cats)), top_cats.values / 1e6, color=GREEN, edgecolor='none')
ax5.set_yticks(range(len(top_cats)))
ax5.set_yticklabels(top_cats.index, color=SUB, fontsize=9)
for i, val in enumerate(top_cats.values / 1e6):
    ax5.text(val + 0.01, i, f'R${val:.2f}M', va='center', color=TEXT, fontsize=8)
style(ax5, ' Top 10 Categories by Revenue', 'Revenue (R$ M)', '')

# 6. Delivery Days
ax6 = fig.add_subplot(gs[2, 2])
delivery_data = delivered['delivery_days'].dropna()
ax6.hist(delivery_data, bins=40, color=BLUE, edgecolor='none', alpha=0.85)
ax6.axvline(delivery_data.mean(), color=AMBER, linestyle='--', linewidth=2,
            label=f'Mean: {delivery_data.mean():.1f}d')
ax6.axvline(delivery_data.median(), color=GREEN, linestyle='--', linewidth=2,
            label=f'Median: {delivery_data.median():.0f}d')
style(ax6, ' Delivery Time Distribution', 'Days', 'Count')
ax6.legend(fontsize=8, facecolor=CARD, labelcolor=TEXT)

# 7. Orders by Hour
ax7 = fig.add_subplot(gs[3, 0])
ax7.bar(hour_dist.index, hour_dist.values, color=AMBER, edgecolor='none', width=0.8)
style(ax7, ' Orders by Hour of Day', 'Hour', 'Orders')

# 8. Late vs On-Time
ax8 = fig.add_subplot(gs[3, 1])
late_counts = pd.Series({
    'On Time ': (delivered['is_late'] == 0).sum(),
    'Late ':    (delivered['is_late'] == 1).sum()
})
ax8.bar(late_counts.index, late_counts.values, color=[GREEN, RED], edgecolor='none', width=0.5)
for i, (label, val) in enumerate(late_counts.items()):
    ax8.text(i, val + 500, f'{val:,}\n({val/len(delivered)*100:.1f}%)',
             ha='center', color=TEXT, fontsize=9)
style(ax8, ' On-Time vs Late Deliveries', '', 'Count')

# 9. Payment Value
ax9 = fig.add_subplot(gs[3, 2])
pay_clip = payments['payment_value'].clip(upper=payments['payment_value'].quantile(0.95))
ax9.hist(pay_clip, bins=40, color=PINK, edgecolor='none', alpha=0.85)
ax9.axvline(payments['payment_value'].mean(), color=AMBER, linestyle='--', linewidth=2,
            label=f'Mean: R${payments["payment_value"].mean():.0f}')
style(ax9, ' Payment Value Distribution', 'R$', 'Count')
ax9.legend(fontsize=8, facecolor=CARD, labelcolor=TEXT)

plt.savefig(OUT + 'EDA_Dashboard.png', dpi=150, bbox_inches='tight', facecolor=DARK)
print(f"\n✅ The image has been saved outputs/EDA_Dashboard.png")
print("\n" + "=" * 55)
print("  ✅ EDA :Finshing  02_Feature_Engineering.py")
print("=" * 55)
