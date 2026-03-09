
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.preprocessing import LabelEncoder
import joblib

import os
DATA = 'data/'
OUT  = 'outputs/'
os.makedirs(OUT, exist_ok=True)

print("=" * 55)
print("  STEP 4: Advanced Analysis")
print("=" * 55)

# ── data downloaded──
print("\n The data is being downloaded....")
items     = pd.read_csv(DATA + 'olist_order_items_dataset.csv')
products  = pd.read_csv(DATA + 'cleaned_products_english.csv')
customers = pd.read_csv(DATA + 'olist_customers_dataset.csv')   # ده بيحتوي customer_unique_id
orders    = pd.read_csv(OUT  + 'final_orders_complete.csv')     # من Step 2
print("All files have been uploaded...")

# ── Data processing ──
orders['order_purchase_timestamp'] = pd.to_datetime(orders['order_purchase_timestamp'])
orders['purchase_hour'] = orders['order_purchase_timestamp'].dt.hour
orders['purchase_dow']  = orders['order_purchase_timestamp'].dt.day_name()

items_prod = items.merge(
    products[['product_id', 'product_category_name_english']], on='product_id', how='left'
)

# ============================================================
# ANALYSIS 1: TOP PRODUCTS
# ============================================================
print("\n Analysis 1: Top Products...")

cat_analysis = items_prod.groupby('product_category_name_english').agg(
    units_sold    = ('order_item_id', 'count'),
    total_revenue = ('price', 'sum'),
    avg_price     = ('price', 'mean')
).sort_values('total_revenue', ascending=False)

top10_rev   = cat_analysis.head(10)
top10_units = cat_analysis.sort_values('units_sold', ascending=False).head(10)

print(f"\n   Top Category by Revenue: {top10_rev.index[0]} (R${top10_rev['total_revenue'].iloc[0]:,.0f})")
print(f"  Top Category by Units:   {top10_units.index[0]} ({top10_units['units_sold'].iloc[0]:,} units)")
print(f"   Most Expensive Category: {cat_analysis['avg_price'].idxmax()} (avg R${cat_analysis['avg_price'].max():.2f})")

# ============================================================
# ANALYSIS 2: PEAK BUYING TIMES
# ============================================================
print("\n Analysis 2: Peak Buying Times...")

hour_orders = orders.groupby('purchase_hour').size()
dow_orders  = orders.groupby('purchase_dow').size().reindex(
    ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
)

peak_hour = hour_orders.idxmax()
peak_day  = dow_orders.idxmax()

print(f"\n   Peak Hour: {peak_hour}:00 ({hour_orders.max():,} orders)")
print(f"  Peak Day:  {peak_day} ({dow_orders.max():,} orders)")
print(f"  Slowest Day: {dow_orders.idxmin()} ({dow_orders.min():,} orders)")

# ============================================================
# ANALYSIS 3: CUSTOMER RETENTION
# ============================================================
print("\n Analysis 3: Customer Retention...")

# نربط كل أوردر بـ customer_unique_id
cust_orders = orders.merge(
    customers[['customer_id', 'customer_unique_id']], on='customer_id', how='left'
)
repeat = cust_orders.groupby('customer_unique_id').size()

total_unique   = repeat.shape[0]
bought_once    = (repeat == 1).sum()
bought_twice   = (repeat == 2).sum()
bought_3plus   = (repeat >= 3).sum()
max_orders     = repeat.max()
top_customer   = repeat.idxmax()

print(f"\n   Total Unique Customers: {total_unique:,}")
print(f"  1️⃣  Bought Once:     {bought_once:,} ({bought_once/total_unique*100:.1f}%)")
print(f"  2️⃣  Bought 2 Times:  {bought_twice:,} ({bought_twice/total_unique*100:.1f}%)")
print(f"  🔄 Bought 3+ Times: {bought_3plus:,} ({bought_3plus/total_unique*100:.1f}%)")
print(f"   Max by 1 Customer: {max_orders} orders!")
print(f"\n   Insight: {bought_once/total_unique*100:.1f}% of customers never return!")
print(f"   Opportunity: Loyalty program could boost revenue significantly")

# ============================================================
# ANALYSIS 4: PRICE PREDICTION MODEL
# ============================================================
print("\n Analysis 4: Price Prediction...")

# دمج items مع products للحصول على features
items_full = items.merge(products, on='product_id', how='left')
items_full = items_full.dropna(subset=['product_weight_g', 'product_category_name_english'])

# Encode categories
le = LabelEncoder()
items_full['cat_enc'] = le.fit_transform(items_full['product_category_name_english'])

# Features للموديل
feature_cols = [
    'product_weight_g',
    'product_length_cm',
    'product_height_cm',
    'product_width_cm',
    'freight_value',
    'cat_enc'
]

# إزالة outliers (أعلى 1%)
df_price = items_full[feature_cols + ['price']].dropna()
df_price = df_price[df_price['price'] < df_price['price'].quantile(0.99)]

X = df_price[feature_cols]
y = df_price['price']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"\n  📊 Training Data: {len(X_train):,} products")
print(f"  💵 Price Range: R${y.min():.2f} - R${y.max():.2f}")
print(f"  💵 Avg Price:   R${y.mean():.2f} | Median: R${y.median():.2f}")

# تدريب 3 Models
price_models = {
    'Linear Regression': LinearRegression(),
    'Random Forest':     RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
    'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
}

price_results = {}
print("\n   Training price models...")
for name, model in price_models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    r2  = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    price_results[name] = {'model': model, 'r2': r2, 'mae': mae, 'y_pred': y_pred}
    print(f"    [{name}] R²={r2:.4f} | MAE=R${mae:.2f}")

best_price_model = max(price_results, key=lambda k: price_results[k]['r2'])
print(f"\n   Best: {best_price_model} (R²={price_results[best_price_model]['r2']:.4f})")
print(f"  → بيتنبأ بالسعر بدقة ±R${price_results[best_price_model]['mae']:.2f} في المتوسط")

# Feature importance
rf = price_results['Random Forest']['model']
feat_imp = pd.Series(rf.feature_importances_, index=feature_cols).sort_values(ascending=False)
print(f"\n  : {feat_imp.idxmax()} ({feat_imp.max()*100:.1f}%)")

# ============================================================
# PLOTS - FIGURE 1: Products & Time
# ============================================================
print("\n🎨 بيتم رسم التحليلات...")

DARK='#0f172a'; CARD='#1e293b'; TEXT='#e2e8f0'; SUB='#94a3b8'
PURP='#6366f1'; GREEN='#10b981'; AMBER='#f59e0b'; RED='#ef4444'
BLUE='#3b82f6'; PINK='#ec4899'

def style(ax, title, xlabel='', ylabel=''):
    ax.set_facecolor(CARD)
    ax.set_title(title, color=TEXT, fontsize=11, fontweight='bold', pad=10)
    ax.tick_params(colors=SUB, labelsize=8)
    ax.set_xlabel(xlabel, color=SUB, fontsize=8)
    ax.set_ylabel(ylabel, color=SUB, fontsize=8)
    for sp in ax.spines.values():
        sp.set_edgecolor('#334155')
    ax.grid(axis='y', color='#334155', linewidth=0.5, alpha=0.5)

# Figure 1: Products & Time
fig1 = plt.figure(figsize=(20, 14), facecolor=DARK)
gs1  = gridspec.GridSpec(2, 2, figure=fig1, hspace=0.45, wspace=0.32,
                          top=0.91, bottom=0.06, left=0.06, right=0.97)
fig1.text(0.5, 0.96, ' Product & Time Analysis', ha='center',
          color=TEXT, fontsize=15, fontweight='bold')

# 1a. Top 10 by Revenue
ax1 = fig1.add_subplot(gs1[0, 0])
colors = [PURP if i == 0 else GREEN if i == 1 else BLUE for i in range(10)]
ax1.barh(range(10), top10_rev['total_revenue'].values[::-1] / 1e6,
         color=colors[::-1], edgecolor='none')
ax1.set_yticks(range(10))
ax1.set_yticklabels([c[:22] for c in top10_rev.index[::-1]], color=SUB, fontsize=8)
for i, val in enumerate(top10_rev['total_revenue'].values[::-1] / 1e6):
    ax1.text(val + 0.01, i, f'R${val:.2f}M', va='center', color=TEXT, fontsize=8)
style(ax1, ' Top 10 Categories by Revenue', 'Revenue (R$ M)', '')
ax1.grid(axis='x', color='#334155', linewidth=0.5, alpha=0.5)
ax1.grid(axis='y', color='none')

# 1b. Top 10 by Units Sold
ax2 = fig1.add_subplot(gs1[0, 1])
ax2.barh(range(10), top10_units['units_sold'].values[::-1], color=AMBER, edgecolor='none')
ax2.set_yticks(range(10))
ax2.set_yticklabels([c[:22] for c in top10_units.index[::-1]], color=SUB, fontsize=8)
for i, val in enumerate(top10_units['units_sold'].values[::-1]):
    ax2.text(val + 50, i, f'{val:,}', va='center', color=TEXT, fontsize=8)
style(ax2, ' Top 10 Categories by Units Sold', 'Units Sold', '')
ax2.grid(axis='x', color='#334155', linewidth=0.5, alpha=0.5)
ax2.grid(axis='y', color='none')

# 1c. Orders by Hour
ax3 = fig1.add_subplot(gs1[1, 0])
bar_colors_h = [RED if h == peak_hour else BLUE for h in hour_orders.index]
ax3.bar(hour_orders.index, hour_orders.values, color=bar_colors_h, edgecolor='none', width=0.8)
ax3.axvline(peak_hour, color=RED, linestyle='--', alpha=0.5, linewidth=1.5)
ax3.text(peak_hour + 0.3, hour_orders.max() * 0.95,
         f'Peak: {peak_hour}:00', color=RED, fontsize=9, fontweight='bold')
style(ax3, ' Orders by Hour of Day', 'Hour (24h)', 'Orders')

# 1d. Orders by Day
ax4 = fig1.add_subplot(gs1[1, 1])
dow_colors = [GREEN if d == peak_day else PURP for d in dow_orders.index]
ax4.bar(range(7), dow_orders.values, color=dow_colors, edgecolor='none', width=0.7)
ax4.set_xticks(range(7))
ax4.set_xticklabels(['Mon','Tue','Wed','Thu','Fri','Sat','Sun'], color=SUB, fontsize=9)
for i, val in enumerate(dow_orders.values):
    ax4.text(i, val + 100, f'{val:,}', ha='center', color=TEXT, fontsize=7.5)
style(ax4, ' Orders by Day of Week', '', 'Orders')

plt.savefig(OUT + 'Analysis_Products_Time.png', dpi=150, bbox_inches='tight', facecolor=DARK)
print(f"  Saved: outputs/Analysis_Products_Time.png")

# ============================================================
# PLOTS - FIGURE 2: Retention & Price Prediction
# ============================================================
fig2 = plt.figure(figsize=(20, 14), facecolor=DARK)
gs2  = gridspec.GridSpec(2, 2, figure=fig2, hspace=0.45, wspace=0.35,
                          top=0.91, bottom=0.06, left=0.06, right=0.97)
fig2.text(0.5, 0.96, '👥 Customer Retention &  Price Prediction',
          ha='center', color=TEXT, fontsize=15, fontweight='bold')

# 2a. Retention Donut
ax5 = fig2.add_subplot(gs2[0, 0])
sizes  = [bought_once, total_unique - bought_once]
labels = [f'Bought Once\n{bought_once:,}', f'Repeat Customers\n{total_unique-bought_once:,}']
wedges, texts, autotexts = ax5.pie(
    sizes, labels=labels, autopct='%1.1f%%',
    colors=[BLUE, GREEN], startangle=90,
    wedgeprops={'width': 0.55},
    textprops={'color': TEXT, 'fontsize': 10}
)
for at in autotexts:
    at.set_color(DARK); at.set_fontsize(10); at.set_fontweight('bold')
ax5.set_title(' Customer Retention Rate', color=TEXT, fontsize=11, fontweight='bold', pad=10)
ax5.set_facecolor(CARD)
ax5.text(0, 0, f'{total_unique:,}\nCustomers', ha='center', va='center',
         color=TEXT, fontsize=10, fontweight='bold')

# 2b. Purchase Frequency
ax6 = fig2.add_subplot(gs2[0, 1])
rpt_vals  = repeat.value_counts().sort_index().head(6)
bar_c_rpt = [GREEN if i == 0 else AMBER if i == 1 else PURP for i in range(len(rpt_vals))]
ax6.bar(rpt_vals.index, rpt_vals.values, color=bar_c_rpt, edgecolor='none', width=0.6)
for x_val, y_val in zip(rpt_vals.index, rpt_vals.values):
    ax6.text(x_val, y_val + 300, f'{y_val:,}', ha='center', color=TEXT, fontsize=9)
ax6.set_xticks(rpt_vals.index)
ax6.set_xticklabels([f'{x}x' for x in rpt_vals.index], color=SUB)
style(ax6, ' How Many Times Did Customers Buy?', 'Number of Orders', 'Customers')

# 2c. Actual vs Predicted Price
ax7 = fig2.add_subplot(gs2[1, 0])
best_preds = price_results[best_price_model]['y_pred']
sample_idx = np.random.choice(len(y_test), 500, replace=False)
ax7.scatter(y_test.values[sample_idx], best_preds[sample_idx],
            color=PURP, alpha=0.5, s=15, edgecolors='none')
max_val = min(float(y_test.max()), 600)
ax7.plot([0, max_val], [0, max_val], color=AMBER, linewidth=2,
         linestyle='--', label='Perfect Prediction')
style(ax7,
      f' Price: Actual vs Predicted\n({best_price_model} | R²={price_results[best_price_model]["r2"]:.3f})',
      'Actual Price (R$)', 'Predicted Price (R$)')
ax7.legend(fontsize=9, facecolor=CARD, labelcolor=TEXT)
ax7.set_xlim(0, max_val); ax7.set_ylim(0, max_val)

# 2d. Model Comparison
ax8 = fig2.add_subplot(gs2[1, 1])
model_names_p = list(price_results.keys())
r2_vals  = [price_results[m]['r2']  for m in model_names_p]
mae_vals = [price_results[m]['mae'] for m in model_names_p]
x_pos = np.arange(3)
bars_plot = ax8.bar(x_pos, r2_vals, 0.5,
                     color=[PURP, AMBER, GREEN], edgecolor='none')
ax8.set_xticks(x_pos)
ax8.set_xticklabels(['Linear\nRegression','Gradient\nBoosting','Random\nForest'],
                     color=SUB, fontsize=9)
for bar, r2v, maev in zip(bars_plot, r2_vals, mae_vals):
    ax8.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
             f'R²={r2v:.3f}\nMAE=R${maev:.0f}',
             ha='center', color=TEXT, fontsize=8.5)
ax8.set_ylim(0, max(r2_vals) * 1.3)
style(ax8, ' Price Prediction Models Comparison', '', 'R² Score (Higher = Better)')

plt.savefig(OUT + 'Analysis_Retention_Price.png', dpi=150, bbox_inches='tight', facecolor=DARK)
print(f"   Saved: outputs/Analysis_Retention_Price.png")

# ── Saved Model──
joblib.dump(rf, OUT + 'price_model.pkl')
joblib.dump(le, OUT + 'price_label_encoder.pkl')
print(f"   Saved: outputs/price_model.pkl")


print("\n" + "=" * 55)
print("   ملخص كل التحليلات")
print("=" * 55)
print(f"\n PRODUCTS:")
print(f"  - Top Revenue:  {top10_rev.index[0]} (R${top10_rev['total_revenue'].iloc[0]/1e6:.2f}M)")
print(f"  - Top Units:    {top10_units.index[0]} ({top10_units['units_sold'].iloc[0]:,} units)")

print(f"\n PEAK TIMES:")
print(f"  - Best Hour: {peak_hour}:00 | Best Day: {peak_day}")

print(f"\n RETENTION:")
print(f"  - {bought_once/total_unique*100:.1f}% bought only once")
print(f"  - {(total_unique-bought_once)/total_unique*100:.1f}% are repeat customers")

print(f"\n PRICE PREDICTION:")
print(f"  - Best Model: {best_price_model}")
print(f"  - R² = {price_results[best_price_model]['r2']:.4f}")
print(f"  - MAE = R${price_results[best_price_model]['mae']:.2f}")

print("\n" + "=" * 55)
print("  ✅ Analysis in :outputs/")
print("=" * 55)
