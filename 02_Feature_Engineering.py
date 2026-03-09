# E-Commerce Project - Step 2: Feature Engineering
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

DATA = 'data/'
OUT  = 'outputs/'

import os
os.makedirs(OUT, exist_ok=True)

print("=" * 55)
print("  STEP 2: Feature Engineering")
print("=" * 55)

# ──Download data──
print("\n the data is being dawnloaded...")
orders    = pd.read_csv(DATA + 'cleaned_orders_dataset.csv')
customers = pd.read_csv(DATA + 'cleaned_customers.csv')
payments  = pd.read_csv(DATA + 'cleaned_payments.csv')
products  = pd.read_csv(DATA + 'cleaned_products_english.csv')
reviews   = pd.read_csv(DATA + 'olist_order_reviews_dataset.csv')
items     = pd.read_csv(DATA + 'olist_order_items_dataset.csv')
print("All files have been uploaded")

# ── Data processing ──
print("\n جاري بناء الـ Features...")
for col in ['order_purchase_timestamp', 'order_approved_at',
            'order_delivered_carrier_date', 'order_delivered_customer_date',
            'order_estimated_delivery_date']:
    orders[col] = pd.to_datetime(orders[col])

orders['delivery_status'] = orders['order_delivered_customer_date'].apply(
    lambda x: 'In Progress' if pd.isna(x) else 'Delivered'
)
orders['delivery_days'] = (
    orders['order_delivered_customer_date'] - orders['order_purchase_timestamp']
).dt.days
orders['approval_days'] = (
    orders['order_approved_at'] - orders['order_purchase_timestamp']
).dt.days
orders['estimated_delivery_days'] = (
    orders['order_estimated_delivery_date'] - orders['order_purchase_timestamp']
).dt.days
orders['is_late']         = (orders['order_delivered_customer_date'] > orders['order_estimated_delivery_date']).astype(float)
orders['purchase_hour']   = orders['order_purchase_timestamp'].dt.hour
orders['purchase_month']  = orders['order_purchase_timestamp'].dt.month
orders['purchase_year']   = orders['order_purchase_timestamp'].dt.year
orders['is_weekend']      = (orders['order_purchase_timestamp'].dt.dayofweek >= 5).astype(int)

# ── Fix Reviews ──
reviews['review_comment_title']   = reviews['review_comment_title'].fillna('No Comment')
reviews['review_comment_message'] = reviews['review_comment_message'].fillna('No Comment')

# ── Payment Features ──
print("  💳 Payment features...")
payment_agg = payments.groupby('order_id').agg(
    total_payment     = ('payment_value', 'sum'),
    max_installments  = ('payment_installments', 'max'),
    num_payment_types = ('payment_type', 'nunique'),
).reset_index()

main_pay = payments.groupby('order_id')['payment_type'].agg(
    lambda x: x.value_counts().index[0]
).reset_index(name='main_payment_type')
payment_agg = payment_agg.merge(main_pay, on='order_id', how='left')
pay_dummies = pd.get_dummies(payment_agg['main_payment_type'], prefix='pay')
payment_agg = pd.concat([payment_agg.drop('main_payment_type', axis=1), pay_dummies], axis=1)

# ── Items + Products Features ──
print("   Items & Products features...")
items_prod = items.merge(
    products[['product_id', 'product_category_name_english']], on='product_id', how='left'
)
items_agg = items_prod.groupby('order_id').agg(
    num_items      = ('order_item_id', 'count'),
    total_price    = ('price', 'sum'),
    avg_price      = ('price', 'mean'),
    total_freight  = ('freight_value', 'sum'),
    num_sellers    = ('seller_id', 'nunique'),
    num_categories = ('product_category_name_english', 'nunique'),
).reset_index()
items_agg['freight_ratio'] = items_agg['total_freight'] / (items_agg['total_price'] + 0.01)

# ── Reviews Features ──
print("   Reviews features...")
rev_agg = reviews.groupby('order_id').agg(
    review_score = ('review_score', 'mean'),
    has_comment  = ('review_comment_message',
                    lambda x: (x != 'No Comment').any().astype(int))
).reset_index()

# ── Merge ──
print("  🔗 جاري دمج كل الـ features...")
master = orders[['order_id', 'customer_id', 'order_status', 'delivery_status',
                  'delivery_days', 'approval_days', 'estimated_delivery_days',
                  'is_late', 'purchase_hour', 'purchase_month',
                  'purchase_year', 'is_weekend']].copy()

master = master.merge(payment_agg, on='order_id', how='left')
master = master.merge(items_agg,   on='order_id', how='left')
master = master.merge(rev_agg,     on='order_id', how='left')
master = master.merge(customers[['customer_id', 'customer_state']], on='customer_id', how='left')

# Encode customer_state
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
master['customer_state_enc'] = le.fit_transform(master['customer_state'].fillna('Unknown'))

# فلتر delivered orders فقط للـ ML
master_ml = master[master['order_status'] == 'delivered'].dropna(subset=['is_late', 'delivery_days'])

# ── save──
master_ml.to_csv(OUT + 'master_features.csv', index=False)
orders.to_csv(OUT + 'final_orders_complete.csv', index=False)
reviews.to_csv(OUT + 'final_reviews_clean.csv', index=False)

print(f"\n✅ master_features.csv → {master_ml.shape[0]:,} orders, {master_ml.shape[1]} features")
print(f"✅ final_orders_complete.csv → saved")
print(f"✅ final_reviews_clean.csv → saved")
print(f"\nLate Deliveries: {master_ml['is_late'].sum():.0f} ({master_ml['is_late'].mean()*100:.1f}%)")

print("\n" + "=" * 55)
print("  ✅ Feature Engineering  :Finish.. 03_ML_Model.py")
print("=" * 55)
