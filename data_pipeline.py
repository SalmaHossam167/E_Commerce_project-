import pandas as pd
from sqlalchemy import create_engine
import urllib

# 1. Database Configuration
SERVER = r'.\MSSQLSERVER01'
DATABASE = 'E_Commerce_Project'
DRIVER = 'SQL Server Native Client 11.0'

def get_engine():
    params = urllib.parse.quote_plus(f'DRIVER={{{DRIVER}}};SERVER={SERVER};DATABASE={DATABASE};Trusted_Connection=yes;')
    return create_engine(f"mssql+pyodbc:///?odbc_connect={params}")

def run_full_pipeline():
    engine = get_engine()
    
    try:
        # --- [1] Orders: Dates, Nulls, and Status ---
        print("Processing: Orders...")
        orders = pd.read_csv('olist_orders_dataset.csv')
        # Convert strings to actual datetime objects for AI models
        date_cols = ['order_purchase_timestamp', 'order_approved_at', 'order_delivered_carrier_date', 
                     'order_delivered_customer_date', 'order_estimated_delivery_date']
        for col in date_cols:
            orders[col] = pd.to_datetime(orders[col])
        
        # Handling Nulls by creating a status column (Business Logic)
        orders['delivery_status'] = orders['order_delivered_customer_date'].apply(
            lambda x: 'In Progress' if pd.isna(x) else 'Delivered'
        )
        # Drop rows with critical missing values for AI training
        orders_final = orders.dropna(subset=['order_approved_at'])
        orders_final.to_sql('Final_Orders', con=engine, if_exists='replace', index=False)

        # --- [2] Products: Translation & Null Cleaning ---
        print("Processing: Products...")
        products = pd.read_csv('olist_products_dataset.csv')
        translation = pd.read_csv('product_category_name_translation.csv')
        # Clean null categories and merge translation
        products.dropna(subset=['product_category_name'], inplace=True)
        products_final = pd.merge(products, translation, on='product_category_name', how='left')
        products_final.to_sql('Final_Products', con=engine, if_exists='replace', index=False)

        # --- [3] Customers & Payments: De-duplication ---
        print("Processing: Customers & Payments...")
        # Customers
        customers = pd.read_csv('olist_customers_dataset.csv').drop_duplicates(subset=['customer_id'])
        customers.to_sql('Final_Customers', con=engine, if_exists='replace', index=False)
        
        # Payments
        payments = pd.read_csv('olist_order_payments_dataset.csv').dropna().drop_duplicates()
        payments.to_sql('Final_Payments', con=engine, if_exists='replace', index=False)

        print("\nPipeline Status: SUCCESS. All tables are cleaned and synced to SQL.")

    except Exception as e:
        print(f"Pipeline Status: FAILED. Error: {str(e)}")

if __name__ == "__main__":
    run_full_pipeline()