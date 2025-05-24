import pandas as pd
import matplotlib.pyplot as plt

# --- 1. Load cleaned data ---
sales = pd.read_csv("clean_sales.csv", parse_dates=['Date'])
products = pd.read_csv("clean_products.csv")

# --- 2. Preparation ---
sales['Quantity'] = pd.to_numeric(sales['Quantity'], errors='coerce')
sales = sales.dropna(subset=['Product', 'Date', 'Quantity'])
products['Product'] = products['Product'].astype(str).str.strip()
products['Commodity'] = products['Commodity'].astype(str).str.strip()

# --- 3. Merge sales with product metadata ---
sales_prod = sales.merge(products, on='Product', how='left')

# --- Clean Commodity and Country columns ---
sales_prod['Commodity'] = sales_prod['Commodity'].astype(str).str.strip()
sales_prod = sales_prod[sales_prod['Commodity'].notna() & (sales_prod['Commodity'] != '')]

sales['Country'] = sales['Country'].astype(str).str.strip()
sales = sales[sales['Country'].notna() & (sales['Country'] != '')]

# --- 3.6.1 Top 10 Commodities by Sales ---
commodity_df = (
    sales_prod
    .groupby('Commodity', as_index=False)['Quantity']
    .sum()
    .sort_values(by='Quantity', ascending=False)
)
top_commodities = commodity_df.head(10)

plt.figure()
plt.bar(top_commodities['Commodity'], top_commodities['Quantity'])
plt.title('Top Commodities by Total Sales')
plt.xlabel('Commodity')
plt.ylabel('Total Quantity Sold')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# --- 3.6.2 Top 10 Countries by Sales ---
country_df = (
    sales
    .groupby('Country', as_index=False)['Quantity']
    .sum()
    .sort_values(by='Quantity', ascending=False)
)
top_countries = country_df.head(10)

plt.figure()
plt.bar(top_countries['Country'], top_countries['Quantity'])
plt.title('Top Countries by Total Sales')
plt.xlabel('Country')
plt.ylabel('Total Quantity Sold')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# --- 3.6.3 Top 10 Products by Sales ---
product_df = (
    sales_prod
    .groupby('Product', as_index=False)['Quantity']
    .sum()
    .sort_values(by='Quantity', ascending=False)
)
top_products = product_df.head(10)

plt.figure()
plt.bar(top_products['Product'], top_products['Quantity'])
plt.title('Top Products by Total Sales')
plt.xlabel('Product')
plt.ylabel('Total Quantity Sold')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
