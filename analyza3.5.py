import pandas as pd
import matplotlib.pyplot as plt


# Analýza situací, kdy nízké zásoby vedly k výpadkům prodejů
#
# Scatter plot zásoby vs. prodané množství

# --- Load cleaned data ---
sales = pd.read_csv("clean_sales.csv", parse_dates=['Date'])
stock = pd.read_csv("clean_stock.csv", parse_dates=['Date'])

# Ensure numeric
sales['Quantity'] = pd.to_numeric(sales['Quantity'], errors='coerce')
stock['StockQuantity'] = pd.to_numeric(stock['StockQuantity'], errors='coerce')

# Drop missing
sales = sales.dropna(subset=['Date', 'Quantity'])
stock = stock.dropna(subset=['Date', 'StockQuantity'])

# --- 3.5.1 Daily aggregates ---
daily_sales = sales.groupby(sales['Date'].dt.date)['Quantity'].sum().reset_index(name='TotalSales')
daily_sales['Date'] = pd.to_datetime(daily_sales['Date'])

daily_stock = stock.groupby(stock['Date'].dt.date)['StockQuantity'].sum().reset_index(name='TotalStock')
daily_stock['Date'] = pd.to_datetime(daily_stock['Date'])

# Merge daily metrics
daily = pd.merge(daily_sales, daily_stock, on='Date', how='inner')

# --- 3.5.2 Scatter plot: Stock vs Sales ---
plt.figure()
plt.scatter(daily['TotalStock'], daily['TotalSales'], alpha=0.6)
plt.title('Scatter: Daily Stock vs Sales')
plt.xlabel('Total Stock')
plt.ylabel('Total Sales')
plt.tight_layout()
plt.show()

# --- 3.5.3 Identify and plot low-stock days with sales drops ---
threshold = daily['TotalStock'].quantile(0.1)  # bottom 10% stock as low-stock threshold
low_stock = daily[daily['TotalStock'] <= threshold]

plt.figure()
plt.scatter(daily['Date'], daily['TotalSales'], label='Normal', alpha=0.4)
plt.scatter(low_stock['Date'], low_stock['TotalSales'], color='red', label='Low Stock Days')
plt.title('Daily Sales with Low-Stock Days Highlighted')
plt.xlabel('Date')
plt.ylabel('Total Sales')
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
