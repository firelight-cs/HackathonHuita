import pandas as pd

# --- 1. Define file paths (adjust if needed) ---
paths = {
    "sales": "data/sell_data.csv",
    "stock": "data/stock.csv",
    "marketing": "data/marketing_campaign.csv",
    "products": "data/products.csv"
}

# --- 2. Load & clean sales data ---
sales = pd.read_csv(
    paths["sales"],
    sep=';',
    parse_dates=['Date'],
    dtype={
        'ID': int,
        'Product': str,
        'Quantity': float,
        'Country': str,
        'CountryStatus': str,
        'ProductStatus': str
    }
)
sales.columns = sales.columns.str.strip()
sales['Product'] = sales['Product'].str.strip()
sales['Country'] = sales['Country'].str.strip()
sales = sales.drop_duplicates().dropna(subset=['ID', 'Product', 'Date', 'Quantity'])

# --- 3. Load & clean stock data ---
stock = pd.read_csv(
    paths["stock"],
    sep=';',
    parse_dates=['Datum'],
    dtype={'AnonymniProdukt': str, 'Mnozstvi': float}
)
stock.rename(columns={
    'Datum': 'Date',
    'AnonymniProdukt': 'Product',
    'Mnozstvi': 'StockQuantity'
}, inplace=True)
stock['Product'] = stock['Product'].str.strip()
stock = stock.drop_duplicates().dropna(subset=['Date', 'Product', 'StockQuantity'])

# --- 4. Load & clean marketing campaign data ---
marketing = pd.read_csv(
    paths["marketing"],
    sep=';',
    parse_dates=['ValidFrom', 'ValidTo'],
    dtype={'IdCampaign': int, 'Product': str, 'Country': str}
)
marketing.rename(columns={
    'ValidFrom': 'StartDate',
    'ValidTo': 'EndDate'
}, inplace=True)
marketing['Product'] = marketing['Product'].str.strip()
marketing['Country'] = marketing['Country'].str.strip()
marketing = marketing.drop_duplicates().dropna(subset=['IdCampaign', 'StartDate', 'EndDate', 'Product'])

# --- 5. Load & clean products data ---
products = pd.read_csv(
    paths["products"],
    sep=';',
    dtype={'Product': str, 'Commodity': str}
)
products['Product'] = products['Product'].str.strip()
products['Commodity'] = products['Commodity'].str.strip()
products = products.drop_duplicates().dropna(subset=['Product', 'Commodity'])

# --- 6. (Optional) Save cleaned files ---
sales.to_csv("clean_sales.csv", index=False)
stock.to_csv("clean_stock.csv", index=False)
marketing.to_csv("clean_marketing.csv", index=False)
products.to_csv("clean_products.csv", index=False)

# --- 7. Quick preview ---
print("Sales data sample:")
print(sales.head(3), "\n")
print("Stock data sample:")
print(stock.head(3), "\n")
print("Marketing data sample:")
print(marketing.head(3), "\n")
print("Products data sample:")
print(products.head(3))