import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# Korelační analýza
# Heatmapa korelací mezi klíčovými numerickými proměnnými (Quantity, StockQuantity, délka kampaně apod.)
# --- Load cleaned data ---
sales = pd.read_csv("clean_sales.csv", parse_dates=['Date'])
stock = pd.read_csv("clean_stock.csv", parse_dates=['Date'])
marketing = pd.read_csv("clean_marketing.csv", parse_dates=['StartDate', 'EndDate'])

# --- Prepare daily aggregates ---
daily_sales = sales.groupby(sales['Date'].dt.date)['Quantity'].sum().reset_index(name='TotalSales')
daily_sales['Date'] = pd.to_datetime(daily_sales['Date'])

daily_stock = stock.groupby(stock['Date'].dt.date)['StockQuantity'].sum().reset_index(name='TotalStock')
daily_stock['Date'] = pd.to_datetime(daily_stock['Date'])

# --- Calculate daily active campaign count ---
date_range = pd.date_range(start=min(marketing['StartDate']), end=max(marketing['EndDate']))
campaign_days = pd.DataFrame({'Date': date_range})
campaign_days['ActiveCampaigns'] = campaign_days['Date'].apply(
    lambda d: marketing[(marketing['StartDate'] <= d) & (d <= marketing['EndDate'])].shape[0]
)

# --- Merge all daily metrics ---
daily = daily_sales.merge(daily_stock, on='Date', how='outer').merge(campaign_days, on='Date', how='outer').fillna(0)

# --- Correlation matrix ---
corr = daily[['TotalSales', 'TotalStock', 'ActiveCampaigns']].corr()

# --- Heatmap ---
plt.figure(figsize=(6, 5))
sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm', square=True)
plt.title('Correlation Matrix: Sales, Stock, and Campaigns')
plt.tight_layout()
plt.show()
