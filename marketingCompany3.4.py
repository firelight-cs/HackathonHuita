import pandas as pd
import matplotlib.pyplot as plt

# Dopad marketingových kampaní
#
# Porovnání průměrného počtu prodaných kusů během kampaní vs mimo ně
#
# Grafy "skutečné vs. předpokládané" v období kampaní

# Load cleaned data
sales = pd.read_csv('clean_sales.csv', parse_dates=['Date'])
marketing = pd.read_csv('clean_marketing.csv', parse_dates=['StartDate', 'EndDate'])

# Ensure Quantity is numeric and drop invalid rows
sales['Quantity'] = pd.to_numeric(sales['Quantity'], errors='coerce')
sales = sales.dropna(subset=['Date', 'Quantity']).reset_index(drop=True)

# Add a unique index to sales for grouping after merge
sales['SaleIdx'] = sales.index

# Merge sales with marketing campaigns on Product
merged = sales.merge(
    marketing[['Product', 'StartDate', 'EndDate']],
    on='Product',
    how='left'
)

# Mark if sale is in campaign period
merged['InCampaign'] = (
    (merged['Date'] >= merged['StartDate']) &
    (merged['Date'] <= merged['EndDate'])
)

# For each sale, flag if it was in any campaign
flag = merged.groupby('SaleIdx')['InCampaign'].any()
sales['InCampaign'] = sales['SaleIdx'].map(flag)
sales = sales.drop(columns=['SaleIdx'])

# --- Average quantity per sale: campaign vs non-campaign ---
avg_qty = sales.groupby('InCampaign')['Quantity'].mean().reset_index()
avg_qty['Context'] = avg_qty['InCampaign'].map({True: 'During campaign', False: 'Outside campaign'})

plt.figure()
plt.bar(avg_qty['Context'], avg_qty['Quantity'])
plt.title('Average Quantity per Sale: Campaign vs Non-Campaign')
plt.xlabel('')
plt.ylabel('Average Quantity')
plt.tight_layout()
plt.show()

# --- Daily total sales with campaign periods highlighted ---
daily = sales.groupby(sales['Date'].dt.date)['Quantity'].sum().reset_index(name='TotalSales')
daily['Date'] = pd.to_datetime(daily['Date'])

ranges = marketing[['StartDate', 'EndDate']].drop_duplicates()

plt.figure()
plt.plot(daily['Date'], daily['TotalSales'], label='Daily Sales')
for _, r in ranges.iterrows():
    plt.axvspan(r['StartDate'], r['EndDate'], alpha=0.3, color='orange')
plt.title('Daily Sales with Campaign Periods Highlighted')
plt.xlabel('Date')
plt.ylabel('Total Sales')
plt.legend()
plt.tight_layout()
plt.show()
