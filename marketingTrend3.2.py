import pandas as pd
import matplotlib.pyplot as plt

# Update paths to /mnt/data
sales = pd.read_csv("clean_sales.csv", sep=',', parse_dates=['Date'])
marketing = pd.read_csv("clean_marketing.csv", sep=',', parse_dates=['StartDate', 'EndDate'])
sales = sales.dropna(subset=['Date', 'Quantity'])

# Merge sales and marketing on Product
merged = sales.merge(marketing[['Product', 'StartDate', 'EndDate']], on='Product', how='left')

# Check if sale date is within any campaign period
in_campaign = (merged['Date'] >= merged['StartDate']) & (merged['Date'] <= merged['EndDate'])
sales['InCampaign'] = merged.groupby(['ID'])['Date'].transform(
    lambda x: in_campaign[merged['ID'] == x.iloc[0]].any()
)

# Compare avg quantity per sale
avg_sale = sales.groupby('InCampaign')['Quantity'].mean().reset_index()
avg_sale['InCampaign'] = avg_sale['InCampaign'].map({True: 'During Campaign', False: 'Outside Campaign'})

plt.figure()
plt.bar(avg_sale['InCampaign'], avg_sale['Quantity'])
plt.title('Average Quantity per Sale: Campaign vs Non-Campaign')
plt.xlabel('Context')
plt.ylabel('Avg Quantity')
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()

# Daily total sales with shaded campaigns
daily = sales.groupby(sales['Date'].dt.date)['Quantity'].sum().reset_index()
daily['Date'] = pd.to_datetime(daily['Date'])
ranges = marketing[['StartDate', 'EndDate']].drop_duplicates()

plt.figure()
plt.plot(daily['Date'], daily['Quantity'], label='Daily Sales')
for _, r in ranges.iterrows():
    plt.axvspan(r['StartDate'], r['EndDate'], color='orange', alpha=0.3)
plt.title('Daily Sales with Campaign Periods')
plt.xlabel('Date')
plt.ylabel('Total Quantity')
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
