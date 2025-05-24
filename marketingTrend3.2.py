import pandas as pd
import matplotlib.pyplot as plt
#
# Sezónní a cyklické vzory
#
# Boxplot pro prodeje podle dnů v týdnu
#
# Průměrné prodeje po měsících a kvartálech

# --- Load cleaned sales data ---
sales = pd.read_csv("clean_sales.csv", parse_dates=['Date'])

# Ensure Quantity is numeric
sales['Quantity'] = pd.to_numeric(sales['Quantity'], errors='coerce')
sales = sales.dropna(subset=['Date', 'Quantity'])

# Add temporal features
sales['DayOfWeek'] = sales['Date'].dt.day_name()
sales['Month'] = sales['Date'].dt.month
sales['Quarter'] = sales['Date'].dt.to_period('Q')

# --- 3.2.1 Boxplot of sales by day of week ---
box_data = [sales[sales['DayOfWeek'] == day]['Quantity']
            for day in ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']]

plt.figure()
plt.boxplot(box_data, labels=['Mon','Tue','Wed','Thu','Fri','Sat','Sun'])
plt.title('Sales Distribution by Day of Week')
plt.xlabel('Day of Week')
plt.ylabel('Quantity Sold')
plt.tight_layout()
plt.show()

# --- 3.2.2 Average sales by month ---
monthly_avg = sales.groupby('Month')['Quantity'].mean()

plt.figure()
plt.plot(monthly_avg.index, monthly_avg.values, marker='o')
plt.title('Average Daily Sales by Month')
plt.xlabel('Month')
plt.ylabel('Average Quantity Sold')
plt.xticks(range(1,13))
plt.tight_layout()
plt.show()

# --- 3.2.3 Average sales by quarter ---
quarter_avg = sales.groupby('Quarter')['Quantity'].mean().reset_index()
quarter_avg['QuarterStr'] = quarter_avg['Quarter'].astype(str)

plt.figure()
plt.bar(quarter_avg['QuarterStr'], quarter_avg['Quantity'])
plt.title('Average Daily Sales by Quarter')
plt.xlabel('Quarter')
plt.ylabel('Average Quantity Sold')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
