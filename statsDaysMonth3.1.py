import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose

#
# Vývoj prodejů v čase
#
# Časové řady denních a měsíčních celkových prodejů
#
# Identifikace trendu, sezónnosti a výkyvů pomocí dekompozice

# --- Загрузить и минимально очистить данные ---
sales = pd.read_csv("clean_sales.csv", sep=',', parse_dates=['Date'])
sales['Quantity'] = pd.to_numeric(sales['Quantity'], errors='coerce')
sales = sales.dropna(subset=['Date', 'Quantity'])

# --- 3.1.1 Суточный тренд продаж ---
daily = sales.groupby(sales['Date'].dt.date)['Quantity'].sum().reset_index()
daily['Date'] = pd.to_datetime(daily['Date'])

plt.figure()
plt.plot(daily['Date'], daily['Quantity'])
plt.title('Daily Sales Trend')
plt.xlabel('Date')
plt.ylabel('Total Quantity Sold')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# --- 3.1.2 Месячный тренд продаж ---
monthly = sales.set_index('Date').resample('M')['Quantity'].sum().reset_index()
monthly.columns = ['Date', 'TotalQuantity']

plt.figure()
plt.plot(monthly['Date'], monthly['TotalQuantity'])
plt.title('Monthly Sales Trend')
plt.xlabel('Month')
plt.ylabel('Total Quantity Sold')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# --- 3.1.3 Декомпозиция месячного ряда ---
ts = monthly.set_index('Date')['TotalQuantity']
decomp = seasonal_decompose(ts, model='additive', period=12)

# Тренд
plt.figure()
decomp.trend.plot()
plt.title('Trend Component (Monthly)')
plt.xlabel('Month')
plt.ylabel('Quantity')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Сезонность
plt.figure()
decomp.seasonal.plot()
plt.title('Seasonal Component (Monthly)')
plt.xlabel('Month')
plt.ylabel('Seasonal Effect')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Остатки
plt.figure()
decomp.resid.plot()
plt.title('Residual Component (Monthly)')
plt.xlabel('Month')
plt.ylabel('Residuals')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
