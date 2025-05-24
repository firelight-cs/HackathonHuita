import pandas as pd
import matplotlib.pyplot as plt

# 1. Загрузка очищенных данных (здесь – clean_sales.csv)
sales = pd.read_csv("clean_sales.csv", parse_dates=['Date'])

# 2. Ежедневная агрегация по продуктам
daily_prod = (
    sales
    .groupby(['Product', sales['Date'].dt.date])['Quantity']
    .sum()
    .reset_index(name='DailySales')
)
daily_prod['Date'] = pd.to_datetime(daily_prod['Date'])

# 3. Сортировка и установка индекса
daily_prod = daily_prod.sort_values(['Product', 'Date']).set_index('Date')

# 4. Вычисление временных фич
def add_ts_features(group):
    group = group.copy()
    group['Lag_1']   = group['DailySales'].shift(1)
    group['Lag_7']   = group['DailySales'].shift(7)
    group['Lag_30']  = group['DailySales'].shift(30)
    group['RollMean_7']  = group['DailySales'].rolling(7).mean()
    group['RollMean_30'] = group['DailySales'].rolling(30).mean()
    group['RollStd_7']   = group['DailySales'].rolling(7).std()
    group['PctChange_1'] = group['DailySales'].pct_change(1)
    return group

features = (
    daily_prod
    .groupby('Product', group_keys=False)
    .apply(add_ts_features)
    .reset_index()
)

# 5. Визуализация (пример для первого продукта)
sample_product = features['Product'].unique()[0]
df = features[features['Product'] == sample_product].set_index('Date')

# График: продажи и скользящие средние
plt.figure()
plt.plot(df.index, df['DailySales'], label='DailySales')
plt.plot(df.index, df['RollMean_7'], label='7-day Rolling Mean')
plt.title(f'Sales and Rolling Mean for {sample_product}')
plt.xlabel('Date')
plt.ylabel('Quantity')
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# График: лаги
plt.figure()
plt.plot(df.index, df['Lag_1'], label='Lag 1')
plt.plot(df.index, df['Lag_7'], label='Lag 7')
plt.title(f'Lag Features for {sample_product}')
plt.xlabel('Date')
plt.ylabel('Quantity')
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
