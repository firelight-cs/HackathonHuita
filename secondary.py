import pandas as pd

sell_df = pd.read_csv('./data/sell_data_cleaned.csv', sep=';')

print(sell_df.groupby('Product').count().sort_values('Quantity', ascending=False).head(10))