import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Dict, Union


def read_data(filepath):
    data = pd.read_csv(filepath, delimiter=';', index_col='Date', parse_dates=True, decimal=".")
    return data

def sort_by_quantity(df: pd.DataFrame, order: str = "descend") -> pd.DataFrame:
    if order == "ascend":
        df_sorted = df.sort_values(by='Quantity', ascending=True)
    elif order == "descend":
        df_sorted = df.sort_values(by='Quantity', ascending=False)
    else:
        raise ValueError("Order must be 'ascend' or 'descend'")

    return df_sorted

def filter_by_product(data, product_name) -> pd.DataFrame:
    df_filtered = data[data['Product'] == product_name].copy()
    return df_filtered

def sort_by_date(data, start_date, end_date) -> pd.DataFrame:
    df_filtered = data[(data.index >= start_date) & (data.index <= end_date)].copy()
    return df_filtered

def remove_columns(data, columns: List[str]) -> pd.DataFrame:
    for column in columns:
        if column in data.columns:
            data.drop(columns=column, inplace=True)
        else:
            print(f"Column {column} not found in DataFrame.")
    return data

def filter_by_country(data, countries: List[str]) -> pd.DataFrame:
    for country in countries:
        if country in data['Country'].values:
            data = data[data['Country'] == country]
        else:
            print(f"Country {country} not found in DataFrame.")
    return data


def main():
    df = read_data("data/sell_data_cleaned.csv")

    data = filter_by_product(df, "ProduktX2475")
    data = sort_by_quantity(data, order="descend")
    remove_columns_table = ["Date", "ID", "CountryStatus", "ProductStatus"]
    country_names = ["PL"]
    data = remove_columns(data, remove_columns_table)
    data = filter_by_country(data, country_names)
    print(data.head(100))

if __name__ == "__main__":
    main()





# def get_dates_from_data(data):
#     def split_data(data, nomenclature):
#         max_date, previous_year_max_date = get_dates_from_data(data)
#         train_data = data[data.nomenclature == nomenclature]['Quantity']
#         test_data = train_data[:]
#         train_data = train_data[:previous_year_max_date]
#
#         train_data = train_data.resample('MS').sum().sort_index()
#         test_data = test_data.resample('MS').sum().sort_index()
#         test_data = test_data[:max_date]
#
#         return train_data, test_data
