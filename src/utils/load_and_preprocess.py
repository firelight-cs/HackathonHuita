import pandas as pd
from sklearn.preprocessing import LabelEncoder
import numpy as np

def load_and_preprocess(sales_path, campaign_path, stock_path):

    '''
    This function loads and preprocesses sales, campaign, and stock data from CSV files.

    Steps performed:
    1. Load sales, campaign, and stock data from CSV files.
    2. Standardize column names for consistency.
    3. Merge sales and stock data on ['Date', 'Product'].
    4. Expand campaign date ranges into individual daily rows.
    5. Merge campaign information into the main dataset and create an 'ActiveCampaign' binary indicator.

    input:
        sales_path : str
            Path to the CSV file containing sales data.
        campaign_path : str
            Path to the CSV file containing campaign data with date ranges.
        stock_path : str
            Path to the CSV file containing stock data.

    output:
        merged_df : pandas DataFrame
            A preprocessed dataset with columns from sales, stock, and campaign sources,
            including a binary 'ActiveCampaign' column indicating campaign activity.
    '''
    
    # Load data
    sales_df = pd.read_csv(sales_path, sep=';', parse_dates=['Date'])
    campaign_df = pd.read_csv(campaign_path, sep=';', parse_dates=['ValidFrom', 'ValidTo'])
    stock_df = pd.read_csv(stock_path, sep=';', parse_dates=['Datum'])
    
    # Rename columns
    stock_df.rename(columns={'Datum': 'Date', 'AnonymniProdukt': 'Product', 'Mnozstvi': 'Stock'}, inplace=True)
    
    # Merge sales with stock data
    merged_df = pd.merge(sales_df, stock_df, on=['Date', 'Product'], how='left')
    
    # Handle missing stock: assume missing means zero stock
    merged_df['Stock'] = merged_df['Stock'].fillna(0)
    
    # Process campaigns to create ActiveCampaign feature
    campaign_dates = []
    for _, row in campaign_df.iterrows():
        dates = pd.date_range(row['ValidFrom'], row['ValidTo'], freq='D')
        temp_df = pd.DataFrame({
            'Date': dates,
            'Product': row['Product'],
            'Country': row['Country'],
            'ActiveCampaign': 1
        })
        campaign_dates.append(temp_df)
    
    campaign_dates_df = pd.concat(campaign_dates, ignore_index=True)
    merged_df = pd.merge(merged_df, campaign_dates_df, on=['Date', 'Product', 'Country'], how='left')
    merged_df['ActiveCampaign'] = merged_df['ActiveCampaign'].fillna(0)
    
    return merged_df

def missing_data(input_data):
    '''
    This function returns dataframe with information about the percentage of nulls in each column and the column data type.
    
    input: pandas df
    output: pandas df
    '''
    
    total = input_data.isnull().sum()
    percent = (input_data.isnull().sum()/input_data.isnull().count()*100)
    table = pd.concat([total, percent], axis = 1, keys = ['Total', 'Percent'])
    types = []
    for col in input_data.columns: 
        dtype = str(input_data[col].dtype)
        types.append(dtype)
    table["Types"] = types
    return(pd.DataFrame(table))


def split_train_test(df, date_col='Date', test_size=0.2):

    '''
    This function splits a DataFrame into training and testing sets based on chronological order.

    input: 
        df : pandas DataFrame
            The input dataset to be split.
        date_col : str (default='Date')
            The name of the column containing datetime values used for sorting.
        test_size : float (default=0.2)
            The proportion of the dataset to be used as the test set.

    output: 
        train : pandas DataFrame
            The training portion of the dataset.
        test : pandas DataFrame
            The testing portion of the dataset.
        split_date : datetime
            The date at which the split occurred.
    '''

    df = df.sort_values(date_col)
    split_idx = int((1 - test_size) * len(df))
    train = df.iloc[:split_idx]
    test = df.iloc[split_idx:]
    return train, test, df.iloc[split_idx][date_col]


def create_features(df):

    '''
    This function adds temporal, lag, moving average, and encoded categorical features to the dataset.

    input: 
        df : pandas DataFrame
            The original dataset containing at least 'Date', 'Stock', 'Product', 'Country', 
            'CountryStatus', and 'ProductStatus' columns.

    output: 
        df : pandas DataFrame
            The same dataset with additional engineered features, including:
            - Year, Month, Week, DayOfWeek, DayOfMonth
            - Lag features: StockLag1, StockLag7
            - Moving averages: StockMA7, StockMA14
            - Encoded categorical columns: CountryStatus, ProductStatus, StatusCombination, 
              CountryEncoded, ProductEncoded
    '''

    # Temporal features
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    df['Week'] = df['Date'].dt.isocalendar().week.astype(np.int32)
    df['DayOfWeek'] = df['Date'].dt.dayofweek
    df['DayOfMonth'] = df['Date'].dt.day
    
    # Lag and moving average features for stock
    df = df.sort_values(['Product', 'Date'])
    df['StockLag1'] = df.groupby('Product')['Stock'].shift(1)
    df['StockLag7'] = df.groupby('Product')['Stock'].shift(7)
    df['StockMA7'] = df.groupby('Product')['Stock'].transform(lambda x: x.rolling(7, min_periods=1).mean())
    df['StockMA14'] = df.groupby('Product')['Stock'].transform(lambda x: x.rolling(14, min_periods=1).mean())
    
    # Fill NaN in lag features
    df[['StockLag1', 'StockLag7', 'StockMA7', 'StockMA14']] = df[['StockLag1', 'StockLag7', 'StockMA7', 'StockMA14']].fillna(0)
    
    # Encode categorical variables
    df['CountryStatus'] = df['CountryStatus'].map({'A': 1, 'Z': 0})
    df['ProductStatus'] = df['ProductStatus'].map({'A': 1, 'Z': 0})
    df['StatusCombination'] = df['CountryStatus'].astype(str) + df['ProductStatus'].astype(str)
    
    le_country = LabelEncoder()
    le_product = LabelEncoder()
    df['CountryEncoded'] = le_country.fit_transform(df['Country'])
    df['ProductEncoded'] = le_product.fit_transform(df['Product'])
    
    return df

def merge_external(df, df_ext):
    return df.merge(df_ext, on=['Date', 'Product'], how='left')