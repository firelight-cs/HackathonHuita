import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error

def load_and_preprocess(sales_path, campaign_path, stock_path):
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

def create_features(df):
    # Temporal features
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    df['Week'] = df['Date'].dt.isocalendar().week
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

def split_train_test(df, test_size=0.2):
    df = df.sort_values('Date')
    cutoff_idx = int(len(df) * (1 - test_size))
    cutoff_date = df.iloc[cutoff_idx]['Date']
    train_df = df[df['Date'] <= cutoff_date]
    test_df = df[df['Date'] > cutoff_date]
    return train_df, test_df, cutoff_date

def train_model(train_df, features, target='Quantity'):
    X_train = train_df[features]
    y_train = train_df[target]
    model = xgb.XGBRegressor(objective='reg:squarederror')
    model.fit(X_train, y_train)
    return model

def predict(model, test_df, features):
    X_test = test_df[features]
    test_df['Predicted'] = model.predict(X_test)
    # Post-process: set predictions to zero where stock is zero
    test_df.loc[test_df['Stock'] == 0, 'Predicted'] = 0
    return test_df

def plot_method1(train_df, test_df, cutoff_date):
    countries = train_df['Country'].unique()
    for country in countries:
        train_country = train_df[train_df['Country'] == country]
        test_country = test_df[test_df['Country'] == country]
        if len(test_country) == 0:
            continue
        combined = pd.concat([train_country, test_country])
        aggregated = combined.groupby('Date').agg({'Quantity': 'sum', 'Predicted': 'sum'}).reset_index()
        train_agg = aggregated[aggregated['Date'] <= cutoff_date]
        test_agg = aggregated[aggregated['Date'] > cutoff_date]
        
        plt.figure(figsize=(12, 6))
        plt.plot(train_agg['Date'], train_agg['Quantity'], label='Train Actual')
        plt.plot(test_agg['Date'], test_agg['Quantity'], label='Test Actual', linestyle='-', color='yellow', alpha=0.5)
        plt.plot(test_agg['Date'], test_agg['Predicted'], label='Predicted', linestyle='-', color='red')
        plt.title(f'Method 1: Country {country}')
        plt.xlabel('Date')
        plt.ylabel('Quantity')
        plt.legend()
        plt.show()

def plot_method2(train_df, test_df, cutoff_date):
    combined = pd.concat([train_df, test_df])
    aggregated = combined.groupby('Date').agg({'Quantity': 'sum', 'Predicted': 'sum'}).reset_index()
    train_agg = aggregated[aggregated['Date'] <= cutoff_date]
    test_agg = aggregated[aggregated['Date'] > cutoff_date]
    
    plt.figure(figsize=(12, 6))
    plt.plot(train_agg['Date'], train_agg['Quantity'], label='Train Actual')
    plt.plot(test_agg['Date'], test_agg['Quantity'], label='Test Actual', linestyle='-', color='yellow', alpha=0.5)
    plt.plot(test_agg['Date'], test_agg['Predicted'], label='Predicted', linestyle='-', color='red')
    plt.title('Method 2: All Products and Countries')
    plt.xlabel('Date')
    plt.ylabel('Quantity')
    plt.legend()
    plt.show()

def plot_method3(train_df, test_df, cutoff_date):
    status_combinations = train_df['StatusCombination'].unique()
    for status in status_combinations:
        train_status = train_df[train_df['StatusCombination'] == status]
        test_status = test_df[test_df['StatusCombination'] == status]
        if len(test_status) == 0:
            continue
        combined = pd.concat([train_status, test_status])
        aggregated = combined.groupby('Date').agg({'Quantity': 'sum', 'Predicted': 'sum'}).reset_index()
        train_agg = aggregated[aggregated['Date'] <= cutoff_date]
        test_agg = aggregated[aggregated['Date'] > cutoff_date]
        
        plt.figure(figsize=(12, 6))
        plt.plot(train_agg['Date'], train_agg['Quantity'], label='Train Actual')
        plt.plot(test_agg['Date'], test_agg['Quantity'], label='Test Actual', linestyle='-', color='yellow', alpha=0.5)
        plt.plot(test_agg['Date'], test_agg['Predicted'], label='Predicted', linestyle='-', color='red')
        plt.title(f'Method 3: Status {status}')
        plt.xlabel('Date')
        plt.ylabel('Quantity')
        plt.legend()
        plt.show()

# Example usage:
# Load and preprocess data
merged_df = load_and_preprocess('./data/sell_data_cleaned.csv', './data/marketing_campaign.csv', './data/stock.csv')
featured_df = create_features(merged_df)
train_df, test_df, cutoff_date = split_train_test(featured_df)

# Define features
features = [
    'Year', 'Month', 'Week', 'DayOfWeek', 'DayOfMonth',
    'CountryEncoded', 'ProductEncoded', 'CountryStatus', 'ProductStatus',
    'ActiveCampaign', 'Stock', 'StockLag1', 'StockLag7', 'StockMA7', 'StockMA14'
]

# Train model
model = train_model(train_df, features)

# Predict
test_df = predict(model, test_df, features)

# Evaluate
rmse = np.sqrt(mean_squared_error(test_df['Quantity'], test_df['Predicted']))
mae = mean_absolute_error(test_df['Quantity'], test_df['Predicted'])
print(f'RMSE: {rmse:.2f}, MAE: {mae:.2f}')

# Generate plots
plot_method1(train_df, test_df, cutoff_date)
plot_method2(train_df, test_df, cutoff_date)
plot_method3(train_df, test_df, cutoff_date)