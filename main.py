import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt
import numpy as np

# Configuration
TEST_SIZE = 0.2
PLOT_RESAMPLE = 'W'  # Resample to weekly for plotting ('M' for monthly)

def prepare_data(sales_df, campaigns_df, country=None):
    # Filter data by country if specified
    filtered_sales = sales_df[sales_df['Country'] == country] if country else sales_df
    
    # Create complete date range for entire dataset
    min_date = filtered_sales['Date'].min()
    max_date = filtered_sales['Date'].max()
    full_dates = pd.DataFrame({'ds': pd.date_range(min_date, max_date)})
    
    # Aggregate sales data with complete date range
    agg_sales = filtered_sales.groupby('Date')['Quantity'].sum().reset_index()
    agg_sales.columns = ['ds', 'y']
    df = full_dates.merge(agg_sales, on='ds', how='left').fillna(0)
    
    # Process campaigns
    cmp = campaigns_df[campaigns_df['Country'] == country] if country else campaigns_df
    
    # Generate campaign dates
    campaign_dates = pd.DatetimeIndex([])
    for _, row in cmp.iterrows():
        campaign_dates = campaign_dates.union(pd.date_range(row['ValidFrom'], row['ValidTo'], freq='D'))
    
    df['campaign'] = df['ds'].isin(campaign_dates).astype(int)
    return df.sort_values('ds')

def time_based_split(df, test_size):
    split_date = df['ds'].quantile(1 - test_size, interpolation='nearest')
    train = df[df['ds'] < split_date]
    test = df[df['ds'] >= split_date]
    return train, test

def create_plot_data(df, resample_freq):
    return df.set_index('ds').resample(resample_freq).mean().reset_index()

def plot_comparison(train_df, test_df, forecast_df, title):
    plt.figure(figsize=(15, 7))
    
    # Resample data for plotting
    plot_train = create_plot_data(train_df, PLOT_RESAMPLE)
    plot_test = create_plot_data(test_df, PLOT_RESAMPLE)
    plot_forecast = create_plot_data(forecast_df, PLOT_RESAMPLE)
    
    plt.plot(plot_train['ds'], plot_train['y'], 'b-', label='Train')
    plt.plot(plot_test['ds'], plot_test['y'], 'g-', label='Test')
    plt.plot(plot_forecast['ds'], plot_forecast['yhat'], 'r--', label='Forecast')
    
    plt.fill_between(plot_forecast['ds'], 
                    plot_forecast['yhat_lower'], 
                    plot_forecast['yhat_upper'], 
                    color='pink', alpha=0.3, label='Uncertainty')
    
    plt.axvline(x=train_df['ds'].max(), color='k', linestyle='--', label='Train/Test Split')
    plt.title(title)
    plt.xlabel('Date')
    plt.ylabel('Quantity')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
def create_future_dataframe(train, test):
    """Create future dataframe exactly matching test dates"""
    future_dates = test['ds'].unique()
    return pd.DataFrame({'ds': future_dates})
# Main processing function
def process_data(method='per_country'):
    sales_df = pd.read_csv('./data/sell_data_cleaned.csv', sep=';', parse_dates=['Date'])
    campaigns_df = pd.read_csv('./data/marketing_campaign.csv', sep=';', parse_dates=['ValidFrom', 'ValidTo'])
    if method == 'per_country':
        for country in sales_df['Country'].unique():
            country_df = prepare_data(sales_df, campaigns_df, country)
            train, test = time_based_split(country_df, TEST_SIZE)
            
            model = Prophet()
            model.add_regressor('campaign')
            model.fit(train)
            
            # Create future dates from test data
            future = create_future_dataframe(train, test)
            
            # Merge campaign data from original full dataset
            future = future.merge(country_df[['ds', 'campaign']], on='ds', how='left')
            future['campaign'] = future['campaign'].fillna(0)
            
            forecast = model.predict(future)
            plot_comparison(train, test, forecast, f'Country: {country} Forecast')

    else:  # Global method
        global_df = prepare_data(sales_df, campaigns_df)
        train, test = time_based_split(global_df, TEST_SIZE)
        
        model = Prophet()
        model.add_regressor('campaign')
        model.fit(train)
        
        future = create_future_dataframe(train, test)
        future = future.merge(global_df[['ds', 'campaign']], on='ds', how='left')
        future['campaign'] = future['campaign'].fillna(0)
        
        forecast = model.predict(future)
        plot_comparison(train, test, forecast, 'Global Forecast')

# Execute both methods
process_data(method='per_country')
process_data(method='global')