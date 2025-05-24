import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt
import numpy as np
from models.prophet import create_prophet_model

# Configuration
TEST_SIZE = 0.2
PLOT_RESAMPLE = 'W'  # Weekly resampling for visualization
MIN_DATA_DAYS = 56  # Minimum 8 weeks of data for modeling

def get_data():
    sales_df = pd.read_csv('./data/sell_data_cleaned.csv', sep=';', parse_dates=['Date'])
    campaigns_df = pd.read_csv('./data/marketing_campaign.csv', sep=';', parse_dates=['ValidFrom', 'ValidTo'])
    return sales_df, campaigns_df

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

def plot_comparison(train, test, forecast, title):
    """Enhanced visualization with status annotations"""
    plt.figure(figsize=(15, 7))
    
    # Resample data for plotting
    resampled_train = train.set_index('ds').resample(PLOT_RESAMPLE).mean()
    resampled_test = test.set_index('ds').resample(PLOT_RESAMPLE).mean()
    resampled_forecast = forecast.set_index('ds').resample(PLOT_RESAMPLE).mean()
    
    # Plot components
    plt.plot(resampled_train.index, resampled_train['y'], 'b-', label='Train')
    plt.plot(resampled_test.index, resampled_test['y'], 'g-', label='Test')
    plt.plot(resampled_forecast.index, resampled_forecast['yhat'], 'r--', label='Forecast')
    
    # Uncertainty region
    plt.fill_between(resampled_forecast.index,
                    resampled_forecast['yhat_lower'],
                    resampled_forecast['yhat_upper'],
                    color='pink', alpha=0.3)
    
    # Annotations
    plt.axvline(x=train['ds'].max(), color='k', linestyle='--')
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
    sales_df, campaigns_df = get_data()
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
        

def prepare_combination_data(sales_df, campaigns_df, p_status, c_status):
    """Prepare data for specific status combination"""
    # Filter relevant sales data
    comb_df = sales_df[
        (sales_df['ProductStatus'] == p_status) & 
        (sales_df['CountryStatus'] == c_status)
    ].copy()
    
    # Get related products and countries
    products = comb_df['Product'].unique()
    countries = comb_df['Country'].unique()
    
    # Aggregate sales
    agg_df = comb_df.resample('D', on='Date')['Quantity'].sum().reset_index()
    agg_df.columns = ['ds', 'y']
    
    # Get relevant campaigns
    cmp_filtered = campaigns_df[
        campaigns_df['Product'].isin(products) & 
        campaigns_df['Country'].isin(countries)
    ]
    
    # Create campaign indicators
    campaign_dates = pd.DatetimeIndex([])
    for _, row in cmp_filtered.iterrows():
        campaign_dates = campaign_dates.union(
            pd.date_range(row['ValidFrom'], row['ValidTo'], freq='D'))
    
    agg_df['campaign'] = agg_df['ds'].isin(campaign_dates).astype(int)
    return agg_df.sort_values('ds')
    
def analyze_combination(p_status, c_status, sales_df, campaigns_df):
    """Full analysis pipeline for one status combination"""
    print(f"\nAnalyzing {p_status}{c_status} combination...")
    
    # Prepare data
    comb_data = prepare_combination_data(sales_df, campaigns_df, p_status, c_status)
    
    # Skip combinations with insufficient data
    if len(comb_data) < MIN_DATA_DAYS:
        print(f"Insufficient data for {p_status}{c_status} ({len(comb_data)} days)")
        return
    
    # Train-test split
    split_date = comb_data['ds'].quantile(1 - TEST_SIZE)
    train = comb_data[comb_data['ds'] <= split_date]
    test = comb_data[comb_data['ds'] > split_date]
    
    # Initialize and fit model
    model = create_prophet_model()
    model.add_regressor('campaign')
    model.fit(train)
    
    # Create future dataframe
    future = pd.DataFrame({'ds': test['ds']})
    future = future.merge(comb_data[['ds', 'campaign']], on='ds', how='left')
    future['campaign'] = future['campaign'].fillna(0)
    
    # Generate forecast
    forecast = model.predict(future)
    
    # Plot results
    plot_comparison(train, test, forecast, 
                   f"{p_status}{c_status} Forecast: {len(train)} train days, {len(test)} test days")
    
def run_analysis(sales_df, campaigns_df):
    # First analyze AA combination as reference
    analyze_combination('A', 'A', sales_df, campaigns_df)
    
    # Then analyze all other combinations
    statuses = sales_df[['ProductStatus', 'CountryStatus']].drop_duplicates().values
    for p_status, c_status in statuses:
        if p_status == 'A' and c_status == 'A':
            continue  # Already processed
        analyze_combination(p_status, c_status, sales_df, campaigns_df)

# Execute analysis
sales_df, campaigns_df = get_data()
#run_analysis(sales_df, campaigns_df)

# Execute both methods
#process_data(method='per_country')
process_data(method='global')