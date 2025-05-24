import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pmdarima import auto_arima
from joblib import Parallel, delayed
import warnings

warnings.filterwarnings("ignore")

# ================== OPTIMIZED PREPROCESSING ==================
def preprocess_data(sales_df, campaigns_df):
    # Convert campaign dates to binary flags
    campaigns_df['ValidFrom'] = pd.to_datetime(campaigns_df['ValidFrom'])
    campaigns_df['ValidTo'] = pd.to_datetime(campaigns_df['ValidTo'])
    
    # Create date range covering all possible dates
    min_date = sales_df['Date'].min().floor('D')
    max_date = sales_df['Date'].max().ceil('D')
    full_date_range = pd.date_range(min_date, max_date, freq='D', name='Date')
    
    # Create all product-country-date combinations
    pc_combinations = sales_df[['Product', 'Country']].drop_duplicates()
    full_index = pd.MultiIndex.from_product(
        [pc_combinations['Product'], 
         pc_combinations['Country'],
         full_date_range],
        names=['Product', 'Country', 'Date']
    )
    
    # Merge with campaigns
    campaigns_flags = campaigns_df.assign(
        Date=lambda x: x.apply(
            lambda r: pd.date_range(r['ValidFrom'], r['ValidTo'], freq='D'),
            axis=1
        )
    ).explode('Date').drop_duplicates()
    campaigns_flags['CampaignActive'] = 1
    
    merged_df = (
        pd.DataFrame(index=full_index)
        .reset_index()
        .merge(campaigns_flags,
               on=['Product', 'Country', 'Date'],
               how='left')
        .fillna({'CampaignActive': 0})
    )
    
    # Merge with sales data
    sales_agg = sales_df.groupby(['Product', 'Country', 'Date'])['Quantity'].sum().reset_index()
    final_df = merged_df.merge(sales_agg, on=['Product', 'Country', 'Date'], how='left')
    
    # Add status information
    status_map = sales_df[['Product', 'Country', 'ProductStatus', 'CountryStatus']].drop_duplicates()
    final_df = final_df.merge(status_map, on=['Product', 'Country'], how='left')
    
    # Weekly aggregation
    final_df['Week'] = final_df['Date'].dt.to_period('W').dt.start_time
    weekly_df = final_df.groupby(['Product', 'Country', 'Week']).agg(
        Quantity=('Quantity', 'sum'),
        CampaignActive=('CampaignActive', 'max'),
        ProductStatus=('ProductStatus', 'first'),
        CountryStatus=('CountryStatus', 'first')
    ).reset_index()
    
    return weekly_df

# ================== OPTIMIZED MODELING ==================
def safe_auto_arima(train_data, exog_data):
    """Constrain auto_arima to prevent invalid configurations"""
    try:
        return auto_arima(
            train_data,
            exogenous=exog_data,
            seasonal=True,
            m=4,  # Monthly seasonality instead of yearly
            suppress_warnings=True,
            stepwise=True,
            error_action='ignore',
            max_order=6,  # Prevent over-complex models
            trace=False
        )
    except:
        return None

def forecast_group(group_data, group_name, model_type):
    if len(group_data) < 52:  # Minimum 1 year of weekly data
        return None
    
    # Split data preserving temporal order
    train_size = int(len(group_data) * 0.8)
    train = group_data.iloc[:train_size]
    test = group_data.iloc[train_size:]
    
    if len(test) == 0:
        return None
    
    # Prepare data
    y_train = train['Quantity']
    exog_train = train[['CampaignActive']]
    exog_test = test[['CampaignActive']]
    
    # Model training with fallback
    model = safe_auto_arima(y_train, exog_train)
    if model is None:
        return None
    
    # Forecasting
    try:
        forecast = model.predict(n_periods=len(test), exogenous=exog_test)
    except:
        return None
    
    # Create plot
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(train['Week'], y_train, label='Train')
    ax.plot(test['Week'], test['Quantity'], label='Test')
    ax.plot(test['Week'], forecast, label='Forecast')
    ax.set_title(f"{model_type} - {group_name}")
    ax.legend()
    plt.close()  # Close plot to prevent display in notebooks
    
    return fig

# ================== PARALLEL PROCESSING METHODS ==================
def method1(sales_weekly):
    groups = sales_weekly.groupby('Country')
    results = Parallel(n_jobs=-1)(
        delayed(forecast_group)(group, name, "Method1") 
        for name, group in groups
    )
    _ = [plt.show(fig) for fig in results if fig is not None]

def method2(sales_weekly):
    global_group = sales_weekly.groupby('Week').agg(
        Quantity=('Quantity', 'sum'),
        CampaignActive=('CampaignActive', 'max')
    ).reset_index()
    fig = forecast_group(global_group, "Global", "Method2")
    if fig:
        plt.show(fig)

def method3(sales_weekly):
    groups = sales_weekly.groupby(['ProductStatus', 'CountryStatus'])
    results = Parallel(n_jobs=-1)(
        delayed(forecast_group)(group, f"{ps}{cs}", "Method3") 
        for (ps, cs), group in groups
    )
    _ = [plt.show(fig) for fig in results if fig is not None]


sales_df = pd.read_csv('./data/sell_data_cleaned.csv', sep=';', parse_dates=['Date'])
campaigns_df = pd.read_csv('./data/marketing_campaign.csv', sep=';', 
                         parse_dates=['ValidFrom', 'ValidTo'])

# Preprocess
print("Preprocessing data...")
sales_weekly = preprocess_data(sales_df, campaigns_df)

# Execute methods
print("Running Method1...")
method1(sales_weekly)

# print("Running Method2...")
# method2(sales_weekly)

# print("Running Method3...")
# method3(sales_weekly)