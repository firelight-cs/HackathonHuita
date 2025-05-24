import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX

def preprocess_main_data(main_df):
    main_df['Date'] = pd.to_datetime(main_df['Date'])
    main_df['YearMonth'] = main_df['Date'].dt.to_period('M').dt.to_timestamp()
    product_status = main_df.drop_duplicates('Product')[['Product', 'ProductStatus']].set_index('Product')
    country_status = main_df.drop_duplicates('Country')[['Country', 'CountryStatus']].set_index('Country')
    return main_df, product_status, country_status

def preprocess_campaign_data(campaign_df, product_status, country_status):
    campaign_df['ValidFrom'] = pd.to_datetime(campaign_df['ValidFrom'])
    campaign_df['ValidTo'] = pd.to_datetime(campaign_df['ValidTo'])
    campaign_df['Date'] = campaign_df.apply(
        lambda x: pd.date_range(x['ValidFrom'], x['ValidTo'], freq='D'), axis=1)
    campaign_df = campaign_df.explode('Date')
    campaign_df = campaign_df.merge(product_status, on='Product', how='left')
    campaign_df = campaign_df.merge(country_status, on='Country', how='left')
    campaign_df['StatusGroup'] = campaign_df['ProductStatus'] + campaign_df['CountryStatus']
    return campaign_df

def create_monthly_regressor(campaign_daily, group_cols):
    campaign_daily['YearMonth'] = campaign_daily['Date'].dt.to_period('M').dt.to_timestamp()
    regressor_monthly = campaign_daily.groupby(group_cols + ['YearMonth']).size().reset_index(name='CampaignCount')
    return regressor_monthly

def split_train_test(ts, regressor, test_size=0.2):
    split_idx = int(len(ts) * (1 - test_size))
    train = ts.iloc[:split_idx]
    test = ts.iloc[split_idx:]
    train_reg = regressor.iloc[:split_idx]
    test_reg = regressor.iloc[split_idx:]
    return train, test, train_reg, test_reg

def fit_sarima(train, exog_train, order=(1,0,0), seasonal_order=(1,0,0,12)):
    model = SARIMAX(train, exog=exog_train, order=order, seasonal_order=seasonal_order)
    return model.fit(disp=False)

def forecast(model_results, steps, exog_test):
    return model_results.get_forecast(steps=steps, exog=exog_test)

def plot_results(train, test, forecast, title):
    plt.figure(figsize=(12,6))
    plt.plot(train.index, train, label='Train')
    plt.plot(test.index, test, label='Test')
    forecast_index = forecast.predicted_mean.index
    plt.plot(forecast_index, forecast.predicted_mean, label='Forecast')
    plt.title(title)
    plt.legend()
    plt.show()

def method1(main_df, campaign_df):
    main_agg = main_df.groupby(['Country', 'YearMonth'])['Quantity'].sum().reset_index()
    campaign_daily = campaign_df.groupby(['Country', 'Date']).size().reset_index(name='CampaignCount')
    campaign_monthly = create_monthly_regressor(campaign_daily, ['Country'])
    for country in main_agg['Country'].unique():
        country_data = main_agg[main_agg['Country'] == country].set_index('YearMonth')['Quantity']
        country_reg = campaign_monthly[campaign_monthly['Country'] == country].set_index('YearMonth')['CampaignCount']
        full_idx = pd.date_range(country_data.index.min(), country_data.index.max(), freq='MS')
        country_data = country_data.reindex(full_idx).fillna(0)
        country_reg = country_reg.reindex(full_idx).fillna(0)
        train, test, train_reg, test_reg = split_train_test(country_data, country_reg)
        model = fit_sarima(train, train_reg)
        forecast_result = forecast(model, len(test), test_reg)
        plot_results(train, test, forecast_result, f'Method 1 - {country}')

def method2(main_df, campaign_df):
    main_agg = main_df.groupby('YearMonth')['Quantity'].sum()
    campaign_daily = campaign_df.groupby('Date').size().reset_index(name='CampaignCount')
    campaign_monthly = create_monthly_regressor(campaign_daily, [])
    full_idx = pd.date_range(main_agg.index.min(), main_agg.index.max(), freq='MS')
    main_ts = main_agg.reindex(full_idx).fillna(0)
    reg_ts = campaign_monthly.set_index('YearMonth')['CampaignCount'].reindex(full_idx).fillna(0)
    train, test, train_reg, test_reg = split_train_test(main_ts, reg_ts)
    model = fit_sarima(train, train_reg)
    forecast_result = forecast(model, len(test), test_reg)
    plot_results(train, test, forecast_result, 'Method 2 - All Products and Countries')

def method3(main_df, campaign_df):
    main_df['StatusGroup'] = main_df['ProductStatus'] + main_df['CountryStatus']
    main_agg = main_df.groupby(['StatusGroup', 'YearMonth'])['Quantity'].sum().reset_index()
    campaign_daily = campaign_df.groupby(['StatusGroup', 'Date']).size().reset_index(name='CampaignCount')
    campaign_monthly = create_monthly_regressor(campaign_daily, ['StatusGroup'])
    for group in main_agg['StatusGroup'].unique():
        group_data = main_agg[main_agg['StatusGroup'] == group].set_index('YearMonth')['Quantity']
        group_reg = campaign_monthly[campaign_monthly['StatusGroup'] == group].set_index('YearMonth')['CampaignCount']
        full_idx = pd.date_range(group_data.index.min(), group_data.index.max(), freq='MS')
        group_data = group_data.reindex(full_idx).fillna(0)
        group_reg = group_reg.reindex(full_idx).fillna(0)
        if len(group_data) < 2: continue
        train, test, train_reg, test_reg = split_train_test(group_data, group_reg)
        model = fit_sarima(train, train_reg)
        forecast_result = forecast(model, len(test), test_reg)
        plot_results(train, test, forecast_result, f'Method 3 - {group}')

# Example usage:
sales_df = pd.read_csv('./data/sell_data_cleaned.csv', sep=';', parse_dates=['Date'])
campaigns_df = pd.read_csv('./data/marketing_campaign.csv', sep=';', parse_dates=['ValidFrom', 'ValidTo'])
main_df, product_status, country_status = preprocess_main_data(sales_df)
campaign_df = preprocess_campaign_data(campaigns_df, product_status, country_status)
# method1(main_df, campaign_df)
# method2(main_df, campaign_df)
method3(main_df, campaign_df)