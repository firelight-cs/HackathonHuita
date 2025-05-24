import pandas as pd
from prophet import Prophet
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

def load_data():
    # Load datasets (example placeholders)
    df_train = pd.read_csv('./data/sell_data_cleaned.csv', sep=';', parse_dates=['Date'])
    df_campaign = pd.read_csv('./data/marketing_campaign.csv', sep=';', parse_dates=['ValidFrom', 'ValidTo'])
    df_stock = pd.read_csv('./data/stock.csv', sep=';', parse_dates=['Datum'])
    df_stock.rename(columns={'Datum': 'Date', 'AnonymniProdukt': 'Product', 'Mnozstvi': 'Stock'}, inplace=True)
    return df_train, df_campaign, df_stock

def preprocess_data(df_train, df_stock):
    df_train = pd.merge(df_train, df_stock, on=['Product', 'Date'], how='left')
    df_train['Quantity'] = df_train.apply(lambda x: 0 if x['Stock'] == 0 else x['Quantity'], axis=1)
    df_train['StatusCombo'] = df_train['CountryStatus'] + df_train['ProductStatus']
    return df_train

def create_aggregated_data(df, group_cols, df_campaign):
    # Aggregate Quantity
    agg_df = df.groupby(group_cols + ['Date'])['Quantity'].sum().reset_index()
    
    # Process campaign data
    campaign_dates = []
    for _, row in df_campaign.iterrows():
        dates = pd.date_range(row['ValidFrom'], row['ValidTo'], freq='D')
        for date in dates:
            campaign_dates.append({
                'Date': date,
                'Product': row['Product'],
                'Country': row['Country'],
                'CampaignActive': 1
            })
    campaign_df = pd.DataFrame(campaign_dates)
    if not group_cols:
        campaign_agg = campaign_df.groupby('Date')['CampaignActive'].max().reset_index()
        agg_df = pd.merge(agg_df, campaign_agg, on='Date', how='left').fillna(0)
    else:
        # Merge campaign activity
        if 'Country' in group_cols:
            campaign_agg = campaign_df.groupby(['Country', 'Date'])['CampaignActive'].max().reset_index()
            merge_cols = ['Country', 'Date']
        else:
            campaign_agg = campaign_df.groupby(['Date'])['CampaignActive'].max().reset_index()
            merge_cols = ['Date']
        agg_df = pd.merge(agg_df, campaign_agg, on=merge_cols, how='left').fillna(0)
    if 'CampaignActive' not in agg_df.columns:
        agg_df['CampaignActive'] = 0
    
    # Aggregate Stock features (CORRECTED SECTION)
    df['InStock'] = df['Stock'] > 0
    stock_agg = df.groupby(group_cols + ['Date'])[['Stock', 'InStock']].agg({
        'Stock': ['sum', 'mean'],
        'InStock': 'sum'
    }).reset_index()
    stock_agg.columns = [f"{col[0]}_{col[1]}" if col[1] else col[0] for col in stock_agg.columns]
    agg_df = pd.merge(agg_df, stock_agg, on=group_cols + ['Date'], how='left')
    
    # Create lag features and moving averages
    agg_df.sort_values(group_cols + ['Date'], inplace=True)
    
    for col in ['Stock_sum', 'Stock_mean', 'InStock_sum']:
        # Handle global case (no grouping)
        if not group_cols:  # Method 2 scenario
            agg_df[f'{col}_lag1'] = agg_df[col].shift(1)
            agg_df[f'{col}_7d_avg'] = agg_df[col].rolling(7, min_periods=1).mean()
        else:  # Grouped scenarios (Methods 1 & 3)
            agg_df[f'{col}_lag1'] = agg_df.groupby(group_cols)[col].shift(1)
            agg_df[f'{col}_7d_avg'] = agg_df.groupby(group_cols)[col].transform(
                lambda x: x.rolling(7, min_periods=1).mean()
            )
        
        # Fill initial NaN values
        agg_df[f'{col}_lag1'] = agg_df[f'{col}_lag1'].fillna(0)
    return agg_df

def train_test_split_chronological(df, group_cols, test_size=0.2):
    # Handle empty group_cols by adding a dummy group
    dummy_added = False
    if not group_cols:
        df = df.assign(_dummy_group=0)
        group_cols = ['_dummy_group']
        dummy_added = True

    missing = [col for col in group_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing group columns: {missing}")

    # Check for empty groups
    if df[group_cols].nunique().prod() == 0:
        raise ValueError("No valid groups for splitting")

    train_dfs, test_dfs = [], []
    for name, group in df.groupby(group_cols):
        group = group.sort_values('Date')
        split_idx = int(len(group) * (1 - test_size))
        
        # Ensure at least one test sample
        split_idx = max(1, min(split_idx, len(group)-2))
        
        train_part = group.iloc[:split_idx].copy()
        test_part = group.iloc[split_idx:].copy()
        
        # Verify temporal sequence
        assert test_part['Date'].min() > train_part['Date'].max(), \
            "Test data must start after training data"
        
        train_dfs.append(train_part)
        test_dfs.append(test_part)
    
    # Concatenate results and remove dummy if added
    train = pd.concat(train_dfs)
    test = pd.concat(test_dfs)
    
    if dummy_added:
        train = train.drop(columns=['_dummy_group'])
        test = test.drop(columns=['_dummy_group'])
        group_cols = []
    
    return train, test

def train_prophet(train_data, group_cols, regressors):
    models = {}
    
    # Handle empty group_cols by adding a dummy group
    dummy_added = False
    if not group_cols:
        train_data = train_data.assign(_dummy_group=0)
        group_cols = ['_dummy_group']
        dummy_added = True

    for name, group in train_data.groupby(group_cols):
        df = group.rename(columns={'Date': 'ds', 'Quantity': 'y'})
        model = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=True,
            daily_seasonality=False
        )
        model.add_seasonality(name='monthly', period=30.5, fourier_order=5)
        
        # Add regressors
        for reg in regressors:
            model.add_regressor(reg)
            
        # Fit with regressor data
        model.fit(df[['ds', 'y'] + regressors])
        
        # Use "Global" as the key if dummy was added
        if dummy_added:
            models['Global'] = model
        else:
            models[name] = model
    
    return models
def predict_and_plot(models, test_data, group_cols, regressors):
    # Handle empty group_cols by adding a dummy group
    dummy_added = False
    if not group_cols:
        test_data = test_data.assign(_dummy_group=0)
        group_cols = ['_dummy_group']
        dummy_added = True

    for name, group in test_data.groupby(group_cols):
        # Use "Global" as the key if dummy was added
        model_key = 'Global' if dummy_added else name
        model = models.get(model_key)
        if not model:
            continue

        # Get training history (already handled in models)
        train_history = model.history[['ds', 'y']].rename(
            columns={'ds': 'Date', 'y': 'Quantity'}
        )

        # Create future dataframe with test dates
        future = pd.DataFrame({'ds': pd.to_datetime(group['Date'].unique())})
        future = future.sort_values('ds').reset_index(drop=True)

        # Merge regressors from test data
        regressor_df = group.drop_duplicates('Date')[['Date'] + regressors]
        future = future.merge(regressor_df, left_on='ds', right_on='Date', how='left')

        # Fill missing regressors
        future[regressors] = future[regressors].fillna(0)

        # Generate forecast
        forecast = model.predict(future)

        # Plotting logic (unchanged)
        plt.figure(figsize=(15, 6))
        plt.plot(train_history['Date'], train_history['Quantity'], 
                 label='Train Actual', color='blue', marker='o')
        plt.plot(group['Date'], group['Quantity'], 
                 label='Test Actual', color='green', marker='x')
        plt.plot(forecast['ds'], forecast['yhat'], 
                 label='Forecast', color='red', linestyle='--')
        plt.fill_between(forecast['ds'], forecast['yhat_lower'], forecast['yhat_upper'],
                         alpha=0.2, color='red')
        plt.axvline(train_history['Date'].max(), color='gray', linestyle='--',
                     label='Train/Test Split')
        plt.title(f'{model_key} - Forecast vs Actual')
        plt.xlabel('Date')
        plt.ylabel('Quantity')
        plt.legend()
        plt.grid(True)
        plt.show()

    # Cleanup dummy column if added
    if dummy_added:
        test_data.drop(columns=['_dummy_group'], inplace=True)

def method1(df_train, df_campaign):
    print("Method 1 proccessing: ")
    
    # Country-level aggregation
    agg_df = create_aggregated_data(df_train, ['Country'], df_campaign)
    train, test = train_test_split_chronological(agg_df, ['Country'])
    regressors = ['CampaignActive', 'Stock_sum_lag1', 'InStock_sum_7d_avg']
    models = train_prophet(train, ['Country'], regressors)
    predict_and_plot(models, test, ['Country'], regressors)

def method2(df_train, df_campaign):
    print("Method 2 proccessing")
    # Global aggregation
    agg_df = create_aggregated_data(df_train, [], df_campaign)
    train, test = train_test_split_chronological(agg_df, [])
    regressors = ['CampaignActive', 'Stock_sum_lag1', 'InStock_sum_7d_avg']
    models = train_prophet(train, [], regressors)  # Returns {'Global': model}
    predict_and_plot(models, test, [], regressors)
    
def method3(df_train, df_campaign):
    print("Method 3 proccessing: ")
    # Status combination aggregation
        # Verify StatusCombo exists
    if 'StatusCombo' not in df_train.columns:
        raise KeyError("'StatusCombo' column missing - check preprocessing")
    
    # Status combination aggregation
    agg_df = create_aggregated_data(
        df_train, 
        group_cols=['StatusCombo'],  # Explicit list
        df_campaign=df_campaign
    )
    
    # Verify aggregation output
    if 'StatusCombo' not in agg_df.columns:
        raise RuntimeError("StatusCombo not preserved in aggregation")
    
    # Split data (now with guaranteed grouping column)
    train, test = train_test_split_chronological(
        agg_df, 
        group_cols=['StatusCombo']  # Explicit grouping
    )
    regressors = ['CampaignActive', 'Stock_sum_lag1', 'InStock_sum_7d_avg']
    models = train_prophet(train, ['StatusCombo'], regressors)
    predict_and_plot(models, test, ['StatusCombo'], regressors)

def main():
    df_train, df_campaign, df_stock = load_data()
    df_train = preprocess_data(df_train, df_stock)
    
    # Example: Test with AA status
    #df_aa = df_train[df_train['StatusCombo'] == 'AA']
    #method3(df_aa, df_campaign)
    
    # Run all methods
    method1(df_train, df_campaign)
    method2(df_train, df_campaign)
    method3(df_train, df_campaign)

if __name__ == "__main__":
    main()