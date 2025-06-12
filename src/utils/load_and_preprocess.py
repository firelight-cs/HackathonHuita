import pandas as pd

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