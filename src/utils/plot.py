import matplotlib.pyplot as plt
import pandas as pd

def plot_forecast(original, preds, split_date, title):
    
    '''
    Plots the actual and predicted time series data with a split line indicating the cutoff date.

    Parameters:
    ----------
    original : pd.Series
        A pandas Series containing the actual time series values (train + test period). 
        The index should be datetime values.

    preds : pd.Series
        A pandas Series containing the predicted values. Typically only for the test period.
        The index should be datetime values.

    split_date : datetime
        A datetime object or string representing the cutoff date between training and testing periods.
        This will be shown as a vertical line on the plot.

    title : str
        The title of the plot.
    '''

    plt.figure(figsize=(12,6), facecolor='#242424')
    plt.plot(original.index, original.values, label='Train+Test', color = '#fefefe')
    plt.plot(preds.index, preds.values, label='Predicted', linestyle='--', color = "red")
    plt.axvline(split_date, color='blue', linestyle=':')

    ax = plt.gca()
    ax.set_facecolor('#242424')
    ax.tick_params(axis='x', colors='#fefefe')
    ax.tick_params(axis='y', colors='#fefefe')
    ax.spines['bottom'].set_color('#fefefe')
    ax.spines['top'].set_color('#fefefe')
    ax.spines['right'].set_color('#fefefe')
    ax.spines['left'].set_color('#fefefe')
    plt.title(title, color='#fefefe')
    plt.xlabel('Date', color='#fefefe')
    plt.ylabel('Quantity', color='#fefefe')
    plt.legend(labelcolor='#fefefe', facecolor='#242424')
    plt.grid(False)
    plt.show()

def plot_method1(train_df, test_df, cutoff_date):

    '''
    Purpose: Visualizes model performance over time for each country by comparing actual vs. predicted sales quantities.

    - Loops through each country in the training data.

    For each country skips countries that have no test data.

    1. Aggregates total actual sales (Quantity) and predicted sales (Predicted) for each date.

    2. Splits the aggregated data into:

        - Training period (up to the cutoff date)
        - Testing period (after the cutoff date)

    3. Plots the actual vs. predicted sales:

        - Actual training data: white line
        - Actual test data: yellow line
        - Predicted test data: red line

    4. Styles the plot with a dark background and labeled axes.
    '''

    countries = train_df['Country'].unique()
    for country in countries:
        train_country = train_df[train_df['Country'] == country]
        test_country = test_df[test_df['Country'] == country]
        if len(test_country) == 0:
            continue
        combined = pd.concat([train_country, test_country])

        # Group by date, sum Quantity and Predicted
        aggregated = combined.groupby('Date').agg({'Quantity': 'sum', 'Predicted': 'sum'}).reset_index()

        # Prepare the original series (actual Quantity for all dates)
        original_series = aggregated.set_index('Date')['Quantity']
        
        # Prepare the predicted series only for dates after cutoff
        preds_series = aggregated[aggregated['Date'] > cutoff_date].set_index('Date')['Predicted']

        # Use plot_forecast to plot
        plot_forecast(
            original=original_series,
            preds=preds_series,
            split_date=cutoff_date,
            title=f'Method 1: Country {country}'
        )

def plot_method2(train_df, test_df, cutoff_date):

    '''
    Shows the model's performance over time across all products and countries.
    '''

    combined = pd.concat([train_df, test_df])
    aggregated = combined.groupby('Date').agg({'Quantity': 'sum', 'Predicted': 'sum'}).reset_index()

    # Prepare the original series (actual Quantity for all dates)
    original_series = aggregated.set_index('Date')['Quantity']
    
    # Prepare the predicted series only for dates after cutoff
    preds_series = aggregated[aggregated['Date'] > cutoff_date].set_index('Date')['Predicted']

    plot_forecast(
        original=original_series,
        preds=preds_series,
        split_date=cutoff_date,
        title="Forecast for all countries"
    )


def plot_method3(train_df, test_df, cutoff_date):

    '''
    Useful to analyze model performance across status-driven segments (e.g., active vs inactive products in certain regions). It helps answer:

    -Does the model perform better when both country and product are active?

    -Do inactive combinations behave differently over time?
    '''

    status_combinations = train_df['StatusCombination'].unique()
    for status in status_combinations:
        train_status = train_df[train_df['StatusCombination'] == status]
        test_status = test_df[test_df['StatusCombination'] == status]
        if len(test_status) == 0:
            continue
        combined = pd.concat([train_status, test_status])

        aggregated = combined.groupby('Date').agg({'Quantity': 'sum', 'Predicted': 'sum'}).reset_index()

        # Prepare the original series (actual Quantity for all dates)
        original_series = aggregated.set_index('Date')['Quantity']
        
        # Prepare the predicted series only for dates after cutoff
        preds_series = aggregated[aggregated['Date'] > cutoff_date].set_index('Date')['Predicted']

        plot_forecast(
            original=original_series,
            preds=preds_series,
            split_date=cutoff_date,
            title=f'Forecast for {status}'
        )


    
    