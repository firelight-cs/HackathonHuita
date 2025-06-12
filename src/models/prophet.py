from prophet import Prophet

def train_model_prophet(train_df, regressors):
    model = Prophet(yearly_seasonality=True, weekly_seasonality=True)
    for reg in regressors:
        model.add_regressor(reg)
    
    prophet_df = train_df.rename(columns={'Date': 'ds', 'Quantity': 'y'})
    model.fit(prophet_df[['ds', 'y'] + regressors])
    return model

def predict_prophet(model, test_df, regressors):
    future = test_df[['Date'] + regressors].rename(columns={'Date': 'ds'})
    forecast = model.predict(future)
    test_df['Predicted'] = forecast['yhat'].values
    test_df.loc[test_df['Stock'] == 0, 'Predicted'] = 0  # Optional: stock-aware correction
    return test_df