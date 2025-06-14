from statsmodels.tsa.statespace.sarimax import SARIMAX
import pandas as pd

def train_model_sarima(train_df, regressors, order=(1, 1, 1)):
    """
    Trains a SARIMA (SARIMAX) model with optional exogenous regressors.

    Parameters:
        train_df (DataFrame): Must contain 'Quantity' and regressors.
        regressors (list of str): Column names for exogenous variables.
        order (tuple): SARIMA order (p, d, q).
        seasonal_order (tuple): Seasonal order (P, D, Q, s). Default assumes yearly seasonality.

    Returns:
        model_fit: Fitted SARIMAXResultsWrapper object.
    """
    y_train = train_df['Quantity']
    X_train = train_df[regressors] if regressors else None

    model = SARIMAX(
        y_train,
        exog=X_train,
        order=order,
        seasonal_order=(1, 0, 1, 52),  # Weekly seasonality,
        enforce_stationarity=False,
        enforce_invertibility=False,
        low_memory=True
    )

    model_fit = model.fit(disp=False)
    return model_fit


def predict_sarima(model_fit, test_df, regressors):
    """
    Predicts Quantity using a fitted SARIMA model with exogenous variables.

    Parameters:
        model_fit: Fitted SARIMAXResultsWrapper object.
        test_df (DataFrame): Must contain same regressors as training set.
        regressors (list of str): Same regressors used during training.

    Returns:
        test_df (DataFrame): Original data with 'Predicted' column added.
    """
    X_test = test_df[regressors] if regressors else None
    n_periods = len(test_df)

    forecast = model_fit.predict(start=model_fit.nobs, end=model_fit.nobs + n_periods - 1, exog=X_test)

    test_df = test_df.copy()
    test_df['Predicted'] = forecast.values

    # Optional: Set prediction to 0 where Stock is 0
    if 'Stock' in test_df.columns:
        test_df.loc[test_df['Stock'] == 0, 'Predicted'] = 0

    return test_df