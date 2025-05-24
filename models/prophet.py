from prophet import Prophet

def create_prophet_model():
    """Configure Prophet model with seasonality"""
    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False,
        seasonality_mode='multiplicative',
        changepoint_prior_scale=0.05
    )
    model.add_seasonality(name='monthly', period=30.5, fourier_order=5)
    return model
