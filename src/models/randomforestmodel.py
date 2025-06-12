import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt

def train_model(train_df, regressors, target='Quantity'):
    X_train = train_df[regressors]
    y_train = train_df[target]
    
    model = RandomForestRegressor(
        n_estimators=100,
        max_features='sqrt',
        random_state=42,
        n_jobs=-1,
        oob_score=False
    )
    model.fit(X_train, y_train)
    return model

def predict(model, test_df, features, target='Quantity'):
    test_df = test_df.copy()
    X_test = test_df[features]
    
    # Predict with the model
    test_df['Predicted'] = model.predict(X_test)
    
    # Optional: set predicted quantity to 0 where there's no stock
    if 'Stock' in test_df.columns:
        test_df.loc[test_df['Stock'] == 0, 'Predicted'] = 0

    return test_df

