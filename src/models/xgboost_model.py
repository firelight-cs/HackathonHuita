from xgboost import XGBRegressor

def train_model_xgboost(train_df, features, target='Quantity'):
    X_train = train_df[features]
    y_train = train_df[target]
    model = XGBRegressor(objective='reg:squarederror')
    model.fit(X_train, y_train)
    return model

def predict_xgboost(model, test_df, features):
    X_test = test_df[features]
    test_df['Predicted'] = model.predict(X_test)
    
    # Post-process: set predictions to zero where stock is zero
    test_df.loc[test_df['Stock'] == 0, 'Predicted'] = 0
    return test_df