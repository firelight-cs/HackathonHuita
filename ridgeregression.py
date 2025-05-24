import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import TimeSeriesSplit, train_test_split
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

# --- 1. Load feature-engineered dataset ---
# Assumes a DataFrame `features` exists or load from CSV if saved
# features = pd.read_csv("features_with_all.csv", parse_dates=['Date'])
# For demo, we load the daily sales series:
sales = pd.read_csv("clean_sales.csv", parse_dates=['Date'])
# Here you'd merge features computed in previous steps:
# data = features.merge(other_feature_tables, on=['Product','Date'], how='left')

# For simplicity, we'll forecast total daily sales (aggregated)
daily = sales.groupby('Date')['Quantity'].sum().reset_index()
daily['DayOfWeek'] = daily['Date'].dt.dayofweek
daily['Month'] = daily['Date'].dt.month
# Lag features
daily['Lag_1'] = daily['Quantity'].shift(1)
daily['Lag_7'] = daily['Quantity'].shift(7)
# Rolling mean
daily['RollMean_7'] = daily['Quantity'].rolling(7).mean()
# Drop NaNs
daily = daily.dropna().reset_index(drop=True)

# --- 2. Split into training and test (last 6 months as test) ---
cutoff = daily['Date'].max() - pd.DateOffset(months=6)
train = daily[daily['Date'] <= cutoff]
test  = daily[daily['Date'] > cutoff]

X_cols = ['DayOfWeek','Month','Lag_1','Lag_7','RollMean_7']
y_col = 'Quantity'

X_train, y_train = train[X_cols], train[y_col]
X_test, y_test   = test[X_cols], test[y_col]

# --- 3. Train models ---
models = {
    'Ridge': Ridge(),
    'RandomForest': RandomForestRegressor(n_estimators=100, random_state=42)
}

results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    results[name] = {'MAE': mae, 'RMSE': rmse, 'preds': preds}

# --- 4. Print metrics ---
for name, res in results.items():
    print(f"{name}: MAE = {res['MAE']:.2f}, RMSE = {res['RMSE']:.2f}")

# --- 5. Plot actual vs predicted for best model ---
best = min(results.items(), key=lambda x: x[1]['RMSE'])[0]
plt.figure()
plt.plot(test['Date'], y_test, label='Actual')
plt.plot(test['Date'], results[best]['preds'], label=f'Predicted ({best})')
plt.title(f'Actual vs Predicted Daily Sales ({best})')
plt.xlabel('Date')
plt.ylabel('Quantity Sold')
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# --- 6. Residual analysis ---
residuals = y_test - results[best]['preds']
plt.figure()
plt.hist(residuals, bins=30)
plt.title(f'Residual Distribution ({best})')
plt.xlabel('Error')
plt.ylabel('Frequency')
plt.tight_layout()
plt.show()