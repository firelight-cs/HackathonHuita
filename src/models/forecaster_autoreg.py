from skforecast.recursive import ForecasterRecursive
from skforecast.model_selection import grid_search_forecaster
from sklearn.tree import DecisionTreeRegressor
import pandas as pd
from typing import Union, List, Tuple, Dict

def train_forecaster_autoreg_with_grid_search(
    y: pd.Series,
    initial_train_size: int,
    param_grid: Dict,
    lags_grid: List[Union[int, List[int]]],
    metric: str = 'mean_squared_error',
    verbose: bool = False
) -> Tuple[pd.DataFrame, ForecasterRecursive]:
    """
    Train a ForecasterAutoreg model using grid search.

    Parameters:
        y (pd.Series): Time series data (indexed by datetime).
        initial_train_size (int): Initial size of training set.
        param_grid (dict): Hyperparameters for the regressor.
        lags_grid (list): List of lags or lag combinations to try.
        metric (str): Metric to evaluate (e.g. 'mean_squared_error').
        verbose (bool): Whether to show progress.

    Returns:
        Tuple containing:
            - results_df (pd.DataFrame): Grid search results.
            - best_model (ForecasterAutoreg): Best fitted model.
    """
    forecaster = ForecasterRecursive(
        regressor=DecisionTreeRegressor(random_state=123),
        lags=30  # placeholder, will be overridden by grid
    )

    results = grid_search_forecaster(
        forecaster=forecaster,
        y=y,
        param_grid=param_grid,
        lags_grid=lags_grid,
        refit=True,
        metric=metric,
        initial_train_size=initial_train_size,
        fixed_train_size=False,
        return_best=True,
        verbose=verbose
    )

    best_model = results.iloc[0]['forecaster']
    return results, best_model