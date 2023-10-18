from typing import List

from sklearn.metrics import mean_absolute_error, mean_squared_error


def print_results(true_values: List, predicted_values: List) -> None:
    print(f"MSE: {mean_squared_error(true_values, predicted_values)}")
    print(f"RMSE: {mean_squared_error(true_values, predicted_values, squared=False)}")
    print(f"MAE: {mean_absolute_error(true_values, predicted_values)}")
