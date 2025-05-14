import numpy as np

def mape(y_true, y_pred):
    """"Mean Absolute Percentage Error."""
    return np.mean(np.array((np.abs((y_true - y_pred) / y_true)))) * 100

def nrmse(y_true, y_pred):
    """Normalized RMSE."""
    range_y = y_true.max() - y_true.min()
    return (np.sqrt(np.mean(np.array((y_true - y_pred) ** 2))) / range_y)

def r_squared(y_true, y_pred):
    """R-squared."""
    # Convert to numpy arrays
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    ss_total = np.sum((y_true - np.mean(y_true)) ** 2)
    ss_residual = np.sum((y_true - y_pred) ** 2)
    return 1 - (ss_residual / ss_total)

def mae(y_true, y_pred):
    """Mean Absolute Error."""
    return np.mean(np.abs(np.array((y_true - y_pred))))

def standardized_rmse(y_true, y_pred):
    """Standardized RMSE."""
    std_y = np.std(np.array(y_true))
    return np.sqrt(np.mean(np.array((y_true - y_pred) ** 2))) / std_y

def rmse(y_true, y_pred):
    """Root Mean Squared Error."""
    return np.sqrt(np.mean(np.array((y_true - y_pred) ** 2)))