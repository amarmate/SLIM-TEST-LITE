import numpy as np
from sklearn.metrics import precision_recall_fscore_support, f1_score


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

def calc_scores_from_summary(class_summary):
    """
    class_summary: np.array, shape=(n_true_classes, n_predicted_classes)
    Gibt accuracy, macro_f1 und weighted_f1 zurück.
    """
    
    # 1) Pad matrix auf quadratisch (n_true_classes × n_true_classes)
    n_true, n_pred = class_summary.shape
    if n_pred < n_true:
        padding = np.zeros((n_true, n_true - n_pred), dtype=int)
        cm = np.hstack([class_summary, padding])
    else:
        cm = class_summary[:, :n_true]

    # 2) Rekonstruiere y_true, y_pred
    y_true, y_pred = [], []
    for true_label in range(n_true):
        for pred_label in range(n_true):
            cnt = cm[true_label, pred_label]
            y_true.extend([true_label] * cnt)
            y_pred.extend([pred_label] * cnt)

    # 3) Berechne Scores
    accuracy = np.sum(np.diag(cm)) / np.sum(cm) if np.sum(cm) > 0 else 0.0
    macro_f1    = f1_score(y_true, y_pred, average='macro',    labels=list(range(n_true)), zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, average='weighted', labels=list(range(n_true)), zero_division=0)

    return accuracy, macro_f1, weighted_f1
