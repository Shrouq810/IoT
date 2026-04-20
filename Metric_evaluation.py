"""
Title:Hydrologic–Hydrodynamic Flood Inundation Modeling Enhanced by IoT Water-Level Measurements in a Data-Scarce Basin
Author: Shrouq abuismail, github: ShrouqAbuismail
Date:20/04/2026
Description: Evaluation metrics for the paper titled "Hydrologic–Hydrodynamic Flood Inundation Modeling Enhanced by IoT Water-Level Measurements in a Data-Scarce Basin" 
"""
# =============================================================================
# EVENT-LEVEL EVALUATION METRICS
# =============================================================================
# Metrics computed per grid cell over a flood event (time series).
# Each function returns a list of values across selected cells.
# =============================================================================
import numpy as np
def rmse_median(true_event, pred_event, cell_indices):
    """RMSE per cell over the event."""
    rmse_cells = []

    for (i, j) in cell_indices:
        true_series = true_event[:, i, j]
        pred_series = pred_event[:, i, j]

        if np.all(true_series == 0) and np.all(pred_series == 0):
            continue

        mse = np.mean((true_series - pred_series) ** 2)
        rmse_cells.append(np.sqrt(mse))

    return rmse_cells
def bias_median(true_event, pred_event, cell_indices):
    """Mean bias per cell over the event."""
    biases = []

    for (i, j) in cell_indices:
        true_series = true_event[:, i, j]
        pred_series = pred_event[:, i, j]

        if np.all(true_series == 0) and np.all(pred_series == 0):
            continue

        biases.append(np.mean(pred_series - true_series))
    return biases
def TimeDiff(true_event,
              pred_event,
              cell_indices,
              dt=1.0,
              return_abs=True,
              eps=1e-6):
    """Peak timing difference per cell."""
    dt_cells = []

    for (i, j) in cell_indices:
        true_series = true_event[:, i, j]
        pred_series = pred_event[:, i, j]

        if np.max(true_series) <= eps or np.max(pred_series) <= eps:
            continue

        t_true_peak = int(np.argmax(true_series))
        t_pred_peak = int(np.argmax(pred_series))

        diff_time = (t_pred_peak - t_true_peak) * dt

        if return_abs:
            diff_time = abs(diff_time)

        dt_cells.append(diff_time)
    return dt_cells
def PeakDiff(true_event, pred_event, cell_indices, eps=1e-6):
    """Peak magnitude difference per cell (max over event)."""
    peak_diffs = []

    for (i, j) in cell_indices:
        true_series = true_event[:, i, j]
        pred_series = pred_event[:, i, j]

        if np.max(true_series) <= eps and np.max(pred_series) <= eps:
            continue

        true_peak = np.max(true_series)
        pred_peak = np.max(pred_series)

        peak_diffs.append(pred_peak - true_peak)

    return peak_diffs