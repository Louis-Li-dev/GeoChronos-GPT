import numpy as np
import pandas as pd
from typing import Union, Optional, List
import matplotlib.pyplot as plt
from geo_tools.geo_plot import *
from copy import deepcopy
import ot
from tqdm import tqdm
import os
import sys
# Add parent directory if needed
parent_dir = os.path.join(os.getcwd(), '..')
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from utility.loss import spatial_intensity_chamfer_distance_parallel
from sklearn.model_selection import KFold
import torch
import torch.nn as nn
import numpy as np
import imageio.v2 as imageio
from io import BytesIO

def scale_max_to_one(
    preds: np.ndarray,
    per_image: bool = True,
    eps: float = 1e-12
) -> np.ndarray:
    """
    Scale `preds` so that the maximum per "map" becomes 1.0.

    Args:
      preds: np.ndarray of shape
             - (B, 1, H, W)
             - (B,   H, W)
             - (  H,   W)
      per_image: if True, each H×W slice is normalized on its own.
                 if False, the entire array is normalized by its global max.
      eps:      to avoid divide-by-zero when a map is entirely zero.

    Returns:
      same shape as preds, but dtype float, with max(...)=1.0.
    """
    arr = np.array(preds, dtype=float)
    if per_image:
        if arr.ndim >= 3:
            # assume first axis is "batch" and everything else is spatial
            spatial_axes = tuple(range(1, arr.ndim))
            maxes = arr.max(axis=spatial_axes, keepdims=True)
        elif arr.ndim == 2:
            # just one map
            maxes = arr.max()
        else:
            raise ValueError(f"Cannot normalize array with ndim={arr.ndim}")
        # avoid zeros
        maxes = np.where(maxes < eps, 1.0, maxes)
        return arr / maxes
    else:
        # global normalization
        m = arr.max()
        if m < eps:
            m = 1.0
        return arr / m



def create_prediction_comparison_gif(
    actual,
    model_preds: dict = None,
    num_samples: int = 30,
    output_path: str = 'predictions_comparison.gif',
    duration: float = 1.0,
    figsize: tuple = None,
    cmap: str = 'inferno',
    loss_fn=None,
    save_frames_dir: str = None,
    big_grid_path: str = None,           # NEW: path to save the giant grid
    big_grid_figsize: tuple = None       # NEW: optional figsize for that grid
):
    """
    Build a GIF comparing `actual` vs. each array in `model_preds`, and
    optionally save each frame as a separate PNG and/or one giant grid.

    Parameters
    ----------
    actual : np.ndarray, shape (N, H, W)
        Ground‐truth images.
    model_preds : dict[str, np.ndarray], optional
        Mapping model name → array of shape (N, H, W).
    num_samples : int
        How many samples/frames to process.
    output_path : str
        Where to write the GIF.
    duration : float
        Seconds per frame in the GIF.
    figsize : tuple, optional
        Figure size for each frame; defaults to (4*(1+len(model_preds)), 4).
    cmap : str
        Matplotlib colormap.
    loss_fn : callable, optional
        If given, called as `loss_fn(pred, actual)`; otherwise uses
        `spatial_intensity_chamfer_distance_parallel`.
    save_frames_dir : str, optional
        If provided, each frame is also saved as
        `<save_frames_dir>/frame_0000.png`, etc.
    big_grid_path : str, optional
        If provided, a single giant grid of all samples × models
        is built and saved here.
    big_grid_figsize : tuple, optional
        Figure size for the big grid; defaults to (4*(cols), 4*(rows)).
    """
    if actual.ndim == 4 and actual.shape[1] == 1:
        actual = actual[:, 0, :, :]

    for name, arr in list(model_preds.items()):
        if arr.ndim == 4 and arr.shape[1] == 1:
            model_preds[name] = arr[:, 0, :, :]
    if model_preds is None:
        model_preds = {}

    n_cols = 1 + len(model_preds)
    if figsize is None:
        figsize = (4 * n_cols, 4)

    # ensure frame‐dir exists if requested
    if save_frames_dir:
        os.makedirs(save_frames_dir, exist_ok=True)

    # PREPARE indices we'll use
    total = actual.shape[0]
    k = min(num_samples, total)
    indices = list(np.random.choice(total, size=k, replace=False))
    # === NEW: build & save one giant grid ===
    if big_grid_path:
        n_rows = len(indices)
        # auto‐compute grid figsize if not provided
        if big_grid_figsize is None:
            big_grid_figsize = (4 * n_cols, 4 * n_rows)

        fig, axes = plt.subplots(n_rows, n_cols, figsize=big_grid_figsize)
        # if only one row or col, axes might be 1D—normalize to 2D
        if n_rows == 1:
            axes = axes[np.newaxis, :]
        if n_cols == 1:
            axes = axes[:, np.newaxis]

        for row_idx, i in enumerate(indices):
            act = actual[i]
            vmin = act.min()
            vmax = act.max() + .01

            # plot actual
            ax = axes[row_idx, 0]
            ax.imshow(act, cmap=cmap, vmin=vmin, vmax=vmax)
            if row_idx == 0:
                ax.set_title("Actual")
            ax.axis('off')

            # plot each model
            for col_idx, (name, pred_arr) in enumerate(model_preds.items(), start=1):
                pred = pred_arr[i]
                loss = (loss_fn(pred, act)
                        if loss_fn is not None
                        else spatial_intensity_chamfer_distance_parallel(pred, act))
                ax = axes[row_idx, col_idx]
                ax.imshow(pred, cmap=cmap, vmin=vmin, vmax=vmax)
                title = f"{name}\ROSA: {loss:.4f}"
                if row_idx == 0:
                    ax.set_title(title)
                else:
                    # for lower rows, just show loss
                    ax.set_title(f"{loss:.4f}")
                ax.axis('off')

        plt.tight_layout()
        os.makedirs(os.path.dirname(big_grid_path) or '.', exist_ok=True)
        fig.savefig(big_grid_path, bbox_inches='tight', pad_inches=0.2)
        plt.close(fig)
        print(f"Saved big grid to {big_grid_path}")

    # === UNCHANGED: build GIF and optional per‐frame PNGs ===
    frames = []
    for idx in tqdm(indices, desc="Making GIF frames"):
        act = actual[idx]
        frame_arrays = [act] + [pred[idx] for pred in model_preds.values()]
        vmin = min(arr.min() for arr in frame_arrays)
        vmax = max(arr.max() for arr in frame_arrays)

        fig, axes = plt.subplots(1, n_cols, figsize=figsize)
        axes[0].imshow(act, cmap=cmap, vmin=vmin, vmax=vmax)
        axes[0].set_title("Actual")
        axes[0].axis('off')

        for j, (name, pred_arr) in enumerate(model_preds.items(), start=1):
            pred = pred_arr[idx]
            loss = (loss_fn(pred, act)
                    if loss_fn is not None
                    else spatial_intensity_chamfer_distance_parallel(pred, act))
            axes[j].imshow(pred, cmap=cmap, vmin=vmin, vmax=vmax)
            axes[j].set_title(f"{name}\nLoss: {loss:.4f}")
            axes[j].axis('off')

        plt.tight_layout()

        if save_frames_dir:
            frame_path = os.path.join(save_frames_dir, f"frame_{idx:04d}.png")
            fig.savefig(frame_path, format='png', bbox_inches='tight', pad_inches=0.2)

        buf = BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight', pad_inches=0.2)
        plt.close(fig)
        buf.seek(0)
        frames.append(imageio.imread(buf))

    imageio.mimsave(output_path, frames, duration=duration)
    print(f"Saved GIF to {output_path}")
    if save_frames_dir:
        print(f"Also saved individual frames to '{save_frames_dir}/'")
def evaluate_models(
    models_dict,
    model_params_dict,
    metrics_dict,
    X_train,
    y_train,
    X_test=None,
    y_test=None,
    do_kfold=False,
    kfold_params={'n_splits': 5, 'shuffle': True, 'random_state': 42},
    width=32,  # Default width, replace with WIDTH from your context
    height=32  # Default height, replace with HEIGHT from your context
):
    """
    Evaluate models using k-fold cross-validation or direct train/test, computing specified metrics.

    Parameters:
    - models_dict (dict): {model_name: model_class} where model_class is callable to instantiate a model.
    - model_params_dict (dict): {model_name: {'init_params': dict, 'train_params': dict, 'target_shape': str}}
        - 'init_params': Parameters to initialize the model.
        - 'train_params': Parameters for model.fit (e.g., epochs, optimizer, criterion).
        - 'target_shape': '2d' for (n_samples, 1, width, height) or 'flat' for (n_samples, width*height).
    - metrics_dict (dict): {metric_name: callable} where callable takes (y_true, y_pred) as (n_samples, width, height).
    - X_train (np.ndarray): Training features.
    - y_train (np.ndarray): Training targets.
    - X_test (np.ndarray, optional): Test features, required if do_kfold=False.
    - y_test (np.ndarray, optional): Test targets, required if do_kfold=False.
    - do_kfold (bool): If True, perform k-fold CV; if False, train on X_train and test on X_test.
    - kfold_params (dict): Parameters for KFold (e.g., n_splits, shuffle, random_state).
    - width (int): Width of the 2D target images.
    - height (int): Height of the 2D target images.

    Returns:
    - dict: {
        'predictions': dict with keys 'Fold', 'Sample_Index', 'Actual', '{model_name}_Pred',
        'metrics': list of dicts with keys 'Model', 'Fold', '{metric_name}',
        'summary': dict with keys 'Model', '{metric_name}_mean', '{metric_name}_std' (for k-fold only)
    }
    """
    # Helper function to prepare target data
    def prepare_target(y, target_shape, width, height):
        if target_shape == '2d':
            return y.reshape(-1, 1, width, height)
        elif target_shape == 'flat':
            return y.reshape(-1, width * height)
        else:
            raise ValueError(f"Invalid target_shape: {target_shape}")

    # Helper function to reshape predictions to (n_samples, width, height)
    def reshape_predictions(y_pred, target_shape, width, height):
        if target_shape == '2d':
            return y_pred.squeeze(1)  # From (n_samples, 1, width, height) to (n_samples, width, height)
        elif target_shape == 'flat':
            return y_pred.reshape(-1, width, height)  # From (n_samples, width*height) to (n_samples, width, height)
        else:
            raise ValueError(f"Invalid target_shape: {target_shape}")

    if do_kfold:
        # Set up k-fold cross-validation
        kf = KFold(**kfold_params)

        # Initialize predictions dictionary
        predictions_dict = {
            'Fold': [],
            'Sample_Index': [],
            'Actual': [],  # List of (width, height) numpy arrays
        }
        for model_name in models_dict.keys():
            predictions_dict[f'{model_name}_Pred'] = []  # List of (width, height) numpy arrays

        # Initialize metrics list
        metrics_list = []

        for fold, (train_idx, val_idx) in enumerate(kf.split(X_train)):
            # Split data into training and validation folds
            X_train_fold = X_train[train_idx]
            y_train_fold = y_train[train_idx]
            X_val_fold = X_train[val_idx]
            y_val_fold = y_train[val_idx]

            # Reshape actual validation targets to 2D
            y_val_2d = y_val_fold.reshape(-1, width, height)

            for model_name, model_class in models_dict.items():
                # Get model parameters
                params = model_params_dict[model_name]
                target_shape = params['target_shape']
                init_params = params.get('init_params', {})
                train_params = params.get('train_params', {})

                # Instantiate a fresh model for each fold
                model = model_class(**init_params)
                # Prepare optimizer if it's a callable
                if 'optimizer' in train_params and callable(train_params['optimizer']):
                    optimizer = train_params['optimizer'](model)
                    fit_params = {k: v for k, v in train_params.items() if k != 'optimizer'}
                    fit_params['optimizer'] = optimizer
                else:
                    fit_params = train_params

                # Prepare target data
                y_train_fold_prepared = prepare_target(y_train_fold, target_shape, width, height)

                # Train the model
                model.fit(X_train_fold, y_train_fold_prepared, **fit_params)

                # Predict on validation fold
                y_pred = model.predict(X_val_fold)

                # Reshape predictions to (n_samples, width, height)
                y_pred_2d = reshape_predictions(y_pred, target_shape, width, height)

                # Compute metrics
                fold_metrics = {'Model': model_name, 'Fold': fold}
                for metric_name, metric_func in metrics_dict.items():
                    metric_value = metric_func(y_val_2d, y_pred_2d)
                    if isinstance(metric_value, np.ndarray):
                        metric_value = np.mean(metric_value)
                    fold_metrics[metric_name] = metric_value
                metrics_list.append(fold_metrics)

                # Store predictions for this model
                predictions_dict[f'{model_name}_Pred'].extend(y_pred_2d)

            # Store fold, sample index, and actual values (after all models to ensure order consistency)
            predictions_dict['Fold'].extend([fold] * len(val_idx))
            predictions_dict['Sample_Index'].extend(val_idx)
            predictions_dict['Actual'].extend(y_val_2d)

        # Compute summary statistics
        summary_dict = {'Model': list(models_dict.keys())}
        for metric_name in metrics_dict.keys():
            summary_dict[f'{metric_name}_mean'] = []
            summary_dict[f'{metric_name}_std'] = []
            for model_name in models_dict.keys():
                model_metrics = [m[metric_name] for m in metrics_list if m['Model'] == model_name]
                mean_value = np.mean(model_metrics)
                std_value = np.std(model_metrics)
                summary_dict[f'{metric_name}_mean'].append(mean_value)
                summary_dict[f'{metric_name}_std'].append(std_value)

        return {
            'predictions': predictions_dict,
            'metrics': metrics_list,
            'summary': summary_dict
        }

    else:
        # Direct training/testing
        if X_test is None or y_test is None:
            raise ValueError("X_test and y_test must be provided when do_kfold=False")

        # Reshape actual test targets to 2D
        y_test_2d = y_test.reshape(-1, width, height)

        # Initialize predictions dictionary
        predictions_dict = {
            'Fold': ['Test'] * len(y_test),
            'Sample_Index': list(range(len(y_test))),
            'Actual': list(y_test_2d),  # List of (width, height) numpy arrays
        }
        for model_name in models_dict.keys():
            predictions_dict[f'{model_name}_Pred'] = []  # Will be populated below

        # Initialize metrics list
        metrics_list = []

        for model_name, model_class in models_dict.items():
            # Get model parameters
            params = model_params_dict[model_name]
            target_shape = params['target_shape']
            init_params = params.get('init_params', {})
            train_params = params.get('train_params', {})

            # Instantiate the model
            model = model_class(**init_params)
            # Prepare optimizer if it's a callable
            if 'optimizer' in train_params and callable(train_params['optimizer']):
                optimizer = train_params['optimizer'](model)
                fit_params = {k: v for k, v in train_params.items() if k != 'optimizer'}
                fit_params['optimizer'] = optimizer
            else:
                fit_params = train_params

            # Prepare target data
            y_train_prepared = prepare_target(y_train, target_shape, width, height)

            # Train the model on full training data
            model.fit(X_train, y_train_prepared, **fit_params)

            # Predict on test set
            y_pred = model.predict(X_test)

            # Reshape predictions to (n_samples, width, height)
            y_pred_2d = reshape_predictions(y_pred, target_shape, width, height)

            # Compute metrics
            test_metrics = {'Model': model_name, 'Fold': 'Test'}
            for metric_name, metric_func in metrics_dict.items():
                metric_value = metric_func(y_test_2d, y_pred_2d)
                if isinstance(metric_value, np.ndarray):
                    metric_value = np.mean(metric_value)
                test_metrics[metric_name] = metric_value
            metrics_list.append(test_metrics)

            # Store predictions
            predictions_dict[f'{model_name}_Pred'] = list(y_pred_2d)

        return {
            'predictions': predictions_dict,
            'metrics': metrics_list
        }
def smape(y_true, y_pred):
    """Calculate Symmetric Mean Absolute Percentage Error"""
    return 100 * np.mean(2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred) + 1e-10))

def visualize_prediction(output_list: Union[np.ndarray, pd.DataFrame, List[Union[np.ndarray, pd.DataFrame]]],
                         actual_data: Union[np.ndarray, pd.DataFrame] = None,
                         idx: Optional[int] = None,
                         width: Optional[int] = None,
                         height: Optional[int] = None,
                         discretizer=None,
                         coordinator=None,
                         title: str = "Prediction - Japan",
                         plot_configs=None,
                         focus_on_center=False,
                         thresh=3/4,
                         ax: Optional[plt.Axes] = None):
    output_list = deepcopy(output_list)
    
    def process_input(data, idx=None):
        """Helper function to process input into a DataFrame."""
        if isinstance(data, np.ndarray):
            if idx is None or width is None or height is None or discretizer is None or coordinator is None:
                raise ValueError("idx, width, height, discretizer, and coordinator are required for NumPy array input.")
            tmp = data.reshape(-1, width, height)[idx]
            result = np.array(discretizer.inverse_transform(tmp))
            x, y = result[:, 0], result[:, 1]
            longitude, latitude = coordinator.inverse_transform(x, y)
            if isinstance(longitude, float): longitude = [longitude]
            if isinstance(latitude, float): latitude = [latitude]
            return pd.DataFrame({'Latitude': latitude, 'Longitude': longitude})
        elif isinstance(data, pd.DataFrame):
            return data
        else:
            raise ValueError("Input must be a NumPy array or pandas DataFrame.")

    # Default plot configs if none provided
    if plot_configs is None:
        plot_configs = {
            "overlay_density_background": True,
            "density_cmap": "inferno",
            "point_size": 0,
            "density_levels": 4,
            "bw_method": 0.3
        }
    
    tmp_list = deepcopy([output for output in output_list])
    if focus_on_center:
        for i in range(len(tmp_list)):
            tmp_list[i][idx][np.where(tmp_list[i][idx] < (tmp_list[i][idx].max() * thresh))] = 0

    plotters = []
    titles = []

    # If actual_data is not provided, treat the first element of output_list as actual_data
    if actual_data is None and len(tmp_list) > 0:
        actual_df = process_input(tmp_list[0], idx if idx is not None else 0)
        prediction_list = tmp_list[1:]  # Remaining items are predictions
        original_list = output_list[1:]
    else:
        raise Exception("Actual data or valid output list required.")

    # Add the actual data plotter first
    if actual_df is not None:
        actual_plotter = CoordinatePlotter(actual_df)
        plotters.append(actual_plotter)
        titles.append("Actual - Japan")

    # Process predictions
    for i, (item, o_item) in enumerate(zip(prediction_list, original_list)):
        tmp_df = process_input(item, idx if idx is not None else i % len(prediction_list))
        plotters.append(CoordinatePlotter(tmp_df))
        if actual_df is not None:
            # Calculate EMD for each prediction
            error = compute_emd(output_list[0][idx], o_item[idx], grid_size=width)
            titles.append(f"{title} {i+1} (EMD: {error:.2f})")
        else:
            titles.append(f"{title} {i+1}")

    # Update plot_configs with individual titles
    subplot_layout = (1, len(plotters)) if len(plotters) > 1 else (1, 1)
    if isinstance(plot_configs, dict):
        plot_configs_list = [plot_configs.copy() for _ in range(len(plotters))]
    else:
        plot_configs_list = plot_configs or [{} for _ in range(len(plotters))]
    for i, config in enumerate(plot_configs_list):
        config["title"] = titles[i]

    plot_multiple(plotters, subplot_layout, plot_configs_list, figsize=(len(plotters) * 6, 6))
    return None

def compute_emd(actual, prediction, grid_size):
    """
    Compute the Earth Mover's Distance (EMD) between two distributions.
    """
    def normalize_distribution(dist):
        # Clip negative values to zero
        dist = np.clip(dist, 0, None)
        dist_flat = dist.flatten()
        nonzero_mask = dist_flat > 0
        nonzero_values = dist_flat[nonzero_mask]
        if len(nonzero_values) == 0:
            raise ValueError("Distribution has no nonzero values to normalize.")
        total_mass = np.sum(nonzero_values)
        if total_mass == 0:
            raise ValueError("Total mass of nonzero values is zero.")
        dist_flat[nonzero_mask] = nonzero_values / total_mass
        return dist_flat.reshape(dist.shape)

    actual_norm = normalize_distribution(actual)
    prediction_norm = normalize_distribution(prediction)
    
    actual_flat = actual_norm.flatten()
    prediction_flat = prediction_norm.flatten()
    
    nonzero_indices = np.where((actual_flat > 0) | (prediction_flat > 0))[0]
    if len(nonzero_indices) == 0:
        raise ValueError("No nonzero cells found in either distribution.")
    
    actual_nonzero = actual_flat[nonzero_indices]
    prediction_nonzero = prediction_flat[nonzero_indices]
    
    # Re-normalize the filtered arrays so they both sum to 1
    actual_nonzero = actual_nonzero / np.sum(actual_nonzero)
    prediction_nonzero = prediction_nonzero / np.sum(prediction_nonzero)
    
    coords = np.array([(i // grid_size, i % grid_size) for i in nonzero_indices])
    cost_matrix = ot.dist(coords, coords, metric='euclidean')
    
    emd_value = ot.emd2(actual_nonzero, prediction_nonzero, cost_matrix)
    return emd_value

def compute_batch_emd(actual_batch, prediction_batch, width, height):
    """
    Compute the mean EMD loss for a batch of samples.
    
    Parameters:
    - actual_batch: numpy array of shape (N, WIDTH, HEIGHT), the actual distributions
    - prediction_batch: numpy array of shape (N, WIDTH, HEIGHT), the predicted distributions
    - width: integer, the width of the grid
    - height: integer, the height of the grid
    
    Returns:
    - mean_emd: float, the mean EMD loss across the batch
    """
    if actual_batch.shape != prediction_batch.shape:
        raise ValueError("Actual and prediction batches must have the same shape.")
    
    N = actual_batch.shape[0]  # Number of samples in the batch
    emd_scores = []
    
    for i in tqdm(range(N)):
        actual_sample = actual_batch[i]  # Shape: (WIDTH, HEIGHT)
        prediction_sample = prediction_batch[i]  # Shape: (WIDTH, HEIGHT)
        emd_score = compute_emd(actual_sample, prediction_sample, grid_size=width)
        emd_scores.append(emd_score)
    
    mean_emd = np.mean(emd_scores)
    return mean_emd
