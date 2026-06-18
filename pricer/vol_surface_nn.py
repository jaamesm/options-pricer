"""Neural-network implied volatility surface interpolator.

Trains a feedforward neural network on (log-moneyness, T [, r, q]) -> IV
and provides comparison utilities against the bicubic spline implemented
in :func:`~pricer.implied_vol.fit_surface`.

Typical workflow
----------------
>>> from pricer.vol_surface_nn import (
...     load_and_prepare, walk_forward_split, VolSurfaceNet,
...     VolSurfaceTrainer, evaluate, predict_surface,
... )
>>> df = load_and_prepare("data/SPY_chain.csv")
>>> train, val, test = walk_forward_split(df)
>>> model = VolSurfaceNet(input_dim=2, hidden_sizes=[64, 64], dropout_rate=0.1)
>>> trainer = VolSurfaceTrainer(model)
>>> history = trainer.fit(train, val)
>>> metrics = evaluate(trainer, test)
>>> M, T_grid, IV = predict_surface(trainer)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from pricer.implied_vol import solve_chain

__all__ = [
    "FEATURE_COLS",
    "FEATURE_COLS_RATES",
    "load_and_prepare",
    "walk_forward_split",
    "VolSurfaceNet",
    "VolSurfaceTrainer",
    "evaluate",
    "plot_evaluation",
    "plot_loss_curves",
    "predict_surface",
    "plot_surfaces",
]

logger = logging.getLogger(__name__)

# Feature column name sets; exported so callers and tests can reference them.
FEATURE_COLS: list[str] = ["log_moneyness", "T"]
FEATURE_COLS_RATES: list[str] = ["log_moneyness", "T", "r", "q"]

_MIN_IV: float = 0.01
_MAX_IV: float = 5.0
_MIN_EXPIRY_DAYS: int = 7
_DEFAULT_MONEYNESS_BAND: tuple[float, float] = (-0.5, 0.5)


# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------


def load_and_prepare(
    chain_path: str | Path,
    kind: str = "call",
    use_rates: bool = False,
    moneyness_band: tuple[float, float] = _DEFAULT_MONEYNESS_BAND,
    min_expiry_days: int = _MIN_EXPIRY_DAYS,
) -> pd.DataFrame:
    """Load a chain CSV, solve implied volatilities, and engineer features.

    Reads a chain file produced by the daily data pipeline, solves IVs
    using :func:`~pricer.implied_vol.solve_chain`, filters to liquid and
    tractable contracts, then returns a DataFrame ready for model training.

    Parameters
    ----------
    chain_path:
        Path to a ``{ticker}_chain.csv`` file.
    kind:
        ``"call"`` or ``"put"`` — filters the ``kind`` column.
    use_rates:
        If ``True``, include ``r`` and ``q`` as additional input features.
    moneyness_band:
        ``(lo, hi)`` bounds on ``ln(K/S)``.  Contracts outside this range
        are dropped; deep-ITM and far-OTM wings have unstable IV solves.
    min_expiry_days:
        Minimum calendar days to expiry.  Very short-dated options carry
        extreme gamma and micro-structure noise that harms training.

    Returns
    -------
    pd.DataFrame
        Columns: ``log_moneyness``, ``T``, [``r``, ``q``], ``iv``,
        ``expiry_date``.

    Raises
    ------
    ValueError
        If no contracts survive filtering.
    """
    df = pd.read_csv(chain_path, parse_dates=["expiry_date"])
    df = df[df["kind"] == kind].copy()

    # solve_chain() returns a pd.Series aligned with df.index
    df["iv"] = solve_chain(df)

    # Drop failed solves and extreme / obviously wrong vols
    df = df[df["iv"].between(_MIN_IV, _MAX_IV)]

    # Feature engineering ------------------------------------------------
    df["log_moneyness"] = np.log(df["strike"] / df["S"])

    # T is already stored as years-to-expiry in the ``expiry`` column of the
    # chain CSV (the value yfinance computes at fetch time).  Use it directly
    # rather than recomputing from expiry_date to avoid tz-aware subtraction.
    df["T"] = df["expiry"].astype(float)

    # Normalise expiry_date to date-only; used only for the walk-forward split.
    df["expiry_date"] = pd.to_datetime(df["expiry_date"]).dt.normalize()

    # Filters ------------------------------------------------------------
    min_T = min_expiry_days / 365.25
    df = df[df["T"] > min_T]

    lo, hi = moneyness_band
    df = df[df["log_moneyness"].between(lo, hi)]

    feature_cols = FEATURE_COLS_RATES if use_rates else FEATURE_COLS
    keep = feature_cols + ["iv", "expiry_date"]
    df = df[keep].dropna().reset_index(drop=True)

    if df.empty:
        raise ValueError(
            f"No contracts survived filtering in {chain_path}. "
            "Check moneyness_band, min_expiry_days, and kind."
        )

    logger.info(
        "Prepared %d contracts from %s (kind=%s, features=%s)",
        len(df),
        chain_path,
        kind,
        feature_cols,
    )
    return df


def walk_forward_split(
    df: pd.DataFrame,
    test_frac: float = 0.15,
    val_frac: float = 0.15,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split by expiry date to avoid look-ahead bias.

    Contracts are sorted chronologically by expiry date and partitioned
    into three contiguous windows.  The test set contains the *most
    recent* expiries so the evaluation scenario mirrors production use:
    train on past surfaces, predict future surfaces.

    A random split would let the model train on March options and test
    on February options, giving optimistically low error that won't
    appear in deployment.

    Parameters
    ----------
    df:
        Prepared DataFrame from :func:`load_and_prepare`.
    test_frac:
        Fraction of unique expiry dates reserved for the test set.
    val_frac:
        Fraction of unique expiry dates reserved for early-stopping
        validation.

    Returns
    -------
    train, val, test : pd.DataFrame
        Three non-overlapping DataFrames in chronological order.

    Raises
    ------
    ValueError
        If ``df`` contains fewer than 3 distinct expiry dates.
    """
    expiry_dates = sorted(df["expiry_date"].unique())
    n = len(expiry_dates)
    if n < 3:
        raise ValueError(
            f"Need at least 3 distinct expiry dates for a walk-forward "
            f"split; found {n}."
        )

    n_test = max(1, int(n * test_frac))
    n_val = max(1, int(n * val_frac))
    # Guard against val + test consuming all dates
    n_val = min(n_val, n - n_test - 1)

    test_start = expiry_dates[n - n_test]
    val_start = expiry_dates[n - n_test - n_val]

    train = df[df["expiry_date"] < val_start].copy()
    val = df[
        (df["expiry_date"] >= val_start) & (df["expiry_date"] < test_start)
    ].copy()
    test = df[df["expiry_date"] >= test_start].copy()

    logger.info(
        "Walk-forward split — train: %d rows, val: %d rows, test: %d rows",
        len(train),
        len(val),
        len(test),
    )
    return train, val, test


# ---------------------------------------------------------------------------
# Neural network architecture
# ---------------------------------------------------------------------------


class VolSurfaceNet(nn.Module):
    """Feedforward network mapping (moneyness, T [, r, q]) to implied vol.

    Architecture
    ------------
    ``input_dim`` → [Linear → ReLU → Dropout] × len(hidden_sizes) → Linear(1)

    The final layer has no activation, so the output lives on the real
    line.  Training targets are StandardScaler-normalised IVs, so the
    raw output can be negative; the inverse-transform at inference
    restores the physical range.

    Parameters
    ----------
    input_dim:
        Number of input features (2 for ``FEATURE_COLS``, 4 for
        ``FEATURE_COLS_RATES``).
    hidden_sizes:
        Sequence of hidden-layer widths.  The default ``[64, 64]`` gives
        roughly 4,500 parameters — adequate for a few thousand contracts.
    dropout_rate:
        Dropout probability applied after each ReLU.  Acts as a per-neuron
        Bernoulli mask during training; disabled at inference.
    """

    def __init__(
        self,
        input_dim: int = 2,
        hidden_sizes: Sequence[int] = (64, 64),
        dropout_rate: float = 0.1,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        in_size = input_dim
        for h in hidden_sizes:
            layers += [
                nn.Linear(in_size, h),
                nn.ReLU(),
                nn.Dropout(p=dropout_rate),
            ]
            in_size = h
        layers.append(nn.Linear(in_size, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x:
            Float tensor of shape ``(batch_size, input_dim)``.

        Returns
        -------
        torch.Tensor
            Shape ``(batch_size,)`` — predicted normalised implied vol.
        """
        return self.net(x).squeeze(-1)


# ---------------------------------------------------------------------------
# Early stopping
# ---------------------------------------------------------------------------


class _EarlyStopping:
    """Halt training when validation loss stops improving.

    Stores the best model weights so they can be restored after stopping.

    Parameters
    ----------
    patience:
        Number of epochs to wait after the last improvement.
    min_delta:
        Minimum absolute improvement that counts as a new best.
    """

    def __init__(self, patience: int = 20, min_delta: float = 1e-6) -> None:
        self.patience = patience
        self.min_delta = min_delta
        self._best: float = float("inf")
        self._counter: int = 0
        self.best_state: dict | None = None

    def step(self, val_loss: float, model: nn.Module) -> bool:
        """Update state and return ``True`` when training should stop."""
        if val_loss < self._best - self.min_delta:
            self._best = val_loss
            self._counter = 0
            # Store a CPU copy of the state dict so it's safe after
            # subsequent backward passes on GPU.
            self.best_state = {
                k: v.cpu().clone() for k, v in model.state_dict().items()
            }
        else:
            self._counter += 1
        return self._counter >= self.patience


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------


class VolSurfaceTrainer:
    """Manages normalisation, the training loop, and inference.

    Scalers are fit on the training set *only*.  The same fitted scalers
    are applied to validation, test, and any future inference inputs —
    preventing leakage of test statistics into the normalisation.

    At inference time the scaler pipeline is:
    ``raw input  -> scaler_X.transform -> network -> scaler_y.inverse_transform -> IV``

    Parameters
    ----------
    model:
        An untrained :class:`VolSurfaceNet` instance.
    lr:
        Initial Adam learning rate.
    patience:
        Early-stopping patience in epochs.
    max_epochs:
        Hard ceiling on training epochs.
    batch_size:
        Mini-batch size for SGD.
    device:
        ``"cpu"`` or ``"cuda"``.  Defaults to CUDA when available.
    """

    def __init__(
        self,
        model: VolSurfaceNet,
        lr: float = 1e-3,
        patience: int = 30,
        max_epochs: int = 500,
        batch_size: int = 256,
        device: str | None = None,
    ) -> None:
        self.model = model
        self.lr = lr
        self.patience = patience
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.device = torch.device(
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.model.to(self.device)

        self.optimiser = torch.optim.Adam(model.parameters(), lr=lr)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimiser,
            patience=max(1, patience // 3),
            factor=0.5,
        )
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()
        self._feature_cols: list[str] = []

    # ------------------------------------------------------------------
    def fit(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        feature_cols: list[str] | None = None,
    ) -> dict[str, list[float]]:
        """Fit scalers and run the training loop.

        Scalers are fit on ``train_df`` only.  Early stopping monitors
        the validation loss; at the end the best weights are restored.

        Parameters
        ----------
        train_df:
            Training split from :func:`walk_forward_split`.
        val_df:
            Validation split used for early stopping.
        feature_cols:
            Feature column names.  Defaults to ``FEATURE_COLS``.

        Returns
        -------
        dict
            ``{"train_loss": [...], "val_loss": [...]}`` — one float per
            epoch, in MSE on normalised targets.
        """
        if feature_cols is None:
            feature_cols = FEATURE_COLS
        self._feature_cols = feature_cols

        X_tr = self.scaler_X.fit_transform(train_df[feature_cols].values)
        y_tr = self.scaler_y.fit_transform(
            train_df[["iv"]].values
        ).ravel()

        X_vl = self.scaler_X.transform(val_df[feature_cols].values)
        y_vl = self.scaler_y.transform(val_df[["iv"]].values).ravel()

        train_loader = self._make_loader(X_tr, y_tr, shuffle=True)
        val_loader = self._make_loader(X_vl, y_vl, shuffle=False)

        criterion = nn.MSELoss()
        stopper = _EarlyStopping(patience=self.patience)
        history: dict[str, list[float]] = {"train_loss": [], "val_loss": []}

        for epoch in range(self.max_epochs):
            train_loss = self._run_epoch(train_loader, criterion, train=True)
            val_loss = self._run_epoch(val_loader, criterion, train=False)

            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)

            self.scheduler.step(val_loss)

            if stopper.step(val_loss, self.model):
                logger.info("Early stopping triggered at epoch %d.", epoch + 1)
                break

        # Restore the checkpoint with the best validation loss
        if stopper.best_state is not None:
            self.model.load_state_dict(stopper.best_state)
        self.model.eval()

        return history

    # ------------------------------------------------------------------
    def predict(
        self,
        df: pd.DataFrame,
        feature_cols: list[str] | None = None,
    ) -> np.ndarray:
        """Return IV predictions on the original vol scale.

        Applies the fitted ``scaler_X`` to inputs and ``scaler_y``
        inverse-transform to outputs, so the returned array is in the
        same units as the raw IV column (e.g. 0.20 for 20 % vol).

        Parameters
        ----------
        df:
            DataFrame containing the same feature columns used in
            :meth:`fit`.
        feature_cols:
            Override feature columns (must match training columns).

        Returns
        -------
        np.ndarray
            Shape ``(n,)`` — implied vol predictions.
        """
        if feature_cols is None:
            feature_cols = self._feature_cols or FEATURE_COLS

        X = self.scaler_X.transform(df[feature_cols].values)
        x_t = torch.tensor(X, dtype=torch.float32).to(self.device)

        self.model.eval()
        with torch.no_grad():
            y_norm = self.model(x_t).cpu().numpy()

        return self.scaler_y.inverse_transform(
            y_norm.reshape(-1, 1)
        ).ravel()

    # ------------------------------------------------------------------
    def _make_loader(
        self, X: np.ndarray, y: np.ndarray, shuffle: bool
    ) -> DataLoader:
        X_t = torch.tensor(X, dtype=torch.float32)
        y_t = torch.tensor(y, dtype=torch.float32)
        dataset = TensorDataset(X_t, y_t)
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=shuffle)

    def _run_epoch(
        self,
        loader: DataLoader,
        criterion: nn.Module,
        train: bool,
    ) -> float:
        if train:
            self.model.train()
        else:
            self.model.eval()

        total_loss = 0.0
        n_batches = 0

        for X_batch, y_batch in loader:
            X_batch = X_batch.to(self.device)
            y_batch = y_batch.to(self.device)

            if train:
                self.optimiser.zero_grad()
                preds = self.model(X_batch)
                loss = criterion(preds, y_batch)
                loss.backward()
                self.optimiser.step()
            else:
                with torch.no_grad():
                    preds = self.model(X_batch)
                    loss = criterion(preds, y_batch)

            total_loss += loss.item()
            n_batches += 1

        return total_loss / max(n_batches, 1)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def evaluate(
    trainer: VolSurfaceTrainer,
    test_df: pd.DataFrame,
    feature_cols: list[str] | None = None,
    spline_fn: Callable[[np.ndarray, np.ndarray], np.ndarray] | None = None,
) -> dict[str, dict[str, float]]:
    """Compare NN to a reference spline on the held-out test set.

    All metrics are in vol-point units (i.e. 0.01 = 1 vol point) and
    are computed on the inverse-transformed predictions, not on the
    normalised targets.

    Parameters
    ----------
    trainer:
        A fitted :class:`VolSurfaceTrainer`.
    test_df:
        Test split from :func:`walk_forward_split`.
    feature_cols:
        Feature columns — must match those used in :meth:`VolSurfaceTrainer.fit`.
    spline_fn:
        Optional callable ``(log_moneyness_arr, T_arr) -> iv_arr``.
        When provided, spline metrics are included in the output under
        key ``"spline"``.  Wrap :func:`~pricer.implied_vol.fit_surface`
        as needed to match this signature.

    Returns
    -------
    dict
        ``{"nn": {"rmse": ..., "mae": ..., "max_ae": ...},
           "spline": {...}}``
        (``"spline"`` key absent when ``spline_fn`` is ``None``).
    """
    if feature_cols is None:
        feature_cols = trainer._feature_cols or FEATURE_COLS

    y_true = test_df["iv"].values
    y_nn = trainer.predict(test_df, feature_cols)
    results: dict[str, dict[str, float]] = {"nn": _metrics(y_true, y_nn)}

    if spline_fn is not None:
        m = test_df["log_moneyness"].values
        T = test_df["T"].values
        y_spline = np.asarray(spline_fn(m, T), dtype=float)
        results["spline"] = _metrics(y_true, y_spline)

    return results


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Compute RMSE, MAE, and max absolute error."""
    residuals = np.asarray(y_pred, dtype=float) - np.asarray(y_true, dtype=float)
    return {
        "rmse": float(np.sqrt(np.mean(residuals**2))),
        "mae": float(np.mean(np.abs(residuals))),
        "max_ae": float(np.max(np.abs(residuals))),
    }


def plot_evaluation(
    trainer: VolSurfaceTrainer,
    test_df: pd.DataFrame,
    feature_cols: list[str] | None = None,
    spline_fn: Callable[[np.ndarray, np.ndarray], np.ndarray] | None = None,
) -> plt.Figure:
    """Three-panel evaluation figure for the test set.

    Panels:
    1. Predicted vs actual IV scatter with the 45-degree line.
    2. Residuals (pred - actual) vs log-moneyness ln(K/S).
    3. Residuals vs time to expiry T.

    A systematic pattern in panels 2 or 3 indicates the model is
    missing structure — e.g. the NN is underestimating skew at OTM
    puts (negative moneyness), or is badly calibrated at short expiry.

    Parameters
    ----------
    trainer:
        A fitted :class:`VolSurfaceTrainer`.
    test_df:
        Test split.
    feature_cols:
        Feature columns.
    spline_fn:
        Optional spline callable for comparison (same signature as in
        :func:`evaluate`).

    Returns
    -------
    matplotlib.figure.Figure
    """
    if feature_cols is None:
        feature_cols = trainer._feature_cols or FEATURE_COLS

    y_true = test_df["iv"].values
    y_nn = trainer.predict(test_df, feature_cols)
    m = test_df["log_moneyness"].values
    T = test_df["T"].values

    has_spline = spline_fn is not None
    y_spline: np.ndarray | None = None
    if has_spline:
        y_spline = np.asarray(spline_fn(m, T), dtype=float)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Panel 1: predicted vs actual
    ax = axes[0]
    lo, hi = float(y_true.min()), float(y_true.max())
    ax.scatter(y_true, y_nn, s=8, alpha=0.4, label="NN")
    if has_spline and y_spline is not None:
        ax.scatter(y_true, y_spline, s=8, alpha=0.3, marker="x", label="Spline")
        ax.legend(markerscale=2)
    ax.plot([lo, hi], [lo, hi], "k--", linewidth=0.8, label="_nolabel")
    ax.set_xlabel("Actual IV")
    ax.set_ylabel("Predicted IV")
    ax.set_title("Predicted vs Actual")

    # Panel 2: residuals vs moneyness
    ax = axes[1]
    res_nn = y_nn - y_true
    ax.scatter(m, res_nn, s=8, alpha=0.4, label="NN")
    if has_spline and y_spline is not None:
        ax.scatter(m, y_spline - y_true, s=8, alpha=0.3, marker="x", label="Spline")
        ax.legend(markerscale=2)
    ax.axhline(0, color="k", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Log-moneyness  ln(K/S)")
    ax.set_ylabel("Residual (pred \u2212 actual) IV")
    ax.set_title("Residuals vs Moneyness")

    # Panel 3: residuals vs T
    ax = axes[2]
    ax.scatter(T, res_nn, s=8, alpha=0.4, label="NN")
    if has_spline and y_spline is not None:
        ax.scatter(T, y_spline - y_true, s=8, alpha=0.3, marker="x", label="Spline")
        ax.legend(markerscale=2)
    ax.axhline(0, color="k", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Time to expiry T (years)")
    ax.set_ylabel("Residual IV")
    ax.set_title("Residuals vs Expiry")

    fig.suptitle("NN Vol Surface — Test Set Evaluation", y=1.01)
    fig.tight_layout()
    return fig


def plot_loss_curves(history: dict[str, list[float]]) -> plt.Figure:
    """Plot training and validation loss curves on a log scale.

    A large gap between train and val loss indicates over-fitting; both
    curves plateauing at a high level indicates under-fitting.

    Parameters
    ----------
    history:
        Dict returned by :meth:`VolSurfaceTrainer.fit`.

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, ax = plt.subplots(figsize=(8, 4))
    epochs = range(1, len(history["train_loss"]) + 1)
    ax.semilogy(epochs, history["train_loss"], label="Train")
    ax.semilogy(epochs, history["val_loss"], label="Validation")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE loss (normalised, log scale)")
    ax.set_title("Training and Validation Loss")
    ax.legend()
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Surface inference and comparison
# ---------------------------------------------------------------------------


def predict_surface(
    trainer: VolSurfaceTrainer,
    feature_cols: list[str] | None = None,
    moneyness_range: tuple[float, float] = (-0.3, 0.3),
    T_range: tuple[float, float] = (0.05, 2.0),
    n_points: int = 50,
    r: float = 0.05,
    q: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate the fitted NN on a regular (log-moneyness, T) grid.

    Use the returned arrays directly with :func:`plot_surfaces` or any
    3-D plotting routine.

    Parameters
    ----------
    trainer:
        A fitted :class:`VolSurfaceTrainer`.
    feature_cols:
        Feature columns used in training.
    moneyness_range:
        ``(lo, hi)`` for the log-moneyness axis.
    T_range:
        ``(lo, hi)`` for the T axis in years.
    n_points:
        Grid resolution along each axis — total grid is ``n_points ** 2``.
    r:
        Risk-free rate passed to the feature vector when ``"r"`` is in
        ``feature_cols``.
    q:
        Dividend yield, same usage as ``r``.

    Returns
    -------
    M, T_grid, IV_grid : np.ndarray
        Each array has shape ``(n_points, n_points)``.
    """
    if feature_cols is None:
        feature_cols = trainer._feature_cols or FEATURE_COLS

    m_vals = np.linspace(*moneyness_range, n_points)
    t_vals = np.linspace(*T_range, n_points)
    M, T_grid = np.meshgrid(m_vals, t_vals)

    flat_m = M.ravel()
    flat_T = T_grid.ravel()

    data: dict[str, np.ndarray] = {
        "log_moneyness": flat_m,
        "T": flat_T,
    }
    if "r" in feature_cols:
        data["r"] = np.full_like(flat_m, r)
    if "q" in feature_cols:
        data["q"] = np.full_like(flat_m, q)

    grid_df = pd.DataFrame(data)
    iv_flat = trainer.predict(grid_df, feature_cols)
    IV_grid = iv_flat.reshape(n_points, n_points)

    return M, T_grid, IV_grid


def plot_surfaces(
    nn_result: tuple[np.ndarray, np.ndarray, np.ndarray],
    spline_fn: Callable[[np.ndarray, np.ndarray], np.ndarray] | None = None,
) -> plt.Figure:
    """Plot the NN vol surface, optionally alongside the spline surface.

    Parameters
    ----------
    nn_result:
        ``(M, T_grid, IV_grid)`` from :func:`predict_surface`.
    spline_fn:
        Optional callable ``(log_moneyness_2d, T_2d) -> iv_2d``.

    Returns
    -------
    matplotlib.figure.Figure
    """
    M, T_grid, IV_nn = nn_result
    n_cols = 2 if spline_fn is not None else 1
    fig, axes = plt.subplots(
        1,
        n_cols,
        figsize=(7 * n_cols, 6),
        subplot_kw={"projection": "3d"},
    )
    if n_cols == 1:
        axes = [axes]

    _draw_surface(axes[0], M, T_grid, IV_nn, "NN Vol Surface")

    if spline_fn is not None:
        IV_spline = np.asarray(spline_fn(M, T_grid), dtype=float)
        _draw_surface(axes[1], M, T_grid, IV_spline, "Spline Vol Surface")

    fig.tight_layout()
    return fig


def _draw_surface(
    ax: plt.Axes,
    M: np.ndarray,
    T: np.ndarray,
    IV: np.ndarray,
    title: str,
) -> None:
    """Render a single 3-D vol surface panel."""
    ax.plot_surface(M, T, IV, cmap="viridis", alpha=0.85, edgecolor="none")
    ax.set_xlabel("ln(K/S)")
    ax.set_ylabel("T (years)")
    ax.set_zlabel("Implied Vol")
    ax.set_title(title)
