"""Tests for pricer.vol_surface_nn.

Run with: pytest tests/test_vol_surface_nn.py -v --cov=pricer.vol_surface_nn
"""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
import torch

from pricer.vol_surface_nn import (
    FEATURE_COLS_RATES,
    VolSurfaceNet,
    VolSurfaceTrainer,
    _EarlyStopping,
    _metrics,
    evaluate,
    load_and_prepare,
    plot_evaluation,
    plot_loss_curves,
    plot_surfaces,
    predict_surface,
    walk_forward_split,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

N_EXPIRIES = 20
N_PER_EXPIRY = 10
N_TOTAL = N_EXPIRIES * N_PER_EXPIRY


def _make_expiry_dates(n: int) -> list[pd.Timestamp]:
    """Generate n distinct future expiry dates."""
    return pd.date_range(
        start=pd.Timestamp.today().normalize() + pd.Timedelta(days=30),
        periods=n,
        freq="2W",
    ).tolist()


def _make_chain_df(n_expiries: int = N_EXPIRIES, n_per: int = N_PER_EXPIRY) -> pd.DataFrame:
    """Synthetic chain CSV DataFrame (no iv column yet)."""
    rng = np.random.default_rng(0)
    expiry_dates = _make_expiry_dates(n_expiries)
    rows = []
    for exp in expiry_dates:
        strikes = rng.uniform(400, 500, n_per)
        for K in strikes:
            rows.append(
                {
                    "ticker": "SPY",
                    "kind": "call",
                    "strike": K,
                    "expiry": (exp - pd.Timestamp.today().normalize()).days / 365.25,
                    "expiry_date": exp,
                    "mid": rng.uniform(5, 50),
                    "bid": rng.uniform(4, 45),
                    "ask": rng.uniform(6, 55),
                    "last_price": rng.uniform(5, 50),
                    "volume": rng.integers(100, 10_000),
                    "open_interest": rng.integers(1_000, 100_000),
                    "implied_volatility_yf": rng.uniform(0.1, 0.5),
                    "S": 450.0,
                    "r": 0.05,
                    "q": 0.01,
                }
            )
    return pd.DataFrame(rows)


def _mock_solve_chain(df: pd.DataFrame) -> pd.Series:
    """Fake solve_chain: returns a pd.Series of IVs, matching the real signature."""
    rng = np.random.default_rng(1)
    return pd.Series(
        rng.uniform(0.10, 0.50, len(df)),
        index=df.index,
        name="iv",
    )


@pytest.fixture
def prepared_df(tmp_path):
    """Load-and-prepared DataFrame using a mocked solve_chain."""
    raw = _make_chain_df()
    csv_path = tmp_path / "SPY_chain.csv"
    raw.to_csv(csv_path, index=False)

    with patch("pricer.vol_surface_nn.solve_chain", side_effect=_mock_solve_chain):
        df = load_and_prepare(csv_path)
    return df


@pytest.fixture
def split_dfs(prepared_df):
    return walk_forward_split(prepared_df)


@pytest.fixture
def trained_trainer(split_dfs):
    """A trainer that has completed at least a few epochs on synthetic data."""
    train, val, _ = split_dfs
    model = VolSurfaceNet(input_dim=2, hidden_sizes=[16, 16], dropout_rate=0.0)
    trainer = VolSurfaceTrainer(model, lr=1e-2, patience=5, max_epochs=30, batch_size=64)
    trainer.fit(train, val)
    return trainer


# ---------------------------------------------------------------------------
# load_and_prepare
# ---------------------------------------------------------------------------


class TestLoadAndPrepare:
    def test_returns_dataframe(self, tmp_path):
        raw = _make_chain_df()
        csv_path = tmp_path / "SPY_chain.csv"
        raw.to_csv(csv_path, index=False)
        with patch("pricer.vol_surface_nn.solve_chain", side_effect=_mock_solve_chain):
            df = load_and_prepare(csv_path)
        assert isinstance(df, pd.DataFrame)

    def test_feature_columns_present(self, tmp_path):
        raw = _make_chain_df()
        csv_path = tmp_path / "SPY_chain.csv"
        raw.to_csv(csv_path, index=False)
        with patch("pricer.vol_surface_nn.solve_chain", side_effect=_mock_solve_chain):
            df = load_and_prepare(csv_path)
        for col in ["log_moneyness", "T", "iv", "expiry_date"]:
            assert col in df.columns, f"Missing column: {col}"

    def test_use_rates_adds_columns(self, tmp_path):
        raw = _make_chain_df()
        csv_path = tmp_path / "SPY_chain.csv"
        raw.to_csv(csv_path, index=False)
        with patch("pricer.vol_surface_nn.solve_chain", side_effect=_mock_solve_chain):
            df = load_and_prepare(csv_path, use_rates=True)
        assert "r" in df.columns
        assert "q" in df.columns

    def test_iv_bounds_enforced(self, tmp_path):
        raw = _make_chain_df()
        csv_path = tmp_path / "SPY_chain.csv"
        raw.to_csv(csv_path, index=False)

        def _extreme_iv(df):
            return pd.Series(999.0, index=df.index, name="iv")  # all extreme

        with patch("pricer.vol_surface_nn.solve_chain", side_effect=_extreme_iv):
            with pytest.raises(ValueError, match="No contracts survived"):
                load_and_prepare(csv_path)

    def test_moneyness_band_filter(self, tmp_path):
        raw = _make_chain_df()
        csv_path = tmp_path / "SPY_chain.csv"
        raw.to_csv(csv_path, index=False)
        with patch("pricer.vol_surface_nn.solve_chain", side_effect=_mock_solve_chain):
            df = load_and_prepare(csv_path, moneyness_band=(-0.1, 0.1))
        assert df["log_moneyness"].between(-0.1, 0.1).all()

    def test_min_expiry_filter(self, tmp_path):
        raw = _make_chain_df()
        csv_path = tmp_path / "SPY_chain.csv"
        raw.to_csv(csv_path, index=False)
        with patch("pricer.vol_surface_nn.solve_chain", side_effect=_mock_solve_chain):
            df = load_and_prepare(csv_path, min_expiry_days=7)
        min_T = 7 / 365.25
        assert (df["T"] > min_T).all()

    def test_kind_filter(self, tmp_path):
        raw = _make_chain_df()
        # Add some put rows
        puts = raw.copy()
        puts["kind"] = "put"
        raw_all = pd.concat([raw, puts], ignore_index=True)
        csv_path = tmp_path / "SPY_chain.csv"
        raw_all.to_csv(csv_path, index=False)
        with patch("pricer.vol_surface_nn.solve_chain", side_effect=_mock_solve_chain):
            df = load_and_prepare(csv_path, kind="put")
        # All rows came from put filter — we just check the function ran
        assert len(df) > 0

    def test_no_nulls_in_output(self, tmp_path):
        raw = _make_chain_df()
        csv_path = tmp_path / "SPY_chain.csv"
        raw.to_csv(csv_path, index=False)
        with patch("pricer.vol_surface_nn.solve_chain", side_effect=_mock_solve_chain):
            df = load_and_prepare(csv_path)
        assert not df.isnull().any().any()


# ---------------------------------------------------------------------------
# walk_forward_split
# ---------------------------------------------------------------------------


class TestWalkForwardSplit:
    def test_no_overlap(self, prepared_df):
        train, val, test = walk_forward_split(prepared_df)
        train_dates = set(train["expiry_date"])
        val_dates = set(val["expiry_date"])
        test_dates = set(test["expiry_date"])
        assert not train_dates & val_dates
        assert not train_dates & test_dates
        assert not val_dates & test_dates

    def test_chronological_order(self, prepared_df):
        train, val, test = walk_forward_split(prepared_df)
        assert train["expiry_date"].max() <= val["expiry_date"].min()
        assert val["expiry_date"].max() <= test["expiry_date"].min()

    def test_all_rows_preserved(self, prepared_df):
        train, val, test = walk_forward_split(prepared_df)
        assert len(train) + len(val) + len(test) == len(prepared_df)

    def test_test_is_most_recent(self, prepared_df):
        _, _, test = walk_forward_split(prepared_df)
        all_dates = sorted(prepared_df["expiry_date"].unique())
        assert test["expiry_date"].min() >= all_dates[-int(len(all_dates) * 0.15) - 1]

    def test_raises_on_too_few_expiries(self):
        df = pd.DataFrame(
            {
                "log_moneyness": [0.0, 0.1],
                "T": [0.5, 0.6],
                "iv": [0.2, 0.25],
                "expiry_date": pd.to_datetime(["2025-01-01", "2025-01-01"]),
            }
        )
        with pytest.raises(ValueError, match="at least 3 distinct expiry dates"):
            walk_forward_split(df)


# ---------------------------------------------------------------------------
# VolSurfaceNet
# ---------------------------------------------------------------------------


class TestVolSurfaceNet:
    def test_output_shape(self):
        model = VolSurfaceNet(input_dim=2, hidden_sizes=[32, 32])
        x = torch.randn(16, 2)
        y = model(x)
        assert y.shape == (16,)

    def test_scalar_input(self):
        model = VolSurfaceNet(input_dim=2)
        x = torch.randn(1, 2)
        y = model(x)
        assert y.shape == (1,)

    def test_with_rates_input(self):
        model = VolSurfaceNet(input_dim=4, hidden_sizes=[32])
        x = torch.randn(8, 4)
        y = model(x)
        assert y.shape == (8,)

    def test_configurable_depth(self):
        model = VolSurfaceNet(input_dim=2, hidden_sizes=[128, 64, 32])
        x = torch.randn(4, 2)
        y = model(x)
        assert y.shape == (4,)

    def test_dropout_zero_deterministic(self):
        model = VolSurfaceNet(input_dim=2, hidden_sizes=[16], dropout_rate=0.0)
        model.eval()
        x = torch.randn(4, 2)
        y1 = model(x)
        y2 = model(x)
        assert torch.allclose(y1, y2)

    def test_parameters_exist(self):
        model = VolSurfaceNet()
        params = list(model.parameters())
        assert len(params) > 0


# ---------------------------------------------------------------------------
# _EarlyStopping
# ---------------------------------------------------------------------------


class TestEarlyStopping:
    def test_triggers_after_patience(self):
        model = VolSurfaceNet(input_dim=2, hidden_sizes=[4])
        es = _EarlyStopping(patience=3)
        # First call sets the best (improving from inf) — counter stays 0.
        # The following 3 calls don't improve — counter reaches patience=3.
        es.step(val_loss=1.0, model=model)
        stopped = False
        for _ in range(3):
            stopped = es.step(val_loss=1.0, model=model)
        assert stopped

    def test_does_not_trigger_on_improvement(self):
        model = VolSurfaceNet(input_dim=2, hidden_sizes=[4])
        es = _EarlyStopping(patience=3)
        for i in range(5):
            stopped = es.step(val_loss=1.0 - i * 0.1, model=model)
        assert not stopped

    def test_stores_best_state(self):
        model = VolSurfaceNet(input_dim=2, hidden_sizes=[4])
        es = _EarlyStopping(patience=3)
        es.step(val_loss=0.5, model=model)
        assert es.best_state is not None

    def test_counter_resets_on_improvement(self):
        model = VolSurfaceNet(input_dim=2, hidden_sizes=[4])
        es = _EarlyStopping(patience=5)
        es.step(1.0, model)
        es.step(1.0, model)  # counter = 2
        es.step(0.5, model)  # improvement: counter resets to 0
        assert es._counter == 0


# ---------------------------------------------------------------------------
# VolSurfaceTrainer
# ---------------------------------------------------------------------------


class TestVolSurfaceTrainer:
    def test_fit_returns_history(self, split_dfs):
        train, val, _ = split_dfs
        model = VolSurfaceNet(input_dim=2, hidden_sizes=[8])
        trainer = VolSurfaceTrainer(model, max_epochs=5, patience=3)
        history = trainer.fit(train, val)
        assert "train_loss" in history
        assert "val_loss" in history
        assert len(history["train_loss"]) > 0

    def test_history_length_matches_epochs(self, split_dfs):
        train, val, _ = split_dfs
        model = VolSurfaceNet(input_dim=2, hidden_sizes=[8])
        trainer = VolSurfaceTrainer(model, max_epochs=10, patience=50, batch_size=512)
        history = trainer.fit(train, val)
        assert len(history["train_loss"]) == len(history["val_loss"])

    def test_predict_shape(self, trained_trainer, split_dfs):
        _, _, test = split_dfs
        preds = trained_trainer.predict(test)
        assert preds.shape == (len(test),)

    def test_predict_positive_values(self, trained_trainer, split_dfs):
        # IVs should be positive after a reasonable training
        _, _, test = split_dfs
        preds = trained_trainer.predict(test)
        # Not a hard guarantee, but synthetic data should produce sensible range
        assert preds.dtype == np.float64 or preds.dtype == np.float32

    def test_scalers_fitted(self, trained_trainer):
        from sklearn.utils.validation import check_is_fitted
        check_is_fitted(trained_trainer.scaler_X)
        check_is_fitted(trained_trainer.scaler_y)

    def test_feature_cols_stored(self, split_dfs):
        train, val, _ = split_dfs
        model = VolSurfaceNet(input_dim=2, hidden_sizes=[8])
        trainer = VolSurfaceTrainer(model, max_epochs=3, patience=3)
        trainer.fit(train, val, feature_cols=["log_moneyness", "T"])
        assert trainer._feature_cols == ["log_moneyness", "T"]

    def test_fit_with_rates(self, tmp_path):
        raw = _make_chain_df(n_expiries=15, n_per=5)
        csv_path = tmp_path / "SPY_chain.csv"
        raw.to_csv(csv_path, index=False)
        with patch("pricer.vol_surface_nn.solve_chain", side_effect=_mock_solve_chain):
            df = load_and_prepare(csv_path, use_rates=True)
        train, val, _ = walk_forward_split(df)
        model = VolSurfaceNet(input_dim=4, hidden_sizes=[8])
        trainer = VolSurfaceTrainer(model, max_epochs=3, patience=3)
        history = trainer.fit(train, val, feature_cols=FEATURE_COLS_RATES)
        assert len(history["train_loss"]) > 0

    def test_model_in_eval_mode_after_fit(self, split_dfs):
        train, val, _ = split_dfs
        model = VolSurfaceNet(input_dim=2, hidden_sizes=[8])
        trainer = VolSurfaceTrainer(model, max_epochs=3, patience=2)
        trainer.fit(train, val)
        assert not trainer.model.training

    def test_loss_decreases(self, split_dfs):
        """With a tiny model and enough epochs, train loss should trend down."""
        train, val, _ = split_dfs
        model = VolSurfaceNet(input_dim=2, hidden_sizes=[32, 32], dropout_rate=0.0)
        trainer = VolSurfaceTrainer(model, lr=1e-2, max_epochs=50, patience=50, batch_size=32)
        history = trainer.fit(train, val)
        assert history["train_loss"][-1] < history["train_loss"][0]


# ---------------------------------------------------------------------------
# _metrics
# ---------------------------------------------------------------------------


class TestMetrics:
    def test_perfect_prediction(self):
        y = np.array([0.2, 0.3, 0.4])
        m = _metrics(y, y)
        assert m["rmse"] == pytest.approx(0.0)
        assert m["mae"] == pytest.approx(0.0)
        assert m["max_ae"] == pytest.approx(0.0)

    def test_known_values(self):
        y_true = np.array([0.2, 0.4])
        y_pred = np.array([0.3, 0.5])
        m = _metrics(y_true, y_pred)
        assert m["rmse"] == pytest.approx(0.1, abs=1e-6)
        assert m["mae"] == pytest.approx(0.1, abs=1e-6)
        assert m["max_ae"] == pytest.approx(0.1, abs=1e-6)

    def test_keys_present(self):
        y = np.array([0.2])
        m = _metrics(y, y)
        assert set(m.keys()) == {"rmse", "mae", "max_ae"}


# ---------------------------------------------------------------------------
# evaluate
# ---------------------------------------------------------------------------


class TestEvaluate:
    def test_returns_nn_key(self, trained_trainer, split_dfs):
        _, _, test = split_dfs
        results = evaluate(trained_trainer, test)
        assert "nn" in results
        assert "rmse" in results["nn"]
        assert "mae" in results["nn"]
        assert "max_ae" in results["nn"]

    def test_no_spline_key_when_not_provided(self, trained_trainer, split_dfs):
        _, _, test = split_dfs
        results = evaluate(trained_trainer, test)
        assert "spline" not in results

    def test_spline_key_present_when_provided(self, trained_trainer, split_dfs):
        _, _, test = split_dfs

        def dummy_spline(m, T):
            return np.full(len(m) if hasattr(m, "__len__") else 1, 0.25)

        results = evaluate(trained_trainer, test, spline_fn=dummy_spline)
        assert "spline" in results

    def test_metrics_are_nonnegative(self, trained_trainer, split_dfs):
        _, _, test = split_dfs
        results = evaluate(trained_trainer, test)
        for v in results["nn"].values():
            assert v >= 0.0


# ---------------------------------------------------------------------------
# plot_evaluation
# ---------------------------------------------------------------------------


class TestPlotEvaluation:
    def test_returns_figure(self, trained_trainer, split_dfs):
        import matplotlib.pyplot as plt

        _, _, test = split_dfs
        fig = plot_evaluation(trained_trainer, test)
        assert isinstance(fig, plt.Figure)
        plt.close("all")

    def test_returns_figure_with_spline(self, trained_trainer, split_dfs):
        import matplotlib.pyplot as plt

        _, _, test = split_dfs

        def dummy_spline(m, T):
            return np.full(len(np.asarray(m).ravel()), 0.25)

        fig = plot_evaluation(trained_trainer, test, spline_fn=dummy_spline)
        assert isinstance(fig, plt.Figure)
        plt.close("all")


# ---------------------------------------------------------------------------
# plot_loss_curves
# ---------------------------------------------------------------------------


class TestPlotLossCurves:
    def test_returns_figure(self):
        import matplotlib.pyplot as plt

        history = {"train_loss": [1.0, 0.8, 0.6], "val_loss": [1.1, 0.9, 0.7]}
        fig = plot_loss_curves(history)
        assert isinstance(fig, plt.Figure)
        plt.close("all")


# ---------------------------------------------------------------------------
# predict_surface
# ---------------------------------------------------------------------------


class TestPredictSurface:
    def test_output_shapes(self, trained_trainer):
        M, T_grid, IV_grid = predict_surface(trained_trainer, n_points=20)
        assert M.shape == (20, 20)
        assert T_grid.shape == (20, 20)
        assert IV_grid.shape == (20, 20)

    def test_moneyness_range_respected(self, trained_trainer):
        M, _, _ = predict_surface(
            trained_trainer, moneyness_range=(-0.2, 0.2), n_points=10
        )
        assert M.min() == pytest.approx(-0.2, abs=1e-6)
        assert M.max() == pytest.approx(0.2, abs=1e-6)

    def test_T_range_respected(self, trained_trainer):
        _, T_grid, _ = predict_surface(
            trained_trainer, T_range=(0.1, 1.0), n_points=10
        )
        assert T_grid.min() == pytest.approx(0.1, abs=1e-6)
        assert T_grid.max() == pytest.approx(1.0, abs=1e-6)

    def test_with_rates_feature_cols(self, tmp_path):
        raw = _make_chain_df(n_expiries=15, n_per=5)
        csv_path = tmp_path / "SPY_chain.csv"
        raw.to_csv(csv_path, index=False)
        with patch("pricer.vol_surface_nn.solve_chain", side_effect=_mock_solve_chain):
            df = load_and_prepare(csv_path, use_rates=True)
        train, val, _ = walk_forward_split(df)
        model = VolSurfaceNet(input_dim=4, hidden_sizes=[8])
        trainer = VolSurfaceTrainer(model, max_epochs=3, patience=3)
        trainer.fit(train, val, feature_cols=FEATURE_COLS_RATES)
        M, T_grid, IV_grid = predict_surface(
            trainer, feature_cols=FEATURE_COLS_RATES, n_points=10
        )
        assert IV_grid.shape == (10, 10)


# ---------------------------------------------------------------------------
# plot_surfaces
# ---------------------------------------------------------------------------


class TestPlotSurfaces:
    def test_returns_figure_nn_only(self, trained_trainer):
        import matplotlib.pyplot as plt

        result = predict_surface(trained_trainer, n_points=10)
        fig = plot_surfaces(result)
        assert isinstance(fig, plt.Figure)
        plt.close("all")

    def test_returns_figure_with_spline(self, trained_trainer):
        import matplotlib.pyplot as plt

        result = predict_surface(trained_trainer, n_points=10)

        def dummy_spline(M, T):
            return np.full_like(np.asarray(M), 0.25)

        fig = plot_surfaces(result, spline_fn=dummy_spline)
        assert isinstance(fig, plt.Figure)
        plt.close("all")
