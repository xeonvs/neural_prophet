"""Seasonality must not enable price imputation or mix independent series."""

import numpy as np
import pandas as pd
import pytest

from neuralprophet import NeuralProphet, configure, configure_components, df_utils, set_random_seed
from neuralprophet.data.process import _handle_missing_data


def _seasonality(kind):
    if kind == "none":
        return None
    config = configure_components.Seasonalities(seasonality_local_reg=0)
    if kind == "conditional":
        config.append("active_week", period=7, resolution=3, arg=True, condition_name="active")
    return config


@pytest.mark.parametrize("seasonality", ["none", "ordinary", "conditional"])
@pytest.mark.parametrize("impute_missing", [False, True])
@pytest.mark.parametrize("predicting", [False, True])
def test_trailing_targets_with_seasonality(seasonality, impute_missing, predicting, monkeypatch):
    """Restoring forecast rows must work even when interpolation is disabled."""
    frame = pd.DataFrame(
        {
            "ds": pd.date_range("2024-01-01", periods=7),
            "y": [1.0, 2.0, np.nan, 4.0, 5.0, np.nan, np.nan],
            "ID": "a",
            "active": [1.0, 0.0, np.nan, 1.0, 0.0, np.nan, 1.0],
        }
    )
    original = frame.copy(deep=True)
    if not impute_missing:

        def forbidden_imputer(*args, **kwargs):
            pytest.fail("impute_missing=False called the interpolator")

        monkeypatch.setattr(df_utils, "fill_linear_then_rolling_avg", forbidden_imputer)
    output = _handle_missing_data(
        frame.copy(deep=True),
        freq="D",
        n_lags=2,
        n_forecasts=2,
        config_missing=configure.MissingDataHandling(impute_missing=impute_missing),
        config_seasonality=_seasonality(seasonality),
        predicting=predicting,
    )
    assert len(output) == (7 if predicting else 5)
    if predicting:
        assert output.loc[[5, 6], "y"].isna().all()
        if seasonality == "conditional":
            assert output.loc[[5, 6], "active"].tolist() == [0.0, 1.0]
    if not impute_missing:
        assert pd.isna(output.loc[2, "y"])
        assert pd.isna(output.loc[2, "active"])
        pd.testing.assert_series_equal(output.loc[:4, "y"], original.loc[:4, "y"])
    else:
        assert np.isfinite(output.loc[2, "y"])
    # Explicit conditions (including zero) must not be replaced.
    pd.testing.assert_series_equal(output.loc[[0, 1, 3, 4], "active"], original.loc[[0, 1, 3, 4], "active"])


@pytest.mark.parametrize("impute_missing", [False, True])
def test_conditional_restore_does_not_cross_series(impute_missing):
    """Restored rows must inherit their own ID, not the last ID in training."""
    frame = pd.DataFrame(
        {
            "ds": list(pd.date_range("2024-01-01", periods=4)) * 2,
            "y": [1.0, 2.0, 3.0, np.nan, 10.0, 11.0, 12.0, np.nan],
            "ID": ["a"] * 4 + ["b"] * 4,
            "active": [1.0, 1.0, 1.0, np.nan, 0.0, 0.0, 0.0, np.nan],
        }
    )
    output = _handle_missing_data(
        frame,
        freq="D",
        n_lags=2,
        n_forecasts=1,
        config_missing=configure.MissingDataHandling(impute_missing=impute_missing),
        config_seasonality=_seasonality("conditional"),
        predicting=True,
    )
    assert output.loc[3, "active"] == 1.0
    assert output.loc[7, "active"] == 0.0
    assert output.loc[[3, 7], "y"].isna().all()


def test_future_only_condition_is_not_backfilled():
    frame = pd.DataFrame(
        {
            "ds": pd.date_range("2024-01-01", periods=4),
            "y": [1.0, 2.0, np.nan, np.nan],
            "ID": "a",
            "active": [np.nan, np.nan, np.nan, 1.0],
        }
    )
    output = _handle_missing_data(
        frame,
        freq="D",
        n_lags=1,
        n_forecasts=2,
        config_missing=configure.MissingDataHandling(impute_missing=False),
        config_seasonality=_seasonality("conditional"),
        predicting=True,
    )
    assert output.loc[:2, "active"].isna().all()
    assert output.loc[3, "active"] == 1.0
    assert output.loc[2:, "y"].isna().all()


@pytest.mark.parametrize("seasonality", ["ordinary", "conditional"])
def test_non_autoregressive_future_with_seasonality(seasonality):
    frame = pd.DataFrame(
        {
            "ds": pd.date_range("2024-01-01", periods=4),
            "y": [1.0, 2.0, np.nan, np.nan],
            "ID": "a",
            "active": [1.0, 0.0, np.nan, 1.0],
        }
    )
    output = _handle_missing_data(
        frame,
        freq="D",
        n_lags=0,
        n_forecasts=1,
        config_missing=configure.MissingDataHandling(impute_missing=False),
        config_seasonality=_seasonality(seasonality),
        predicting=True,
    )
    assert len(output) == 4
    assert output.loc[2:, "y"].isna().all()
    if seasonality == "conditional":
        assert output.loc[2:, "active"].tolist() == [0.0, 1.0]


@pytest.mark.parametrize("impute_missing", [False, True])
def test_public_forecast_with_weekly_and_conditional_seasonality(impute_missing, tmp_path, monkeypatch):
    """Exercise the real public fit/future/predict path on CPU, without replacing prices."""
    monkeypatch.chdir(tmp_path)
    set_random_seed(42)
    rows = 71
    dates = pd.date_range("2024-01-01", periods=rows)
    frame = pd.DataFrame(
        {
            "ds": dates,
            "y": 20.0 + np.arange(rows) * 0.02 + np.sin(np.arange(rows) * 2 * np.pi / 7),
            "active": (dates.dayofweek < 3).astype(float),
        }
    )
    model = NeuralProphet(
        n_lags=7,
        n_forecasts=3,
        weekly_seasonality=3,
        yearly_seasonality=False,
        daily_seasonality=False,
        impute_missing=impute_missing,
        epochs=1,
        batch_size=32,
        learning_rate=0.01,
        accelerator="cpu",
        trainer_config={"logger": False, "enable_model_summary": False},
    )
    model.add_seasonality("active_week", period=7, fourier_order=2, condition_name="active")
    model.fit(frame, freq="D", minimal=True, progress=None)
    future = model.make_future_dataframe(frame, periods=3, n_historic_predictions=False)
    # The caller knows future calendar conditions and provides both 0 and 1.
    future["active"] = (future["ds"].dt.dayofweek < 3).astype(float)
    assert set(future.tail(3)["active"]) == {0.0, 1.0}
    future_before_predict = future.copy(deep=True)
    assert future.tail(3)["y"].isna().all()
    forecast = model.predict(future)
    future_dates = dates[-1] + pd.to_timedelta(np.arange(1, 4), unit="D")
    rows_by_date = forecast.set_index("ds")
    predictions = [rows_by_date.loc[date, f"yhat{step}"] for step, date in enumerate(future_dates, start=1)]
    assert np.isfinite(predictions).all()
    assert {"season_weekly", "season_active_week"}.issubset(forecast.columns)
    assert forecast.loc[forecast["ds"].isin(future_dates), "y"].isna().all()
    pd.testing.assert_frame_equal(future, future_before_predict)
