"""Compatibility coverage for the maintained NumPy 2 branch."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import neuralprophet
from neuralprophet import NeuralProphet, set_random_seed


def test_removed_numpy_nan_alias_is_absent():
    package_root = Path(neuralprophet.__file__).parent
    offenders = [path for path in package_root.rglob("*.py") if "np.NaN" in path.read_text(encoding="utf-8")]
    assert offenders == []


def test_multihorizon_events_lagged_regressor_and_matplotlib_output(tmp_path):
    """Fit the d-trader-bot package surface and preserve all matplotlib figures."""
    set_random_seed(42)
    rows = 96
    signal = np.sin(np.linspace(0, 8, rows))
    frame = pd.DataFrame(
        {
            "ds": pd.date_range("2024-01-01", periods=rows, freq="D"),
            "y": 20.0 + np.linspace(0, 3, rows) + signal,
            "momentum": signal,
        }
    )
    events = pd.DataFrame({"event": ["report", "report"], "ds": frame.loc[[30, 70], "ds"].to_numpy()})
    model = NeuralProphet(
        n_lags=6,
        n_forecasts=3,
        epochs=1,
        batch_size=32,
        learning_rate=1.0,
        yearly_seasonality=False,
        weekly_seasonality=False,
        daily_seasonality=False,
    )
    model = model.add_events("report")
    model = model.add_lagged_regressor("momentum", n_lags=3)
    history = model.create_df_with_events(frame, events)

    model.fit(history, freq="D", progress=None)
    future = model.make_future_dataframe(
        history,
        events_df=events,
        periods=3,
        n_historic_predictions=12,
    )
    forecast = model.predict(future)

    assert {"yhat1", "yhat2", "yhat3"}.issubset(forecast.columns)
    model.set_plotting_backend("matplotlib")
    figures = (model.plot(forecast), model.plot_components(forecast), model.plot_parameters())
    for index, figure in enumerate(figures):
        assert figure is not None
        output = tmp_path / f"neuralprophet-{index}.png"
        figure.savefig(output)
        assert output.stat().st_size > 0
        plt.close(figure)
