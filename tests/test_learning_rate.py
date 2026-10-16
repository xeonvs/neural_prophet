"""Train-only automatic LR selection rejects flat, invalid and exploded curves."""

from contextlib import nullcontext
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch
from torch.utils.data import DataLoader, Dataset, SequentialSampler, TensorDataset

from neuralprophet import NeuralProphet, set_random_seed
from neuralprophet import utils_lightning as module


def finder(loss, rates=None):
    rates = list(np.geomspace(1e-6, 10.0, len(loss))) if rates is None else rates

    def unsafe_default(**kwargs):
        raise AssertionError("Do not consult a default suggestion from an exploded tail")

    return SimpleNamespace(results={"lr": rates, "loss": loss}, suggestion=unsafe_default)


def test_decreasing_region_preserves_selector_tuple_and_padding_compatibility():
    curve = finder([8.0, 7.0, 5.0, 2.0, 1.0, 1.2, 1.5, 1000.0])
    loss, rates, selected = module.smooth_loss_and_suggest(curve, window=2)
    assert len(loss) == len(rates) + 2
    np.testing.assert_array_equal(loss[1:-1], curve.results["loss"])
    assert rates is curve.results["lr"]
    assert selected in rates[1:4]
    assert selected < rates[4]


def test_captured_comparable_training_curve_replays_without_late_explosion_default():
    recorded = json.loads((Path(__file__).parent / "test-data" / "lr_finder_comparable_training.json").read_text())
    curve = finder(recorded["loss"], recorded["lr"])
    assert len(curve.results["lr"]) == len(curve.results["loss"]) == 104
    assert module.smooth_loss_and_suggest(curve)[2] == recorded["expected_learning_rate"]


def test_captured_curve_with_observed_count_correction_keeps_supported_descent():
    recorded = json.loads((Path(__file__).parent / "test-data" / "lr_finder_comparable_training.json").read_text())
    count = np.arange(len(recorded["loss"]))
    corrected = np.array(recorded["loss"]) * (1 - 0.98 ** (count + 2)) / (1 - 0.98 ** (count + 1))
    assert corrected[0] == pytest.approx(2.8432071208953857)
    assert (
        module.smooth_loss_and_suggest(finder(corrected.tolist(), recorded["lr"]))[2]
        == recorded["expected_learning_rate"]
    )


@pytest.mark.parametrize("global_step", [0, 1, 500])
def test_probe_count_uses_observations_not_automatic_or_manual_optimizer_steps(global_step):
    model = SimpleNamespace()
    old_callbacks = []
    callback = module._LRCallback(num_training=100, early_stop_threshold=4.0)
    other = module._LRCallback(num_training=100, early_stop_threshold=4.0)
    untouched = other.on_train_batch_end.__func__
    trainer = SimpleNamespace(
        callbacks=old_callbacks,
        fit_loop=SimpleNamespace(batch_idx=0),
        accumulate_grad_batches=1,
        global_step=global_step,
        should_stop=False,
        strategy=SimpleNamespace(broadcast=lambda x: x),
    )
    with module._lr_observation_count(model, trainer):
        trainer.callbacks = [callback]
        model.on_train_start()
        callback.on_train_batch_end(trainer, model, torch.tensor(10.0), None, 0)
        assert callback.losses == pytest.approx([10.0])
        assert callback.best_loss == pytest.approx(10.0)
        callback.on_train_batch_end(trainer, model, torch.tensor(100.0), None, 1)
        assert trainer.should_stop
        assert callback.losses[1] == pytest.approx((0.98 * 0.02 * 10 + 0.02 * 100) / (1 - 0.98**2))
        assert other.on_train_batch_end.__func__ is untouched
        assert "on_train_batch_end" not in vars(other) and not other.losses
    assert trainer.callbacks is old_callbacks
    assert "on_train_start" not in vars(model)
    assert "on_train_batch_end" not in vars(callback) and "_np_observed_updates" not in vars(callback)
    assert callback.avg_loss == 0.0 and callback.best_loss == 0.0


def test_missing_probe_loss_preserves_alignment_without_advancing_bias_counter():
    callback = module._LRCallback(100, 4.0)
    model = SimpleNamespace()
    trainer = SimpleNamespace(
        callbacks=[],
        fit_loop=SimpleNamespace(batch_idx=0),
        accumulate_grad_batches=1,
        should_stop=False,
        strategy=SimpleNamespace(broadcast=lambda x: x),
    )
    with module._lr_observation_count(model, trainer):
        trainer.callbacks = [callback]
        model.on_train_start()
        callback.on_train_batch_end(trainer, model, {}, None, 0)
        callback.on_train_batch_end(trainer, model, torch.tensor(10.0), None, 1)
        assert np.isnan(callback.losses[0]) and callback.losses[1] == pytest.approx(10.0)
        assert callback._np_observed_updates == 1
        callback.on_train_batch_end(trainer, model, torch.tensor(float("inf")), None, 2)
        assert np.isnan(callback.losses[-1]) and trainer.should_stop
        assert callback._np_observed_updates == 1


@pytest.mark.parametrize("failure", [False, True])
def test_probe_instance_hook_and_original_callback_list_are_restored(failure):
    calls = []
    hook = lambda: calls.append("original")
    model = SimpleNamespace(on_train_start=hook)
    old_callbacks = [object()]
    trainer = SimpleNamespace(callbacks=old_callbacks)
    callback = module._LRCallback(100, 4.0)
    with pytest.raises(RuntimeError, match="owned-test") if failure else nullcontext():
        with module._lr_observation_count(model, trainer):
            trainer.callbacks = [callback]
            model.on_train_start()
            if failure:
                raise RuntimeError("owned-test")
    assert calls == ["original"]
    assert model.on_train_start is hook and trainer.callbacks is old_callbacks
    assert "on_train_batch_end" not in vars(callback)


@pytest.mark.parametrize(
    "callbacks",
    [
        [],
        [module._LRCallback(10, 4.0), module._LRCallback(10, 4.0)],
        [module._LRCallback(10, None)],
        [module._LRCallback(10, 4.0, beta=0.0)],
    ],
)
def test_incompatible_private_probe_callback_fails_closed_and_restores_hook(callbacks):
    model = SimpleNamespace()
    trainer = SimpleNamespace(callbacks=[])
    original = trainer.callbacks
    with pytest.raises(ValueError, match="^auto_lr_unreliable$"):
        with module._lr_observation_count(model, trainer):
            trainer.callbacks = callbacks
            model.on_train_start()
    assert "on_train_start" not in vars(model) and trainer.callbacks is original


@pytest.mark.parametrize(
    "loss",
    [
        [],
        [1.0],
        [2.0, 1.0],
        [1.0] * 30,
        [1.0, 2.0, 3.0, 4.0, 10.0],
        [1.0, 2.0, 1.5, 1.2, 1.1, 4.5, 1.01],  # the first probe remains best
        [2.0, 1.0] * 20,
        [1.0, 2.0] * 20,  # repeated batches without a supported decrease
        [np.nan, 3.0, 2.0, 1.0],
        [np.inf, 3.0, 2.0, 1.0],
        [1.0, np.nan, 0.5],
    ],
)
def test_no_reliable_decreasing_region_fails_explicitly(loss):
    with pytest.raises(ValueError, match="^auto_lr_unreliable$"):
        module.smooth_loss_and_suggest(finder(loss))


@pytest.mark.parametrize("tail", [[np.nan, 0.0], [np.inf, 0.0], [10.0, 0.01, 0.001], [10000.0, 1e-6]])
def test_nonfinite_and_diverged_tail_cannot_move_valid_selection(tail):
    rates = list(np.geomspace(1e-6, 10.0, 12))
    prefix = [8.0, 7.0, 5.0, 2.0, 1.0]
    expected = module.smooth_loss_and_suggest(finder(prefix, rates[:5]))[2]
    extended = module.smooth_loss_and_suggest(finder(prefix + tail, rates[: len(prefix + tail)]))[2]
    assert extended == expected


def test_first_supported_trough_is_not_replaced_by_later_steeper_recovery():
    rates = list(np.geomspace(1e-6, 10.0, 10))
    prefix = [8.0, 7.0, 5.0, 2.0, 1.0]
    expected = module.smooth_loss_and_suggest(finder(prefix, rates[:5]))[2]
    extended = module.smooth_loss_and_suggest(finder(prefix + [1.1, 1.2, 1.0, 0.5, 0.1], rates))[2]
    assert extended == expected


@pytest.mark.parametrize(
    "rates,loss",
    [
        ([1e-6, 1e-6, 1e-5], [3.0, 2.0, 1.0]),
        ([1e-6, 1e-5, 1e-6], [3.0, 2.0, 1.0]),
        ([1e-6, 0.0, 1e-5], [3.0, 2.0, 1.0]),
        ([1e-6, 1e-5], [3.0, 2.0, 1.0]),
    ],
)
def test_invalid_rate_curve_is_not_a_manual_fallback(rates, loss):
    with pytest.raises(ValueError, match="^auto_lr_unreliable$"):
        module.smooth_loss_and_suggest(finder(loss, rates))


def test_range_test_stops_standard_divergence_and_uses_training_loader_only(monkeypatch):
    model = SimpleNamespace(finding_lr=False)
    loader = DataLoader(TensorDataset(torch.arange(6, dtype=torch.float32)[:, None]), batch_size=2, shuffle=True)
    trainer = SimpleNamespace(fit_loop=SimpleNamespace(max_epochs=3))
    captured = {}

    class Tuner:
        def __init__(self, trainer):
            captured["trainer"] = trainer

        def lr_find(self, **kwargs):
            assert model.finding_lr
            captured.update(kwargs)
            return finder([8.0, 7.0, 5.0, 2.0, 1.0, 1.1, 1000.0])

    monkeypatch.setattr(module, "Tuner", Tuner)
    result = module.find_learning_rate(model, loader, trainer, train_epochs=100)
    assert result > 0 and model.finding_lr is False
    assert captured["early_stop_threshold"] == 4.0
    assert captured["update_attr"] is False
    assert captured["train_dataloaders"].dataset is loader.dataset
    assert captured["train_dataloaders"].batch_size == len(loader.dataset)
    assert captured["train_dataloaders"].collate_fn is loader.collate_fn
    assert loader.batch_size == 2
    assert "val_dataloaders" not in captured
    assert captured["num_training"] > len(loader)
    assert trainer.fit_loop.max_epochs == 3
    assert model.learning_rate == result


@pytest.mark.parametrize("failure", ["probe", "selection", "absent"])
def test_finding_lr_flag_is_restored_on_every_failure(monkeypatch, failure):
    model = SimpleNamespace(finding_lr=False, learning_rate=0.007)
    trainer = SimpleNamespace(fit_loop=SimpleNamespace(max_epochs=3))

    class Tuner:
        def __init__(self, trainer):
            pass

        def lr_find(self, **kwargs):
            model.learning_rate = 0.1  # TimeNet initializes a temporary probe optimizer.
            if failure == "probe":
                raise RuntimeError("probe failed")
            return finder([1.0] * 20) if failure == "selection" else None

    monkeypatch.setattr(module, "Tuner", Tuner)
    with pytest.raises((ValueError, RuntimeError)):
        module.find_learning_rate(
            model, DataLoader(TensorDataset(torch.arange(3.0)), batch_size=1), trainer, train_epochs=3
        )
    assert model.finding_lr is False
    assert model.learning_rate == 0.007
    assert trainer.fit_loop.max_epochs == 3


@pytest.mark.parametrize("maximum", [0, 3, 500, None, -1])
@pytest.mark.parametrize("failure", [False, True])
@pytest.mark.parametrize("original_flag", [False, True])
def test_probe_epoch_budget_is_independent_and_restored_for_finite_or_unlimited(
    monkeypatch, maximum, failure, original_flag
):
    trainer = SimpleNamespace(fit_loop=SimpleNamespace(max_epochs=maximum))
    model = SimpleNamespace(finding_lr=original_flag, learning_rate=None)
    loader = DataLoader(TensorDataset(torch.arange(6.0)), batch_size=2)
    original_batch_size = loader.batch_size

    class Tuner:
        def __init__(self, actual):
            assert actual is trainer

        def lr_find(self, **kwargs):
            expected = (
                kwargs["num_training"] if maximum is not None and 0 <= maximum < kwargs["num_training"] else maximum
            )
            assert trainer.fit_loop.max_epochs == expected
            if failure:
                raise RuntimeError("probe failed")
            return finder([8.0, 7.0, 5.0, 2.0, 1.0, 1.1, 1000.0])

    monkeypatch.setattr(module, "Tuner", Tuner)
    if failure:
        with pytest.raises(RuntimeError, match="^probe failed$"):
            module.find_learning_rate(model, loader, trainer, train_epochs=3)
    else:
        assert module.find_learning_rate(model, loader, trainer, train_epochs=3) > 0
    assert trainer.fit_loop.max_epochs == maximum and model.finding_lr is original_flag
    assert loader.batch_size == original_batch_size


def test_probe_keeps_exact_small_training_set_and_never_mutates_main_loader():
    dataset = TensorDataset(torch.arange(60, dtype=torch.float32).reshape(10, 6))
    main = DataLoader(dataset, batch_size=3, shuffle=True, drop_last=True)
    original_sampler = main.sampler
    probe = module._lr_probe_loader(main)
    assert probe.dataset is dataset and probe.collate_fn is main.collate_fn
    assert probe.batch_size == 10 and not probe.drop_last
    assert isinstance(probe.sampler, SequentialSampler)
    first = next(iter(probe))[0]
    second = next(iter(probe))[0]
    torch.testing.assert_close(first, dataset.tensors[0])
    torch.testing.assert_close(first, second)
    assert main.batch_size == 3 and main.drop_last and main.sampler is original_sampler


def test_large_probe_is_evenly_spread_bounded_and_target_independent():
    dataset = TensorDataset(torch.arange(4096, dtype=torch.float32)[:, None])
    main = DataLoader(dataset, batch_size=8, shuffle=True)
    probe = module._lr_probe_loader(main)
    expected = np.linspace(0, len(dataset) - 1, module.MAX_LR_PROBE_WINDOWS, dtype=int).tolist()
    assert probe.dataset.dataset is dataset
    assert probe.dataset.indices == expected
    assert len(probe.dataset) == module.MAX_LR_PROBE_WINDOWS
    assert expected[0] == 0 and expected[-1] == len(dataset) - 1
    assert main.dataset is dataset and main.batch_size == 8


def test_tensor_byte_limit_bounds_selected_batch_before_collation(monkeypatch):
    monkeypatch.setattr(module, "MAX_LR_PROBE_BYTES", 128)

    class Windows(Dataset):
        def __len__(self):
            return 10

        def __getitem__(self, index):
            return torch.zeros((4, 4), dtype=torch.float32), {"df_name": "fixture"}

    main = DataLoader(Windows(), batch_size=1)
    probe = module._lr_probe_loader(main)
    assert probe.dataset.indices == [0, 9]
    batch = next(iter(probe))[0]
    assert batch.numel() * batch.element_size() == 128
    assert module._lr_sample_bytes(main.dataset[0]) == 64


def test_single_oversized_or_empty_probe_fails_without_large_collation(monkeypatch):
    monkeypatch.setattr(module, "MAX_LR_PROBE_BYTES", 8)
    oversized = DataLoader(TensorDataset(torch.zeros((2, 4))), batch_size=1)
    empty = DataLoader(TensorDataset(torch.zeros((0, 4))), batch_size=1)
    for main in (oversized, empty):
        with pytest.raises(ValueError, match="^auto_lr_unreliable$"):
            module._lr_probe_loader(main)


def test_real_manual_optimizer_probe_first_loss_is_unbiased_and_restores_state(monkeypatch):
    """Run the actual fixed range test, then stop deliberately before normal fit."""
    from neuralprophet.time_net import TimeNet

    captured = []
    actual_losses = []
    original = module.Tuner.lr_find
    original_loss = TimeNet.loss_func

    def lr_find(self, **kwargs):
        result = original(self, **kwargs)
        captured.append(result)
        return result

    monkeypatch.setattr(module.Tuner, "lr_find", lr_find)

    def loss_func(self, *args, **kwargs):
        result = original_loss(self, *args, **kwargs)
        if self.finding_lr:
            actual_losses.append(float(result[0].item()))
        return result

    def stop_after_probe(curve, *args, **kwargs):
        assert curve.results["loss"][0] == pytest.approx(actual_losses[0])
        raise ValueError("owned_test_probe_complete")

    monkeypatch.setattr(TimeNet, "loss_func", loss_func)
    monkeypatch.setattr(module, "smooth_loss_and_suggest", stop_after_probe)
    set_random_seed(17)
    x = np.arange(180)
    history = pd.DataFrame(
        {"ds": pd.date_range("2025-01-01", periods=len(x)), "y": 100 + x * 0.05 + 2 * np.sin(x / 11)}
    )
    model = NeuralProphet(
        n_lags=5,
        n_forecasts=3,
        epochs=3,
        batch_size=16,
        learning_rate=None,
        weekly_seasonality=True,
        yearly_seasonality=False,
        daily_seasonality=False,
        collect_metrics=False,
        accelerator="cpu",
        impute_missing=False,
        trainer_config={
            "logger": False,
            "enable_progress_bar": False,
            "enable_checkpointing": False,
            "deterministic": True,
        },
    )
    with pytest.raises(ValueError, match="^owned_test_probe_complete$"):
        model.fit(history, freq="D", progress=None, checkpointing=False)
    assert len(captured) == 1
    assert 3 <= len(captured[0].results["lr"]) == len(actual_losses) <= 100
    assert model.config_train.epochs == 3 and model.trainer.fit_loop.max_epochs == 3
    assert model.config_train.learning_rate is None and model.model.learning_rate is None
    assert model.model.finding_lr is False and model.fitted is False


def test_validated_automatic_rate_reaches_actual_optimizer_and_forecast(monkeypatch):
    """Mock only the captured range curve, not the actual optimizer/fit/predict."""
    recorded = json.loads((Path(__file__).parent / "test-data" / "lr_finder_comparable_training.json").read_text())
    expected = module.smooth_loss_and_suggest(finder(recorded["loss"], recorded["lr"]))[2]
    probes = []

    def lr_find(self, **kwargs):
        assert kwargs["update_attr"] is False
        kwargs["model"].learning_rate = 0.1  # Temporary TimeNet probe initialization.
        probes.append(kwargs)
        return finder(recorded["loss"], recorded["lr"])

    monkeypatch.setattr(module.Tuner, "lr_find", lr_find)
    set_random_seed(17)
    x = np.arange(180)
    history = pd.DataFrame(
        {"ds": pd.date_range("2025-01-01", periods=len(x)), "y": 100 + x * 0.05 + 2 * np.sin(x / 11)}
    )
    model = NeuralProphet(
        n_lags=5,
        n_forecasts=3,
        epochs=3,
        batch_size=16,
        learning_rate=None,
        weekly_seasonality=True,
        yearly_seasonality=False,
        daily_seasonality=False,
        collect_metrics=False,
        accelerator="cpu",
        impute_missing=False,
        trainer_config={
            "logger": False,
            "enable_progress_bar": False,
            "enable_checkpointing": False,
            "deterministic": True,
        },
    )
    model.fit(history, freq="D", progress=None, checkpointing=False)
    assert len(probes) == 1 and model.fitted
    assert model.config_train.learning_rate == model.model.learning_rate == expected
    optimizer = model.trainer.optimizers[0]
    assert optimizer.param_groups and all(group["max_lr"] == expected for group in optimizer.param_groups)
    assert model.config_train.epochs == model.trainer.fit_loop.max_epochs == 3
    future = model.make_future_dataframe(history, periods=3, n_historic_predictions=False)
    forecast = model.predict(future)
    assert np.isfinite(forecast["yhat3"].dropna()).all()
    assert forecast["yhat3"].notna().sum() == 1
