from types import SimpleNamespace
import numpy as np
import pandas as pd
import pytest
from analytics.estimation import sample_covariance, ledoit_wolf_covariance, shrink_expected_returns
from analytics.optimization import optimize_max_sharpe, random_weights_constrained
from backtesting.metrics import performance_metrics
import backtesting.walk_forward as wf


@pytest.fixture
def returns():
    rng = np.random.default_rng(7)
    return pd.DataFrame(rng.normal(.001, .01, (50, 5)),
                        index=pd.bdate_range("2020-01-01", periods=50), columns=list("ABCDE"))


def test_covariance_scaling_and_labels(returns):
    np.testing.assert_allclose(sample_covariance(returns), returns.cov() * 252)
    cov, intensity = ledoit_wolf_covariance(returns)
    assert cov.columns.equals(returns.columns)
    assert cov.index.equals(returns.columns)
    assert 0 <= intensity <= 1
    assert np.linalg.eigvalsh(cov).min() >= -1e-12


def test_shrinkage_and_zero_variance(returns):
    mu, weights, target = shrink_expected_returns(returns, annualize=False)
    assert weights.between(0, 1).all()
    assert (mu >= np.minimum(returns.mean(), target) - 1e-12).all()
    assert (mu <= np.maximum(returns.mean(), target) + 1e-12).all()
    mu, _, _ = shrink_expected_returns(returns * 0)
    assert (mu == 0).all()


def test_optimizer_constraints(returns):
    result = optimize_max_sharpe(returns.mean() * 252, sample_covariance(returns))
    assert result.success
    assert result.x.sum() == pytest.approx(1)
    assert result.x.min() >= -1e-8 and result.x.max() <= .30000001


def test_impossible_and_boundary_caps():
    with pytest.raises(ValueError, match="fully invested"):
        random_weights_constrained(2, .3)
    np.testing.assert_allclose(random_weights_constrained(5, .2), [.2] * 5)


def test_holding_period_drifts():
    r = pd.DataFrame([[.1, 0], [.1, 0]])
    actual = wf.holding_period_returns(r, np.array([.5, .5]))
    np.testing.assert_allclose(actual, [.05, 1.105 / 1.05 - 1])


def test_turnover_and_cost_from_drift():
    r = pd.DataFrame([[0, 0], [0, 0], [.1, 0], [.1, 0], [0, 0], [0, 0]],
                     index=pd.bdate_range("2020-01-01", periods=6))
    gross, weights, turnover = wf.equal_weight_backtest(r, 2, 2)
    net, _, net_turnover = wf.equal_weight_backtest(r, 2, 2, 10)
    expected = .605 / 1.105 - .5
    assert turnover == pytest.approx([expected])
    assert net_turnover == turnover
    assert net.iloc[2] == pytest.approx(gross.iloc[2] - expected * .001)
    np.testing.assert_allclose(net.iloc[[0, 1, 3]], gross.iloc[[0, 1, 3]])
    assert len(turnover) == len(weights) - 1


def test_common_dates_and_no_lookahead(returns, monkeypatch):
    calls = []
    def optimizer(mu, cov, **kwargs):
        calls.append(mu.copy())
        return SimpleNamespace(success=True, x=np.ones(5) / 5)
    monkeypatch.setattr(wf, "optimize_max_sharpe", optimizer)
    first = returns.index[20]
    out, _, _ = wf.walk_forward_backtest(returns, 10, 7, evaluation_start=first)
    pd.testing.assert_series_equal(calls[0], returns.iloc[10:20].mean() * 252)
    assert out.index.equals(returns.index[20:])
    ew, _, _ = wf.equal_weight_backtest(returns, 15, 7, evaluation_start=first)
    assert out.index.equals(ew.index)
    changed = returns.copy(); changed.iloc[20:] += .1
    calls.clear()
    wf.walk_forward_backtest(changed, 10, 7, evaluation_start=first)
    pd.testing.assert_series_equal(calls[0], returns.iloc[10:20].mean() * 252)


def test_failed_optimization_is_visible(returns, monkeypatch):
    monkeypatch.setattr(wf, "optimize_max_sharpe", lambda *a, **k:
                        SimpleNamespace(success=False, message="test failure"))
    with pytest.raises(RuntimeError, match="test failure"):
        wf.walk_forward_backtest(returns, 10)


@pytest.mark.parametrize("kwargs", [{"train_window": 100}, {"rebalance_frequency": 0},
    {"transaction_cost_bps": -1}, {"evaluation_start": "2010-01-01"},
    {"evaluation_start": "2030-01-01"}])
def test_invalid_backtest_inputs(returns, kwargs):
    params = {"train_window": 10, **kwargs}
    with pytest.raises(ValueError):
        wf.equal_weight_backtest(returns, **params)


def test_unsorted_dates_rejected(returns):
    with pytest.raises(ValueError, match="sorted"):
        wf.walk_forward_backtest(returns.iloc[::-1], 10)


def test_drawdown_includes_initial_wealth():
    metrics = performance_metrics(pd.Series([-.1, .05]))
    assert metrics["Max Drawdown"] == pytest.approx(-.1)
    assert metrics["Total Return"] == pytest.approx(.9 * 1.05 - 1)
    with pytest.raises(ValueError):
        performance_metrics(pd.Series([], dtype=float))


@pytest.mark.parametrize("covariance", ["sample", "ledoit_wolf"])
@pytest.mark.parametrize("mean", ["historical", "shrunk"])
def test_estimator_combinations(returns, covariance, mean):
    out, weights, turnover = wf.walk_forward_backtest(returns, 20, 7,
        covariance_method=covariance, expected_return_method=mean)
    assert out.index.equals(returns.index[20:])
    assert np.isfinite(out).all()
    np.testing.assert_allclose(weights.sum(axis=1), 1)
    assert len(turnover) == len(weights) - 1


def test_sampler_attempt_limit(monkeypatch):
    monkeypatch.setattr(np.random, "random", lambda n: np.array([1., 0., 0., 0., 0.]))
    with pytest.raises(RuntimeError, match="exhausted"):
        random_weights_constrained(5, .3)


@pytest.mark.parametrize("value", [np.inf, -1.0, -1.1])
def test_invalid_return_values(returns, value):
    returns.iloc[25, 0] = value
    with pytest.raises(ValueError, match="finite and greater"):
        wf.walk_forward_backtest(returns, 10)


def test_missing_rows_are_aligned(returns):
    returns.iloc[25, 0] = np.nan
    out, _, _ = wf.equal_weight_backtest(returns, 10)
    assert out.index.equals(returns.dropna().index[10:])
