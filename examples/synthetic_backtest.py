"""Offline API demonstration. Synthetic numbers are not empirical research results."""
import numpy as np
import pandas as pd
from backtesting.walk_forward import walk_forward_backtest, equal_weight_backtest
from backtesting.metrics import performance_metrics

rng = np.random.default_rng(42)
returns = pd.DataFrame(rng.normal(.0004, .01, (300, 5)),
    index=pd.bdate_range("2020-01-01", periods=300), columns=list("ABCDE"))
start = returns.index[126]
strategy, weights, turnover = walk_forward_backtest(
    returns, train_window=126, evaluation_start=start,
    covariance_method="ledoit_wolf", transaction_cost_bps=10)
benchmark, _, _ = equal_weight_backtest(
    returns, train_window=126, evaluation_start=start, transaction_cost_bps=10)
assert strategy.index.equals(benchmark.index)
print("SYNTHETIC DATA — API demonstration only")
print(pd.DataFrame({"Optimizer": performance_metrics(strategy),
                    "Equal weight": performance_metrics(benchmark)}))
