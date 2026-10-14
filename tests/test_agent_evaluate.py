"""Checks for comparison scoring in the offline agent evaluation."""
from agent.evaluate import COMMON, normalize, score_calls


def test_effective_defaults_and_order_independence():
    sample = {**COMMON, "covariance_method": "sample"}
    wolf = {**COMMON, "covariance_method": "ledoit_wolf"}
    assert score_calls([wolf, sample], [sample, wolf])
    assert score_calls([{**sample, "transaction_cost_bps": 10}], [sample])
    assert normalize({"universe": "multi_asset"})["covariance_method"] == "sample"
    assert not score_calls([sample, sample], [sample, wolf])
