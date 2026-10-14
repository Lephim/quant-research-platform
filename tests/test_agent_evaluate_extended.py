"""Checks for the expanded evaluation's pass criteria."""
from agent.evaluate_extended import CASES, score_case


def test_tool_case_requires_all_requested_calls_and_grounded_response():
    case = CASES[0]
    answer = "Backtest results (values reported directly by the engine):\nRun 1\nRun 2"
    assert score_case(case, case["expected_calls"], answer, None)["passed"]
    assert not score_case(case, case["expected_calls"][:1], answer, None)["passed"]


def test_clarification_case_requires_no_tool_and_useful_answer():
    case = CASES[1]
    assert score_case(case, [], "Which universe should I use?", None)["passed"]
    assert not score_case(case, [], "Done.", None)["passed"]
