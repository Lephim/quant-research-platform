"""Offline checks for held-out evaluation scoring."""
from agent.blind_cases import CASES
from agent.evaluate_blind import score_case


def test_snapshot_identity_and_exact_calls_required():
    case = CASES[0]
    answer = "SHA-256: abc\nRun 1\nRun 2"
    assert score_case(case, case["expected"], answer, None, "abc")["passed"]
    assert not score_case(case, case["expected"], answer, None, "different")["passed"]
    assert not score_case(case, case["expected"][:1], answer, None, "abc")["passed"]


def test_no_tool_case_needs_helpful_response():
    case = next(case for case in CASES if case["id"] == "wrong_snapshot_universe")
    assert score_case(case, [], "The snapshot contains multi_asset.", None, "abc")["passed"]
    assert not score_case(case, [], "Done.", None, "abc")["passed"]
