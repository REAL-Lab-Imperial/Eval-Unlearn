"""Unit tests for generic fallback target-prompt synthesis."""
from eval_unlearn.metrics._default_prompts import default_target_prompts


def test_returns_requested_count():
    prompts = default_target_prompts("a fake concept", 37)
    assert len(prompts) == 37


def test_prompts_reference_the_concept():
    prompts = default_target_prompts("a fake concept", 10)
    assert all("a fake concept" in p for p in prompts)


def test_prompts_are_unique_within_template_budget():
    prompts = default_target_prompts("a fake concept", 100)
    assert len(set(prompts)) == len(prompts)


def test_zero_requested_returns_empty():
    assert default_target_prompts("a fake concept", 0) == []


def test_more_than_template_budget_still_returns_exact_count():
    """120 template combos exist; requesting more should still return exactly n,
    cycling with distinguishing suffixes rather than truncating."""
    prompts = default_target_prompts("a fake concept", 250)
    assert len(prompts) == 250
    assert len(set(prompts)) == 250
