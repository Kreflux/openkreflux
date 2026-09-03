"""
Unit tests for ReasoningVerifier and ReasoningLadder.
"""

import pytest

from openkreflux.verifier import (
    LadderLevel,
    ReasoningLadder,
    ReasoningVerifier,
)


def test_extract_thoughts_standard():
    verifier = ReasoningVerifier()
    trace = "<think>Let us analyze the premise first.\nStep 1: check x.</think>The answer is 42."
    thoughts, solution = verifier.extract_thoughts(trace)
    assert len(thoughts) == 1
    assert "Step 1: check x." in thoughts[0]
    assert solution == "The answer is 42."


def test_extract_thoughts_multiple_blocks():
    verifier = ReasoningVerifier()
    trace = "<think>Part 1</think>Interim<think>Part 2</think>Final result."
    thoughts, solution = verifier.extract_thoughts(trace)
    assert thoughts == ["Part 1", "Part 2"]
    assert solution == "InterimFinal result."


def test_extract_thoughts_unclosed():
    verifier = ReasoningVerifier()
    trace = "Initial<think>Incomplete thinking without closing tag"
    thoughts, solution = verifier.extract_thoughts(trace)
    assert thoughts == ["Incomplete thinking without closing tag"]
    assert solution == "Initial"


def test_repetition_detection():
    verifier = ReasoningVerifier()
    # Normal text
    clean_text = (
        "We first evaluate the derivative with respect to x. Then we take the integral "
        "and compare boundary conditions to ensure numerical stability."
    )
    is_rep, _ = verifier.check_repetition(clean_text)
    assert is_rep is False

    # Repetitive loop text
    loop_phrase = "we must repeat this calculation " * 10
    is_rep, msg = verifier.check_repetition(loop_phrase)
    assert is_rep is True
    assert "Repetitive 4-gram loop" in msg


def test_math_validation():
    verifier = ReasoningVerifier()

    # Valid LaTeX math
    valid_trace = "The equation is $$x = \\frac{-b \\pm \\sqrt{b^2 - 4ac}}{2a}$$ which yields $x = 1$."
    valid, issues = verifier.validate_math(valid_trace)
    assert valid is True
    assert len(issues) == 0

    # Unbalanced single dollar sign
    invalid_dollars = "The formula is $x + y = z with no closing dollar."
    valid, issues = verifier.validate_math(invalid_dollars)
    assert valid is False
    assert any("Unbalanced '$'" in i for i in issues)

    # Unmatched brackets inside math
    unmatched_brackets = "Consider $$(a + [b - c)$$."
    valid, issues = verifier.validate_math(unmatched_brackets)
    assert valid is False
    assert any("Unmatched math delimiter" in i for i in issues)


def test_code_validation():
    verifier = ReasoningVerifier()

    # Valid Python code block
    valid_code = "Here is the implementation:\n```python\ndef solve(x: int) -> int:\n    return x * 2\n```"
    valid, issues = verifier.validate_code(valid_code)
    assert valid is True
    assert len(issues) == 0

    # Python code block with syntax error
    invalid_code = "```python\ndef broken(:\n    return\n```"
    valid, issues = verifier.validate_code(invalid_code)
    assert valid is False
    assert any("Python syntax error" in i for i in issues)

    # Unbalanced markdown fences
    unclosed_fence = "```python\ndef test(): pass\n"
    valid, issues = verifier.validate_code(unclosed_fence)
    assert valid is False
    assert any("Unbalanced markdown code fences" in i for i in issues)


def test_reasoning_density_calculation():
    verifier = ReasoningVerifier()
    trace = "<think>" + ("word " * 60) + "</think>" + ("word " * 40)
    result = verifier.verify(trace)
    assert result.thinking_tokens > 0
    assert result.solution_tokens > 0
    # ~60 / 100 ~ 0.60
    assert 0.50 <= result.reasoning_density <= 0.70


def test_ladder_levels_classification():
    # Low level
    assert ReasoningLadder.evaluate(
        thinking_tokens=30, step_count=1, coherence_score=0.2, has_self_correction=False
    ) == LadderLevel.LOW

    # Medium level
    assert ReasoningLadder.evaluate(
        thinking_tokens=200, step_count=3, coherence_score=0.4, has_self_correction=False
    ) == LadderLevel.MEDIUM

    # High level
    assert ReasoningLadder.evaluate(
        thinking_tokens=600, step_count=5, coherence_score=0.6, has_self_correction=False
    ) == LadderLevel.HIGH

    # Ultra level
    assert ReasoningLadder.evaluate(
        thinking_tokens=1500, step_count=8, coherence_score=0.8, has_self_correction=True
    ) == LadderLevel.ULTRA


def test_full_trace_verification():
    verifier = ReasoningVerifier()

    rich_trace = """
<think>
Let us break down this problem systematically.
Step 1: Observe given constraints.
Step 2: Check edge cases for zero and negative numbers.
Wait, let me double check if negative inputs are allowed.
Actually, the problem specifies positive integers only.
Therefore, we can proceed with a simple dynamic programming approach.
Hence, the state transition relation is established.
</think>
The final answer is $O(N)$.
```python
def solve(n: int) -> int:
    return n * (n + 1) // 2
```
"""
    result = verifier.verify(rich_trace, target_ladder=LadderLevel.LOW)
    assert result.is_valid is True
    assert result.has_self_correction is True
    assert result.math_validity is True
    assert result.code_validity is True
    assert result.meets_target_ladder is True
    assert len(result.issues) == 0
