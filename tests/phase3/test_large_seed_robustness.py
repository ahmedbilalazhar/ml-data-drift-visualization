from scripts.evaluate_phase3_large_seed import exact_harm


def test_harm_boundary_is_strict_and_exact():
    # 80/100 minus 75/100 is exactly .05, so the strict rule is false.
    assert not exact_harm(80, 100, 75, 100)
    assert exact_harm(80, 100, 74, 100)


def test_harm_uses_rational_integer_boundary_for_unequal_denominators():
    # 3/4 minus 7/10 is exactly .05; changing the batch count by one crosses it.
    assert not exact_harm(3, 4, 70, 100)
    assert exact_harm(3, 4, 69, 100)
