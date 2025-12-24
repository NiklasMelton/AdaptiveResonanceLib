import numpy as np

from artlib.common.utils import fracsort, fracargmax


def test_fracsort_matches_numpy_division_argsort() -> None:
    """Test that fracsort matches a NumPy argsort on the division-based proxy.

    This test generates random denominators in [1, 10000] and random numerators
    satisfying num <= den, then verifies that the index order returned by the
    compiled division-free fracsort matches NumPy's stable argsort of num/den.

    Notes
    -----
    - We use dtype uint32 to exercise the intended fast path.
    - NumPy sorting uses a stable sort so ties are broken by lowest index.

    """
    rng = np.random.default_rng(0)
    n = 10000

    den = rng.integers(1, n+1, size=n, dtype=np.uint32)
    num = rng.integers(0, den + 1, size=n, dtype=np.uint32)
    eps = 1e-10

    # Force exact ties in the ratio for a few indices:
    # 1/2 == 2/4; tie should pick larger denominator (4), then lower index among those.
    tie_idx = np.array([123, 456, 789], dtype=np.int64)
    den[tie_idx[0]] = np.uint32(2); num[tie_idx[0]] = np.uint32(1)
    den[tie_idx[1]] = np.uint32(4); num[tie_idx[1]] = np.uint32(2)
    den[tie_idx[2]] = np.uint32(4); num[tie_idx[2]] = np.uint32(2)

    # Ensure contiguous arrays (the backend requires C-contiguous input)
    den = np.ascontiguousarray(den)
    num = np.ascontiguousarray(num)

    idx_cpp = fracsort(num, den)

    ratio = num.astype(np.float64) / (den.astype(np.float64) + eps)
    idx_np = np.argsort(-ratio, kind="stable")
    assert np.array_equal(idx_cpp, idx_np)


def test_fracargmax_matches_numpy_with_tiebreaks() -> None:
    """Test that fracargmax matches a NumPy argmax on the division-based proxy.

    We verify that fracargmax returns the index maximizing num/den, using a float
    proxy for the ratio and explicitly implementing the tie-breaks:
      1) larger denominator
      2) lower index

    Notes
    -----
    - Use dtype uint32 (fast path).
    - We keep num <= den so ratios are in [0, 1] and float comparisons are well-behaved.
    - We force ties to ensure tie-break logic is exercised.

    """
    rng = np.random.default_rng(1)
    n = 10000

    den = rng.integers(1, n + 1, size=n, dtype=np.uint32)
    num = rng.integers(0, den + 1, size=n, dtype=np.uint32)
    eps = 1e-10

    # Force exact ties in the ratio for a few indices:
    # 1/2 == 2/4; tie should pick larger denominator (4), then lower index among those.
    tie_idx = np.array([123, 456, 789], dtype=np.int64)
    den[tie_idx[0]] = np.uint32(2); num[tie_idx[0]] = np.uint32(1)
    den[tie_idx[1]] = np.uint32(4); num[tie_idx[1]] = np.uint32(2)
    den[tie_idx[2]] = np.uint32(4); num[tie_idx[2]] = np.uint32(2)

    # Ensure contiguous arrays (the backend requires C-contiguous input)
    den = np.ascontiguousarray(den)
    num = np.ascontiguousarray(num)

    idx_cpp = fracargmax(num, den)

    ratio = num.astype(np.float64) / (den.astype(np.float64) + eps)
    max_ratio = ratio.max()

    # Candidates within exact max (float proxy).
    # Since we constrained num<=den and eps is tiny,
    # this is stable enough for the test; ties are resolved explicitly afterward.
    candidates = np.flatnonzero(ratio == max_ratio)
    # Apply tiebreaks: larger den, then lower index
    best = max(candidates, key=lambda i: (int(den[i]), -int(i)))  # max by den, then min i
    # The key above uses -i so larger key corresponds to lower index.

    assert int(idx_cpp) == int(best)