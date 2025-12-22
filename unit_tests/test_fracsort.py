import numpy as np

from artlib.common.utils import fracsort


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
    n = 10_000

    den = rng.integers(1, 10_001, size=n, dtype=np.uint32)
    num = rng.integers(0, den + 1, size=n, dtype=np.uint32)

    # Ensure contiguous arrays (the backend requires C-contiguous input)
    den = np.ascontiguousarray(den)
    num = np.ascontiguousarray(num)

    idx_cpp = fracsort(num, den)

    ratio = num.astype(np.float64) / den.astype(np.float64)
    idx_np = np.argsort(ratio, kind="stable")[::-1]

    assert np.array_equal(idx_cpp, idx_np)
