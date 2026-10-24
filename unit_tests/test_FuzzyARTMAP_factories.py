from _factory_parity import run_parity
from artlib.optimized.FuzzyARTMAPFactory import FuzzyARTMAPFactory


def test_fuzzy_artmap_factories(capsys):
    RHO, ALPHA, BETA = 0.8, 1e-10, 1.0
    run_parity(
        {
            "torch": FuzzyARTMAPFactory(RHO, ALPHA, BETA, backend="torch"),
            "c++": FuzzyARTMAPFactory(RHO, ALPHA, BETA, backend="c++"),
        },
        capsys,
    )
