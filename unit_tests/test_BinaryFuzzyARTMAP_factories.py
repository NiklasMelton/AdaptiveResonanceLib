from _factory_parity import run_parity
from artlib.optimized.BinaryFuzzyARTMAPFactory import BinaryFuzzyARTMAPFactory


def test_binary_fuzzy_artmap_factories(capsys):
    RHO = 0.8
    run_parity(
        {
            "python": BinaryFuzzyARTMAPFactory(RHO, backend="python"),
            "torch": BinaryFuzzyARTMAPFactory(RHO, backend="torch"),
            "c++": BinaryFuzzyARTMAPFactory(RHO, backend="c++"),
        },
        capsys,
        binary=True,
    )
