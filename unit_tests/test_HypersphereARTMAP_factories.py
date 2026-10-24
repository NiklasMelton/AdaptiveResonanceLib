from _factory_parity import run_parity
from artlib.optimized.HypersphereARTMAPFactory import HypersphereARTMAPFactory


def test_hypersphere_artmap_factories(capsys):
    RHO, ALPHA, BETA, R_HAT = 0.8, 1e-10, 1.0, 28.0
    run_parity(
        {
            "torch": HypersphereARTMAPFactory(RHO, ALPHA, BETA, R_HAT, backend="torch"),
            "c++": HypersphereARTMAPFactory(RHO, ALPHA, BETA, R_HAT, backend="c++"),
        },
        capsys,
    )
