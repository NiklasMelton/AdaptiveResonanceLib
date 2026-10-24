import numpy as np

from _factory_parity import run_parity
from artlib.optimized.GaussianARTMAPFactory import GaussianARTMAPFactory


def test_gaussian_artmap_factories(capsys):
    RHO, ALPHA, SIGMA_INIT = 0.05, 1e-10, 0.33 * np.ones((784,))
    run_parity(
        {
            "torch": GaussianARTMAPFactory(RHO, ALPHA, SIGMA_INIT, backend="torch"),
            "c++": GaussianARTMAPFactory(RHO, ALPHA, SIGMA_INIT, backend="c++"),
        },
        capsys,
    )
