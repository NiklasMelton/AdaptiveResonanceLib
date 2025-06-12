"""Adaptive Resonance Theory (ART) is a cognitive and neural network model that explains
how the brain learns to recognize patterns while maintaining stability in the face of
new, potentially conflicting information. ART networks are known for their ability to
perform unsupervised learning and adaptively categorize data without forgetting
previously learned patterns, a feature known as "plasticity-stability balance.".

The ART modules provided here support classification, clustering, and reinforcement
learning tasks by dynamically adjusting to incoming data streams. They also offer
advanced capabilities, including hierarchical clustering, topological clustering, data
fusion, and regression, enabling flexible exploration of complex data structures.

`Adaptive Resonance Theory <https://en.wikipedia.org/wiki/Adaptive_resonance_theory>`_

"""


from artlib.common.BaseART import BaseART
from artlib.common.BaseARTMAP import BaseARTMAP
from artlib.common.utils import (
    normalize,
    complement_code,
    de_complement_code,
    de_normalize,
)
from artlib.common.VAT import VAT

from artlib.elementary.ART1 import ART1
from artlib.elementary.ART2 import ART2A
from artlib.elementary.BayesianART import BayesianART
from artlib.elementary.EllipsoidART import EllipsoidART
from artlib.elementary.GaussianART import GaussianART
from artlib.elementary.FuzzyART import FuzzyART
from artlib.elementary.BinaryFuzzyART import BinaryFuzzyART
from artlib.elementary.HypersphereART import HypersphereART
from artlib.elementary.QuadraticNeuronART import QuadraticNeuronART

from artlib.cvi.iCVIFuzzyArt import iCVIFuzzyART
from artlib.cvi.CVIART import CVIART

from artlib.supervised.ARTMAP import ARTMAP
from artlib.supervised.SimpleARTMAP import SimpleARTMAP

from artlib.hierarchical.SMART import SMART
from artlib.hierarchical.DeepARTMAP import DeepARTMAP

from artlib.fusion.FusionART import FusionART

from artlib.reinforcement.FALCON import FALCON, TD_FALCON

from artlib.biclustering.BARTMAP import BARTMAP

from artlib.topological.TopoART import TopoART
from artlib.topological.DualVigilanceART import DualVigilanceART

from artlib.cpp_optimized.BinaryFuzzyARTMAP import BinaryFuzzyARTMAP
from artlib.cpp_optimized.FuzzyARTMAP import FuzzyARTMAP
from artlib.cpp_optimized.HypersphereARTMAP import HypersphereARTMAP
from artlib.cpp_optimized.GaussianARTMAP import GaussianARTMAP

__all__ = [
    "BaseART",
    "BaseARTMAP",
    "normalize",
    "complement_code",
    "de_complement_code",
    "de_normalize",
    "VAT",
    "ART1",
    "ART2A",
    "BayesianART",
    "GaussianART",
    "EllipsoidART",
    "HypersphereART",
    "QuadraticNeuronART",
    "FuzzyART",
    "BinaryFuzzyART",
    "TopoART",
    "DualVigilanceART",
    "ARTMAP",
    "SimpleARTMAP",
    "DeepARTMAP",
    "SMART",
    "FusionART",
    "BARTMAP",
    "iCVIFuzzyART",
    "CVIART",
    "FALCON",
    "TD_FALCON",
    "BinaryFuzzyARTMAP",
    "FuzzyARTMAP",
    "HypersphereARTMAP",
    "GaussianARTMAP",
]
