"""The optional tqdm dependency is only needed for progress bars."""

import builtins
from contextlib import contextmanager

import numpy as np
import pytest
from matplotlib.animation import PillowWriter

from artlib.elementary.FuzzyART import FuzzyART
from artlib.supervised.SimpleARTMAP import SimpleARTMAP


@pytest.mark.parametrize("supervised", [False, True])
@pytest.mark.parametrize("method", ["fit", "fit_gif"])
def test_missing_tqdm_message(monkeypatch, tmp_path, supervised, method):
    module = FuzzyART(0.5, 0.01, 1.0)
    model = SimpleARTMAP(module) if supervised else module
    X = module.prepare_data(np.array([[0.2, 0.4]]))
    y = np.array([0])

    @contextmanager
    def saving(*args, **kwargs):
        yield

    # Avoid writing a GIF; the import fails before any frames are rendered.
    monkeypatch.setattr(PillowWriter, "saving", saving)
    original_import = builtins.__import__

    def import_without_tqdm(name, *args, **kwargs):
        if name == "tqdm":
            raise ModuleNotFoundError("No module named 'tqdm'")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", import_without_tqdm)
    args = (X, y) if supervised else (X,)
    kwargs = {"verbose": True}
    if method == "fit_gif":
        kwargs["filename"] = str(tmp_path / "unused.gif")

    with pytest.raises(ImportError, match="pip install tqdm"):
        getattr(model, method)(*args, **kwargs)


@pytest.mark.parametrize("supervised", [False, True])
def test_fit_without_tqdm_when_not_verbose(monkeypatch, supervised):
    module = FuzzyART(0.5, 0.01, 1.0)
    model = SimpleARTMAP(module) if supervised else module
    X = module.prepare_data(np.array([[0.2, 0.4]]))
    y = np.array([0])
    original_import = builtins.__import__

    def import_without_tqdm(name, *args, **kwargs):
        if name == "tqdm":
            raise ModuleNotFoundError("No module named 'tqdm'")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", import_without_tqdm)
    args = (X, y) if supervised else (X,)

    assert model.fit(*args, verbose=False) is model
