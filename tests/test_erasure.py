"""LEACE removes linearly available gender information (closed form and concept-erasure)."""
import numpy as np
import pytest
import torch

from advice_audit.local.erasure import ClosedFormLeace
from advice_audit.local.probes import cv_accuracy


def _data(n=600, d=16, seed=0):
    g = torch.Generator().manual_seed(seed)
    z = torch.randint(0, 2, (n,), generator=g)
    X = torch.randn(n, d, generator=g)
    X[:, 0] += 3.0 * z  # gender signal
    X[:, 1] += 1.5 * z
    groups = np.arange(n) % 10
    return X, z, groups


def test_closed_form_leace_erases():
    X, z, groups = _data()
    assert cv_accuracy(X, z, groups, 5, 1e-2, 100) > 0.9
    e = ClosedFormLeace(X, z)
    Xe = e(X)
    assert cv_accuracy(Xe, z, groups, 5, 1e-2, 100) < 0.6
    # class-conditional means become equal
    torch.testing.assert_close(Xe[z == 1].mean(0), Xe[z == 0].mean(0), atol=1e-3, rtol=0)


def test_matches_concept_erasure_package():
    ce = pytest.importorskip("concept_erasure")
    X, z, _ = _data()
    a = ClosedFormLeace(X, z)(X)
    b = ce.LeaceEraser.fit(X, z, shrinkage=False)(X)  # package default shrinks the covariance
    torch.testing.assert_close(a, b, atol=1e-3, rtol=1e-3)
