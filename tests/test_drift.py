"""tests/test_drift.py"""

import numpy as np

from app.drift import psi


def test_psi_zero_for_identical():
    a = np.random.normal(0, 1, 1000)
    assert psi(a, a) < 0.01


def test_psi_high_for_shifted():
    a = np.random.normal(0, 1, 1000)
    b = np.random.normal(3, 1, 1000)
    assert psi(a, b) > 0.5


def test_psi_handles_empty():
    assert psi(np.array([]), np.array([1, 2, 3])) == 0.0


def test_psi_moderate_shift():
    a = np.random.normal(0, 1, 1000)
    b = np.random.normal(0.5, 1.2, 1000)
    score = psi(a, b)
    assert 0.0 < score < 1.0
