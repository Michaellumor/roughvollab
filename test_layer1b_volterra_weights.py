# -*- coding: utf-8 -*-
"""Regression tests for RVL-007 / RVL-022.

`layer1b_mlmc_asian` used to carry its own `volterra_weights` duplicate: it was
byte-identical to `roughvol_core.volterra_weights` for valid H, but it had no
0<H<1 guard (it returned silent nonsense for H in {0, 1, -0.1}) and it crashed
with ZeroDivisionError at H=1/2. It is now imported from roughvol_core — the
single validated engine. These tests pin the guard, that valid H (incl. H=1/2)
works, and that the layer1b entry point is byte-identical to core.
"""
import os
os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np
import pytest

from layer1b_mlmc_asian import volterra_weights as l1b_volterra_weights
import roughvol_core


def test_volterra_weights_rejects_invalid_H():
    """RVL-007: H outside (0,1) must raise ValueError via the layer1b entry point."""
    for bad_H in (0.0, 1.0, -0.1):
        with pytest.raises(ValueError):
            l1b_volterra_weights(64, bad_H, 1.0)


def test_volterra_weights_valid_H_returns_weights():
    """Valid H still works — including H=1/2, which the old duplicate crashed on."""
    for H in (0.1, 0.5):
        g, v = l1b_volterra_weights(64, H, 1.0)
        assert g.shape == (64,) and v.shape == (64,)
        assert np.all(np.isfinite(g)) and np.all(np.isfinite(v))
    # H=1/2 is the flat-kernel (standard BM) case: g == 1
    g_half, _ = l1b_volterra_weights(64, 0.5, 1.0)
    assert np.allclose(g_half, 1.0)


def test_layer1b_volterra_weights_matches_core():
    """DELETE-path proof: the layer1b entry point IS core's, byte-identical for valid H,T."""
    assert l1b_volterra_weights is roughvol_core.volterra_weights
    for n, H, T in [(64, 0.1, 1.0), (128, 0.3, 2.0), (100, 0.7, 1.5)]:
        g1, v1 = l1b_volterra_weights(n, H, T)
        g2, v2 = roughvol_core.volterra_weights(n, H, T)
        assert np.array_equal(g1, g2) and np.array_equal(v1, v2)


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
