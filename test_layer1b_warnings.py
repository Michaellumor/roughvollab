# -*- coding: utf-8 -*-
"""Regression tests for RVL-040 and RVL-046 — importing the module must not
mutate process-global state.

`layer1b_mlmc_asian` used to run a bare ``warnings.filterwarnings("ignore")`` at
module import, installing a process-global ignore-all filter that silenced every
warning (including numpy overflow ``RuntimeWarning``) for any script whose import
graph reached it. The filter is now gated behind ``if __name__ == "__main__":``.

This test asserts, in a *fresh* interpreter, that merely importing the module no
longer poisons the process: no catch-all ignore lands in ``warnings.filters`` and
an overflow ``RuntimeWarning`` still surfaces.
"""
import os
import subprocess
import sys

_REPO = os.path.dirname(os.path.abspath(__file__))

# Runs in a fresh interpreter so the warning state is pristine (pytest resets
# filters per-test, which is exactly why the in-process suite never caught this).
_PROBE = r"""
import warnings, numpy as np

def _is_catchall(f):
    # A bare filterwarnings("ignore") / simplefilter("ignore") -> action 'ignore',
    # base Warning category, and a message regex that matches everything (empty
    # pattern, or None). A message-scoped ignore installed by numpy/matplotlib
    # (e.g. the np.matrix PendingDeprecationWarning) is NOT a catch-all and is fine.
    action, msg, cat, mod, lineno = f
    matches_all = (msg is None) or (getattr(msg, "pattern", "") == "")
    return action == "ignore" and cat is Warning and matches_all

before = list(warnings.filters)
import layer1b_mlmc_asian                      # must not poison the process
added = [f for f in warnings.filters if f not in before]

# (1) importing the module must add no catch-all ignore filter:
catchall = [f for f in added if _is_catchall(f)]
assert not catchall, "import added a catch-all ignore filter: %r" % catchall

# (2) an overflow RuntimeWarning must still surface under the default filters
#     (do NOT touch the filters here, or we'd mask a poison filter from the import):
np.seterr(over="warn")
with warnings.catch_warnings(record=True) as rec:
    _ = np.exp(np.array([800.0]))
assert any(issubclass(w.category, RuntimeWarning) for w in rec), \
    "overflow RuntimeWarning was silenced after importing layer1b_mlmc_asian"

print("RVL040-OK")
"""


def test_importing_layer1b_does_not_silence_warnings_process_wide():
    r = subprocess.run([sys.executable, "-c", _PROBE],
                       capture_output=True, text=True, cwd=_REPO)
    assert r.returncode == 0 and "RVL040-OK" in r.stdout, (
        "fresh-process warning probe failed:\n"
        "--- stdout ---\n%s\n--- stderr ---\n%s" % (r.stdout, r.stderr)
    )


# ── RVL-046: importing the module must not reseed numpy's GLOBAL RNG ─────────
# `np.random.seed(42)` used to run at module import, silently reseeding the
# global RNG for every importer. It now lives under `if __name__ == "__main__":`.
_RNG_PROBE = r"""
import numpy as np

# Put the global RNG in a KNOWN state distinct from the module's seed(42), so a
# reseed-on-import is unambiguously detectable (not a coincidental match).
np.random.seed(12345)
before = np.random.get_state()
import layer1b_mlmc_asian                      # must not touch the global RNG
after = np.random.get_state()

assert before[0] == after[0], "global RNG bit-generator changed on import"
assert np.array_equal(before[1], after[1]) and before[2] == after[2], \
    "importing layer1b_mlmc_asian reseeded numpy's global RNG"

print("RVL046-OK")
"""


def test_importing_layer1b_does_not_reseed_global_rng():
    r = subprocess.run([sys.executable, "-c", _RNG_PROBE],
                       capture_output=True, text=True, cwd=_REPO)
    assert r.returncode == 0 and "RVL046-OK" in r.stdout, (
        "fresh-process RNG probe failed:\n"
        "--- stdout ---\n%s\n--- stderr ---\n%s" % (r.stdout, r.stderr)
    )
