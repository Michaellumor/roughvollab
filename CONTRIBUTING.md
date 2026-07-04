# Contributing to RoughVolLab

Thank you for your interest in RoughVolLab. Contributions, bug reports and
questions are all welcome. This page explains how to get the code, run the
tests, and propose changes.

## Getting the code

Full clone (includes ~1.1 GB of committed market data under `data/`):

```bash
git clone https://github.com/Michaellumor/roughvollab.git
```

Code-only clone (~10 MB — recommended if you only need the library):

```bash
git clone --filter=blob:none --sparse https://github.com/Michaellumor/roughvollab.git
cd roughvollab
git sparse-checkout set --no-cone '/*' '!data/'
```

## Setting up and running the tests

```bash
pip install -r requirements.txt
python -m pytest -q
```

The suite collects 254 tests. The deep-hedging tests (`test_layer3_deep_hedging.py`)
require PyTorch and skip automatically when it is absent — see
`requirements-layer3.txt` for that optional environment.

## Reporting a bug or asking for help

Please open a GitHub issue at
<https://github.com/Michaellumor/roughvollab/issues>. For bugs, include the
command you ran, what you expected, what happened instead, and your Python and
NumPy versions. Questions about usage or the underlying methods are welcome in
issues too — there is no separate support channel.

## Proposing a change

1. Open an issue first for anything non-trivial, so the approach can be agreed
   before you write code.
2. Keep pull requests small and focused, with a clear description of what
   changed and why.
3. All tests must pass (`python -m pytest -q`), and behavioural changes should
   come with a test that pins the new behaviour.
4. Match the existing conventions: British spelling throughout (including in
   docstrings and comments), NumPy-style docstrings, and no new dependencies
   without prior discussion in an issue.
5. Every path simulation must draw from `roughvol_core.py` — never from the
   quarantined legacy engine (`layer1_rough_vol.py`, known issue L1-1).

## The decision log

Significant design decisions are recorded in `ROADMAP.md` as numbered entries
(D1, D2, …), each written down — with its predicted outcome — *before* the
corresponding experiment is run. If your change embodies a design decision,
please add an entry in the same style.

## Conduct

Be kind, be constructive, and criticise ideas rather than people. Reports of
unacceptable behaviour can be made privately to the maintainer.
