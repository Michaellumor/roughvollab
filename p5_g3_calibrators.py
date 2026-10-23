"""
p5_g3_calibrators.py — P5 gate G3, the two calibrators (docs/protocols/P5_protocol_v2_draft.md,
Objects "The calibrators" and "Architecture and training", D68; Gates, G3, D67; the seeds of D68).

The calibrators, each trained per prior and per noise level on surfaces y = G̃(θ) + s ε
from the emulator (p5_g1_emulator.py), both on one trunk: a multilayer perceptron with
three hidden layers of 128 units and ReLU, PyTorch's default initialisation, no
normalisation, no dropout. Inputs: the 35 quotes standardised by their mean and standard
deviation over the run's n training surfaces; θ, as input or target, mapped affinely to
[0, 1]⁴. The ensemble: five heteroscedastic members with four outputs (the means of H and
ξ₀ in mapped coordinates and two raw variances, softplus(raw) + 1e-6), the Gaussian
negative log-likelihood summed over the two parameters and averaged over the batch; Ĥ
and the ξ₀ estimate the means over members in the box's units; the total variance of H
the mean of the members' variances plus the variance of their means, in box units, the
two reported separately; the 90% interval Ĥ ± 1.645 √(total variance); nothing clipped.
The ratio estimator: the binary-classifier variant, 39 inputs, one logit read as log
r(θ, y); each batch of B joint pairs, labelled 1, paired with the B marginal pairs made by
the cyclic shift of θ by one within the batch, labelled 0; binary cross-entropy over the
2B pairs; its posterior under the training prior P, density ∝ P(H) r(θ, y), sampled by
the sampler of p5_g2_reference.py with log P(H) + logit as the target, the same chains,
lengths, adaptation, acceptance and single repeat, generator default_rng([107, p, q, i, j]);
Ĥ and ξ₀ the means over the pooled sampling steps, the 90% interval equal-tailed; a
failure after the repeat is the calibrator's failure on that pair at that seed.

Optimiser and stopping: Adam at 0.001, β (0.9, 0.999), no weight decay, batch 256 with a
partial final batch kept, the rate constant; a validation set of 4,000 surfaces per
prior and level from default_rng([113, p, j]); the objective on it after every epoch in
drawn order, batches of 256 with the final 160 kept, the marginal pairs formed as in
training; stop when it has not improved for 50 epochs or at 1,000; best parameters kept;
per member. A non-finite loss stops the run (at G3 the rung fails; after G3 one
retraining from seed value 6). Training seeds q = 1 to 5, a seed being the whole
calibrator: the draw default_rng([109, p, j, q]) of one 320,000 block in the order H
(the prior's Beta, scaled), then (ν, ρ, ξ₀) uniform, then ε; a run at n takes the first
n rows; torch's generator 109,000,000 + 10,000 p + 1,000 j + 10 q + m, p = 1 (A) or 2 (B),
j = 1, 2, 3 for the levels 0.1, 0.3, 0.5, m the member or 0 for the ratio estimator.

G3 (line 144): per calibrator, prior and level, on the 16 pilot surfaces at that level
whose reference (p5_g2_reference.py --g3-pilot) was accepted: the least-squares slope of
the calibrator's ξ₀ estimate on the exact posterior mean of ξ₀ under P above 0.95, and
the median of |Ĥ_net,P − Ĥ_exact,P| at 2n within 0.002 of that at n; training sets the
first 10,000 × 2^k of the draw, k = 0 to 4, every rung run; the largest n at which both
hold and the runs at n and 2n are within the budget (D81); the run at 2n made for the
check only (it is the next rung's run, except at the top); budget 320,000 surfaces and
4 h of wall per run, the whole calibrator; a rung whose run at n or 2n exceeds either
fails. Each (calibrator, prior, level) cell passes or fails on its own: a failed cell's
seeds are not trained, and Stage 2 loses that level for every measure needing its
network; P5 stops only if every cell of both calibrators fails (D81). The cost of scoring
both calibrators on the test set's 1,200 pairs (five seeds each) is projected from the
pilot predictions at each passing cell's n, against G2's budget of 1,344 core-hours; above
it, P5 stops (D81). The ratio estimator's sampler runs at the reference's lengths, which
G2's retry doubles if it is needed (line 75, D81). The test set is scored only after Stage 1
has been declared from its references (p5_stage2.py --stage1) and has not stopped P5; the
scores' header records that Stage 1 output's digest and the networks to be scored (D81).

From the command line the emulator is G̃, the build G1 passes (p5_g1_emulator.require_g1),
and an --emulator naming another file is refused; G3 and the training of seeds run only on
a complete G3-pilot record (all 48 pairs) made with G̃; a resumed G3 or training reuses
saved networks only when its record's header matches the call, the emulator included; the
report and the list of trained seeds are not written over; the scoring refuses when G̃,
the set's references, G3's report and (for the test set) Stage 1's output name different
emulators (D82).

Nothing here runs before G1 and G2 after freezing; the tests use toy emulators in
memory and skip without torch. The core stays torch-free (requirements-p5.txt). Run from
the command line, the script refuses numpy, scipy and torch other than the leaf's on the
author's machine (p5_g1_emulator.LEAF_VERSIONS and LEAF_TORCH) unless
--allow-other-versions, recorded (D81, D82).

Usage (from the repository root, in .venv-p5, after G1 and G2):
    py p5_g3_calibrators.py                 the design; holds
    py p5_g3_calibrators.py --g3            the ladder, the checks, output/p5_g3_report.json
    py p5_g3_calibrators.py --train         seeds 2 to 5 at the adequate n per cell, after G3
    py p5_g3_calibrators.py --score SET     the calibrators' outputs on a set (g3 | test)
    options: --emulator PATH (G̃ by default; another file is refused), --threads K (torch threads; default 4; recorded)
Outputs (output/ is git-ignored; force-added by name after a run):
    output/p5_g3_ladder.jsonl, output/p5_g3_report.json, output/p5_g3/<run>.pt and .json
    output/p5_calibrators.jsonl, output/p5_calibrators/<run>.pt and .json
    output/p5_scores_<set>.jsonl
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import sys
import json
import copy
import time
import argparse
import platform

import numpy as np

try:
    import torch
    _HAS_TORCH = True
except Exception:                                   # the core environment (or a broken install), as Layer 3 guards it
    torch = None
    _HAS_TORCH = False

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import p5_g0_pilot as P
import p5_g1_emulator as E
import p5_g2_reference as R
from p5_g0_pilot import git_state, utc_now, append_line, _rel, read_jsonl
from p5_g1_emulator import BOX, NOISE_LEVELS, N_QUOTES, box_arrays

# --------------------------------------------------------------------------- #
# The design, as written in (lines 72, 73, 75, 144)
# --------------------------------------------------------------------------- #
HIDDEN = (128, 128, 128)
N_MEMBERS = 5
LR, BETAS, BATCH = 1e-3, (0.9, 0.999), 256
PATIENCE, MAX_EPOCHS = 50, 1000
VAR_FLOOR = 1e-6
N_VALID = 4000
SEED_TRAIN, SEED_VALID, SEED_NRE, TORCH_BASE = 109, 113, 107, 109_000_000
DRAW_N = 320000
LADDER = (10000, 20000, 40000, 80000, 160000)
BUDGET_SURFACES, BUDGET_WALL_S = 320000, 4 * 3600.0
SLOPE_MIN, STABILITY_TOL = 0.95, 0.002
N_SEEDS, RETRAIN_SEED = 5, 6
PRIOR_INDEX = {"A": 1, "B": 2}
LEVEL_INDEX = {float(s): j + 1 for j, s in enumerate(NOISE_LEVELS)}   # j = 1, 2, 3 (line 75)
CALIBRATORS = ("ensemble", "nre")
THREADS_DEFAULT = 4
Z90 = 1.645
H_LO, H_W = R.H_LO, R.H_W
PRIOR_AB = R.PRIOR_AB

OUT_DIR = P.OUT_DIR
PATHS = {"ladder": os.path.join(OUT_DIR, "p5_g3_ladder.jsonl"), "report": os.path.join(OUT_DIR, "p5_g3_report.json"),
         "nets_g3": os.path.join(OUT_DIR, "p5_g3"), "train": os.path.join(OUT_DIR, "p5_calibrators.jsonl"),
         "nets": os.path.join(OUT_DIR, "p5_calibrators")}
EMULATOR_PATH = E.paths("base")["npz"]


def scores_path(set_name):
    return os.path.join(OUT_DIR, f"p5_scores_{set_name}.jsonl")


def require_torch():
    if not _HAS_TORCH:
        raise SystemExit("torch is not in this environment: use .venv-p5 (requirements-p5.txt); the core stays torch-free")


def torch_seed(p, j, q, m):
    """109,000,000 + 10,000 p + 1,000 j + 10 q + m (line 75)."""
    return TORCH_BASE + 10000 * int(p) + 1000 * int(j) + 10 * int(q) + int(m)


# --------------------------------------------------------------------------- #
# Draws (line 75) and surfaces
# --------------------------------------------------------------------------- #
def draw_block(seed_list, n, prior, box=BOX):
    """One generator: H = 0.02 + 0.46 Beta(a, b) of the prior, then (ν, ρ, ξ₀) uniform on
    the box, then ε standard normal (n, 35). Returns (theta [n, 4], eps [n, 35])."""
    rng = np.random.default_rng(seed_list)
    lo, hi = box_arrays(box)
    a, b = PRIOR_AB[prior]
    H = H_LO + H_W * rng.beta(a, b, size=n)
    other = rng.uniform(lo[1:], hi[1:], size=(n, 3))
    eps = rng.standard_normal((n, N_QUOTES))
    return np.column_stack([H, other]), eps


def training_block(prior, j, q, n_block=DRAW_N, box=BOX):
    """The 320,000 block of seed q under prior p and level j; a run at n takes its first n rows."""
    return draw_block([SEED_TRAIN, PRIOR_INDEX[prior], int(j), int(q)], n_block, prior, box)


def validation_block(prior, j, n=N_VALID, box=BOX):
    return draw_block([SEED_VALID, PRIOR_INDEX[prior], int(j)], n, prior, box)


def make_surfaces(theta, eps, s, em, chunk=8192):
    """y = G̃(θ) + s ε, the emulator evaluated in chunks."""
    Y = np.empty_like(eps)
    for a in range(0, theta.shape[0], chunk):
        Y[a:a + chunk] = em.evaluate_many(theta[a:a + chunk]) + s * eps[a:a + chunk]
    return Y


class Standardiser:
    """Per-quote mean and standard deviation over the run's training surfaces."""

    def __init__(self, Y):
        Y = np.asarray(Y, dtype=float)
        self.mean = Y.mean(axis=0)
        self.sd = np.maximum(Y.std(axis=0), 1e-12)

    def __call__(self, Y):
        return (np.asarray(Y, dtype=float) - self.mean) / self.sd

    def to_json(self):
        return {"mean": self.mean.tolist(), "sd": self.sd.tolist()}

    @classmethod
    def from_json(cls, d):
        obj = cls.__new__(cls)
        obj.mean, obj.sd = np.array(d["mean"], dtype=float), np.array(d["sd"], dtype=float)
        return obj


# --------------------------------------------------------------------------- #
# Models and losses (line 75)
# --------------------------------------------------------------------------- #
def make_mlp(in_dim, out_dim, hidden=HIDDEN):
    require_torch()
    layers, d = [], in_dim
    for h in hidden:
        layers += [torch.nn.Linear(d, h), torch.nn.ReLU()]
        d = h
    layers.append(torch.nn.Linear(d, out_dim))
    return torch.nn.Sequential(*layers)


def n_params(net):
    return int(sum(p.numel() for p in net.parameters()))


def member_loss(out, target):
    """Gaussian negative log-likelihood of the member's two outputs: out[:, :2] the means,
    out[:, 2:] the raw variances, softplus(raw) + 1e-6; summed over the two parameters,
    averaged over the batch. target [B, 2] in mapped coordinates."""
    mu = out[:, :2]
    var = torch.nn.functional.softplus(out[:, 2:]) + VAR_FLOOR
    nll = 0.5 * (torch.log(2.0 * np.pi * var) + (target - mu) ** 2 / var)
    return nll.sum(dim=1).mean()


def nre_loss(net, xb, tb):
    """Binary cross-entropy over the B joint pairs (label 1) and the B marginal pairs
    (θ shifted cyclically by one within the batch, label 0)."""
    joint = net(torch.cat([xb, tb], dim=1)).squeeze(1)
    marg = net(torch.cat([xb, torch.roll(tb, 1, dims=0)], dim=1)).squeeze(1)
    logits = torch.cat([joint, marg])
    labels = torch.cat([torch.ones_like(joint), torch.zeros_like(marg)])
    return torch.nn.functional.binary_cross_entropy_with_logits(logits, labels)


def _batches(n, batch=BATCH):
    """Slices of a shuffled or drawn order: full batches and the partial final one, kept."""
    return [slice(a, min(a + batch, n)) for a in range(0, n, batch)]


def train_net(kind, X, T, Xv, Tv, seed, deadline=None, patience=PATIENCE, max_epochs=MAX_EPOCHS, batch=BATCH,
              lr=LR, threads=THREADS_DEFAULT):
    """One network: 'member' (35 -> 4, Gaussian NLL on T [n, 2]) or 'nre' (39 -> 1, BCE with
    T [n, 4]). torch.manual_seed(seed) once, before construction; the shuffles
    (torch.randperm) continue that stream, which nothing else draws from; Adam; epochs of
    shuffled batches, the partial final batch kept; the validation objective in drawn
    order after every epoch; stop after `patience` epochs without improvement or at
    `max_epochs`; the best state kept. A non-finite training or validation loss -> status
    'nonfinite' (the epoch not counted, its loss not recorded); the wall past `deadline`
    (a perf_counter time, tested after each epoch's validation) -> 'over_budget'; else 'ok'."""
    require_torch()
    torch.set_num_threads(int(threads))
    torch.manual_seed(int(seed))
    net = make_mlp(X.shape[1], 4) if kind == "member" else make_mlp(X.shape[1] + 4, 1)
    opt = torch.optim.Adam(net.parameters(), lr=lr, betas=BETAS, weight_decay=0.0)
    Xt, Tt = torch.as_tensor(X, dtype=torch.float32), torch.as_tensor(T, dtype=torch.float32)
    Xvt, Tvt = torch.as_tensor(Xv, dtype=torch.float32), torch.as_tensor(Tv, dtype=torch.float32)
    loss_fn = (lambda xb, tb: member_loss(net(xb), tb)) if kind == "member" else (lambda xb, tb: nre_loss(net, xb, tb))
    n = Xt.shape[0]
    best, best_state, since, history, status = float("inf"), copy.deepcopy(net.state_dict()), 0, [], "ok"
    t0 = time.perf_counter()
    epochs = 0
    for epoch in range(1, max_epochs + 1):
        net.train()
        perm = torch.randperm(n)
        for sl in _batches(n, batch):
            idx = perm[sl]
            opt.zero_grad()
            loss = loss_fn(Xt[idx], Tt[idx])
            if not torch.isfinite(loss):
                status = "nonfinite"
                break
            loss.backward()
            opt.step()
        if status != "ok":
            break
        net.eval()
        with torch.no_grad():
            vals = [loss_fn(Xvt[sl], Tvt[sl]).item() * (sl.stop - sl.start) for sl in _batches(Xvt.shape[0], batch)]
        val = float(sum(vals) / Xvt.shape[0])
        if not np.isfinite(val):
            status = "nonfinite"
            break
        epochs = epoch
        history.append(val)
        if val < best:
            best, best_state, since = val, copy.deepcopy(net.state_dict()), 0
        else:
            since += 1
        if since >= patience:
            break
        if deadline is not None and time.perf_counter() > deadline:
            status = "over_budget"
            break
    net.load_state_dict(best_state)
    net.eval()
    return {"net": net, "status": status, "epochs": epochs, "best_val": best if np.isfinite(best) else None,
            "history": history, "seconds": time.perf_counter() - t0, "seed": int(seed), "kind": kind}


# --------------------------------------------------------------------------- #
# A calibrator run: train, save, load, predict
# --------------------------------------------------------------------------- #
def run_label(kind, prior, j, q, n):
    return f"{kind}_p{PRIOR_INDEX[prior]}_j{int(j)}_q{int(q)}_n{int(n)}"


def train_calibrator(kind, prior, j, q, n, theta, eps, s, em, val_theta, val_eps, threads=THREADS_DEFAULT,
                     wall_budget=BUDGET_WALL_S, patience=PATIENCE, max_epochs=MAX_EPOCHS, batch=BATCH):
    """The whole calibrator at training-set size n: the first n rows of the block, the
    standardiser from them, the members (seeds m = 1..5) or the one ratio estimator (m = 0),
    one wall budget for the whole run. Returns the run dict (nets, standardiser, status, ...)."""
    t0 = time.perf_counter()
    deadline = t0 + wall_budget
    th, ep = theta[:n], eps[:n]
    Y = make_surfaces(th, ep, s, em)
    std = Standardiser(Y)
    X = std(Y)
    Tu = R.theta_to_unit(th)
    Yv = make_surfaces(val_theta, val_eps, s, em)
    Xv, Tvu = std(Yv), R.theta_to_unit(val_theta)
    p, jj = PRIOR_INDEX[prior], int(j)
    members, status = [], "ok"
    if kind == "ensemble":
        for m in range(1, N_MEMBERS + 1):
            r = train_net("member", X, Tu[:, [0, 3]], Xv, Tvu[:, [0, 3]], torch_seed(p, jj, q, m), deadline, patience,
                          max_epochs, batch, threads=threads)
            members.append(r)
            if r["status"] != "ok":
                status = r["status"]
                break
    else:
        r = train_net("nre", X, Tu, Xv, Tvu, torch_seed(p, jj, q, 0), deadline, patience, max_epochs, batch,
                      threads=threads)
        members.append(r)
        status = r["status"]
    return {"kind": kind, "prior": prior, "j": jj, "q": int(q), "n": int(n), "s": float(s), "status": status,
            "nets": [m["net"] for m in members], "std": std, "threads": int(threads),
            "members": [{k: v for k, v in m.items() if k != "net"} for m in members],
            "seconds": time.perf_counter() - t0, "label": run_label(kind, prior, j, q, n)}


def save_run(dir_path, run):
    os.makedirs(dir_path, exist_ok=True)
    base = os.path.join(dir_path, run["label"])
    torch.save([net.state_dict() for net in run["nets"]], base + ".pt")
    meta = {k: v for k, v in run.items() if k not in ("nets", "std")}
    meta["std"] = run["std"].to_json()
    with open(base + ".json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=1, allow_nan=False)
    return base


def load_run(dir_path, label):
    require_torch()
    base = os.path.join(dir_path, label)
    with open(base + ".json", encoding="utf-8") as f:
        meta = json.load(f)
    states = torch.load(base + ".pt", map_location="cpu")
    nets = []
    for st in states:
        net = make_mlp(N_QUOTES, 4) if meta["kind"] == "ensemble" else make_mlp(N_QUOTES + 4, 1)
        net.load_state_dict(st)
        net.eval()
        nets.append(net)
    meta["nets"], meta["std"] = nets, Standardiser.from_json(meta["std"])
    return meta


def ensemble_predict(run, Y):
    """Per surface: Ĥ, the ξ₀ estimate, the two variance components of H in box units,
    the total and the 90% interval (line 75)."""
    X = torch.as_tensor(run["std"](Y), dtype=torch.float32)
    mus, vars_ = [], []
    with torch.no_grad():
        for net in run["nets"]:
            out = net(X)
            mus.append(out[:, :2].numpy())
            vars_.append(torch.nn.functional.softplus(out[:, 2:]).numpy() + VAR_FLOOR)   # as in training, overflow-safe
    mus, vars_ = np.stack(mus), np.stack(vars_)                           # [M, n, 2]
    lo, hi = box_arrays(BOX)
    scale = np.array([hi[0] - lo[0], hi[3] - lo[3]])
    mean = lo[[0, 3]] + mus.mean(axis=0) * scale                           # box units
    var_alea = vars_.mean(axis=0)[:, 0] * scale[0] ** 2
    var_epi = mus.var(axis=0, ddof=0)[:, 0] * scale[0] ** 2
    total = var_alea + var_epi
    return {"H_hat": mean[:, 0], "xi0_hat": mean[:, 1], "var_alea": var_alea, "var_epi": var_epi,
            "var_total": total, "lo": mean[:, 0] - Z90 * np.sqrt(total), "hi": mean[:, 0] + Z90 * np.sqrt(total)}


def nre_target(run, y):
    """log P(H) + logit(θ, y) as a callable of θ in box units, for the reference's sampler."""
    x = torch.as_tensor(run["std"](np.asarray(y, dtype=float).reshape(1, -1)), dtype=torch.float32)
    net, prior = run["nets"][0], run["prior"]

    def target(th):
        th = np.asarray(th, dtype=float).reshape(-1, 4)
        tu = torch.as_tensor(R.theta_to_unit(th), dtype=torch.float32)
        with torch.no_grad():
            logit = net(torch.cat([x.expand(tu.shape[0], -1), tu], dim=1)).squeeze(1).numpy().astype(float)
        return R.prior_logpdf(th[:, 0], prior) + logit
    return target


def nre_posterior(run, y, i, burn=R.BURN, sample=R.SAMPLE, rhat_max=R.RHAT_MAX, ess_min=R.ESS_MIN):
    """The ratio estimator's posterior for one pair by the reference's sampler (line 75):
    generator default_rng([107, p, q, i, j]); Ĥ and ξ₀ the pooled means, the equal-tailed
    90% interval; a failure after the repeat is the calibrator's failure on the pair."""
    rng = np.random.default_rng([SEED_NRE, PRIOR_INDEX[run["prior"]], int(run["q"]), int(i), int(run["j"])])
    t0 = time.perf_counter()
    chains, attempts = R._run_with_repeat(None, None, None, rng, "uniform", burn, sample, BOX, rhat_max, ess_min,
                                          target=nre_target(run, y))
    out = {"attempts": attempts, "seconds": time.perf_counter() - t0}
    if chains is None:
        return dict(out, status="failure", H_hat=None, xi0_hat=None, lo=None, hi=None, n_pooled=None)
    pooled = R.unit_to_theta(chains["samples"].reshape(-1, 4))
    H, xi0 = pooled[:, 0], pooled[:, 3]
    w = np.ones_like(H)
    return dict(out, status="ok", H_hat=float(H.mean()), xi0_hat=float(xi0.mean()), lo=R.wquantile(H, w, 0.05),
                hi=R.wquantile(H, w, 0.95), n_pooled=int(H.shape[0]))


# --------------------------------------------------------------------------- #
# G3 (line 144): the pilot surfaces, the checks, the ladder
# --------------------------------------------------------------------------- #
def pilot_surfaces(refs, prior, j):
    """The G3 pilot pairs at level j (0-based j_ref = j − 1 in the reference's records) whose
    reference was accepted: (i, θ, y, Ĥ_exact,P, mean ξ₀ under P). Failures are removed and
    counted (line 144)."""
    j_ref = int(j) - 1
    rows, removed = [], []
    for lab, rec in sorted(refs.items()):
        if rec.get("set") != "g3" or rec["j"] != j_ref:
            continue
        if rec["status"] != "ok":
            removed.append(lab)
            continue
        u = rec["under"][prior]
        rows.append({"i": rec["i"], "theta": rec["theta"], "y": np.asarray(rec["y"], dtype=float),
                     "H_exact": u["mean_H"], "xi0_exact": u["mean_xi0"]})
    return rows, removed


def g3_checks(pred_n, pred_2n, rows, slope_min=SLOPE_MIN, tol=STABILITY_TOL):
    """pred_*: per surface (H_hat, xi0_hat) or None (a failed pair, excluded from both
    checks at that rung). The slope of ξ₀_hat on the exact mean of ξ₀ at n; the medians of
    |d| at n and 2n; the two verdicts."""
    def table(pred):
        keep = [k for k, p in enumerate(pred) if p is not None]
        xe = np.array([rows[k]["xi0_exact"] for k in keep])
        xh = np.array([pred[k][1] for k in keep])
        d = np.array([abs(pred[k][0] - rows[k]["H_exact"]) for k in keep])
        return keep, xe, xh, d
    keep_n, xe, xh, d_n = table(pred_n)
    keep_2, _, _, d_2n = table(pred_2n)
    slope = float(np.polyfit(xe, xh, 1)[0]) if len(keep_n) >= 2 and np.ptp(xe) > 0 else None
    med_n = float(np.median(d_n)) if len(d_n) else None
    med_2n = float(np.median(d_2n)) if len(d_2n) else None
    stable = (med_n is not None and med_2n is not None and abs(med_2n - med_n) <= tol)
    return {"slope": slope, "slope_ok": bool(slope is not None and slope > slope_min), "median_abs_d_n": med_n,
            "median_abs_d_2n": med_2n, "stable": bool(stable), "n_used_n": len(keep_n), "n_used_2n": len(keep_2),
            "passed": bool(slope is not None and slope > slope_min and stable)}


def budget_ok(run, n, budget_surfaces=BUDGET_SURFACES, wall=BUDGET_WALL_S):
    return bool(n <= budget_surfaces and run["status"] == "ok" and run["seconds"] <= wall)


def predict_rows(run, rows, burn=R.BURN, sample=R.SAMPLE, rhat_max=R.RHAT_MAX, ess_min=R.ESS_MIN):
    """(H_hat, xi0_hat) per pilot surface, or None where the ratio estimator failed; the
    failures; and the cost of the predictions (the seconds, and for the ratio estimator each
    posterior's sampler runs), from which the scoring cost is projected (D81)."""
    t0 = time.perf_counter()
    if not rows:                                            # no accepted pilot reference at the level: the cell fails
        return [], [], {"seconds": 0.0, "n": 0, "attempts": None if run["kind"] == "ensemble" else [],
                        **({} if run["kind"] == "ensemble" else {"burn": int(burn), "sample": int(sample)})}
    if run["kind"] == "ensemble":
        pr = ensemble_predict(run, np.stack([r["y"] for r in rows]))
        cost = {"seconds": time.perf_counter() - t0, "n": len(rows), "attempts": None}
        return [(float(pr["H_hat"][k]), float(pr["xi0_hat"][k])) for k in range(len(rows))], [], cost
    preds, fails, attempts = [], [], []
    for r in rows:
        po = nre_posterior(run, r["y"], r["i"], burn, sample, rhat_max, ess_min)
        attempts.append(po["attempts"])
        if po["status"] != "ok":
            preds.append(None)
            fails.append(r["i"])
        else:
            preds.append((po["H_hat"], po["xi0_hat"]))
    return preds, fails, {"seconds": time.perf_counter() - t0, "n": len(rows), "attempts": attempts,
                          "burn": int(burn), "sample": int(sample)}


def scoring_projection(cells, threads, n_test=R.N_TEST, n_seeds=N_SEEDS, budget_core_h=R.BUDGET_CORE_H):
    """The projected cost of scoring both calibrators on the test set (line 143, D81): per
    passing cell, its n_test pairs at each of n_seeds seeds, from the pilot predictions of
    its network at n*: the ratio estimator's posteriors by p5_g2_reference.project_cost
    (the median seconds per run of the adopted lengths times the mean runs per posterior,
    repeats counted), the ensemble's by its seconds per surface; each at `threads` cores,
    the torch threads a scoring process holds. In core-hours, against the budget. A passing
    cell whose cost cannot be projected leaves the projection incomplete, and an incomplete
    projection is no go (D81)."""
    per_cell, total, missing = {}, 0.0, []
    for key, c in cells.items():
        if not c.get("passed"):
            continue
        cost = c.get("cost_at_n_star") or {}
        n_items = int(n_test) * int(n_seeds)
        if c["calibrator"] == "nre":
            p = R.project_cost(cost.get("attempts") or [], n_items, int(cost.get("burn", R.BURN)) + int(cost.get("sample", R.SAMPLE)),
                               cores=threads)
        else:
            sec = cost.get("seconds")
            per = sec / cost["n"] if sec is not None and cost.get("n") else None
            p = {"seconds_per_surface": per, "n_items": n_items, "cores": float(threads),
                 "projected_core_hours": n_items * per * float(threads) / 3600.0 if per is not None else None}
        per_cell[key] = p
        if p["projected_core_hours"] is None:
            missing.append(key)
        else:
            total += p["projected_core_hours"]
    return {"per_cell": per_cell, "projected_core_hours": total, "budget_core_hours": budget_core_h,
            "complete": not missing, "cells_not_projected": missing, "go": bool(not missing and total <= budget_core_h),
            "n_test": int(n_test), "n_seeds": int(n_seeds), "threads": int(threads)}


def _override_line(path, versions, mode):
    """A call under --allow-other-versions is recorded in the record it writes to, resumed or
    new, as well as in a new record's header (D81)."""
    if versions and versions.get("override"):
        append_line(path, {"kind": "launch", "mode": mode, "utc": utc_now(), "versions": versions})


def _check_resume(hdr, current, keys, path):
    """A resumed run reuses saved networks only when its record was begun with the same
    settings `keys` and the same emulator as this call (D82), as G1's launch checks its own."""
    pairs = [(k, hdr.get(k), current.get(k)) for k in keys]
    pairs.append(("emulator sha256", (hdr.get("emulator") or {}).get("sha256"), (current.get("emulator") or {}).get("sha256")))
    for k, was, now in pairs:
        if was != now:
            raise SystemExit(f"resume refused: {_rel(path)} was begun with {k}={was!r}, this call has {now!r} (D82)")


def require_pilot(hdr, tasks, emulator_sha=None, n_points=R.N_G3_POINTS, n_levels=len(NOISE_LEVELS)):
    """G3 and the training of seeds run only on a complete G3-pilot record (line 144, D82):
    all 48 pairs (16 surfaces at 3 levels) recorded, accepted or failed, and, when
    `emulator_sha` is given, the record made with that emulator (its header's SHA-256)."""
    want = {f"g3_{i:04d}_{j}" for i in range(int(n_points)) for j in range(int(n_levels))}
    have = {lab for lab, r in tasks.items() if r.get("set") == "g3"} & want
    if hdr is None or len(have) < len(want):
        missing = sorted(want - have)
        raise SystemExit(f"G3 refused: the G3-pilot record holds {len(have)} of its {len(want)} pairs (missing "
                         f"{', '.join(missing[:3])}{' and more' if len(missing) > 3 else ''}); G3 and the seeds run "
                         "only once every pair is recorded, accepted or failed (line 144, D82)")
    if emulator_sha is not None and hdr.get("emulator_sha256") != emulator_sha:
        raise SystemExit("G3 refused: the G3-pilot record was made with another emulator than G̃ (line 142, D82)")
    return True


def ladder_record(run, extra=None):
    commit, clean = git_state(os.path.dirname(os.path.abspath(__file__)))
    rec = {"kind": "task", "record": "run", "task": run["label"], "calibrator": run["kind"], "prior": run["prior"], "j": run["j"],
           "q": run["q"], "n": run["n"], "s": run["s"], "status": run["status"], "seconds": run["seconds"],
           "threads": run["threads"], "members": run["members"], "commit": commit, "clean_tree": clean, "utc": utc_now()}
    rec.update(extra or {})
    return rec


def get_or_train(kind, prior, j, q, n, theta, eps, s, em, val, jsonl_path, nets_dir, threads, train_kw):
    """The run from the record and the saved networks if present, else trained and saved."""
    label = run_label(kind, prior, j, q, n)
    hdr, launches, tasks = read_jsonl(jsonl_path)
    if label in tasks and all(os.path.exists(os.path.join(nets_dir, label + ext)) for ext in (".pt", ".json")):
        run = load_run(nets_dir, label)
        run["label"] = label
        return run, False
    run = train_calibrator(kind, prior, j, q, n, theta, eps, s, em, val[0], val[1], threads, **train_kw)
    save_run(nets_dir, run)
    append_line(jsonl_path, ladder_record(run))
    return run, True


def run_g3(em, refs, jsonl_path=None, nets_dir=None, report_path=None, threads=THREADS_DEFAULT, ladder=LADDER,
           n_block=DRAW_N, n_valid=N_VALID, levels=NOISE_LEVELS, priors=("A", "B"), calibrators=CALIBRATORS,
           train_kw=None, post_kw=None, budget_surfaces=BUDGET_SURFACES, wall=BUDGET_WALL_S, versions=None,
           budget_core_h=R.BUDGET_CORE_H, emulator=None):
    """The ladder of line 144 for every (calibrator, prior, level): the runs at n_k and 2n_k
    (the latter the next rung's run, except at the top), the two checks on the pilot
    surfaces, the budgets, every rung run, the largest passing n (D81). Each cell passes or
    fails on its own; `passed` is every cell passing, `p5_stops` every cell failing or the
    scoring cost projected above the budget (D81). Torch runs at `threads` threads, for the
    predictions as for the training, on a fresh run as on a resumed one. Resumable: a run in
    the record with its networks saved is reused, once the record's header matches this
    call's (the ladder, the draw, the validation size, the sampler's lengths, the emulator);
    an existing report is not written over, the refusal coming before any training (D82).
    `emulator` is G̃'s {path, sha256} (p5_g1_emulator.emulator_info), recorded. Writes the
    report; returns it."""
    jsonl_path, nets_dir, report_path = jsonl_path or PATHS["ladder"], nets_dir or PATHS["nets_g3"], report_path or PATHS["report"]
    E.refuse_overwrite([report_path], "G3")
    train_kw, post_kw = dict(train_kw or {}), dict(post_kw or {})
    train_kw.setdefault("wall_budget", wall)
    if _HAS_TORCH:
        torch.set_num_threads(int(threads))
    v = versions or E.version_check(E.LEAF_VERSIONS, check=False)
    current = header("g3", threads, {"ladder": list(ladder), "n_block": n_block, "n_valid": n_valid, "post_kw": post_kw,
                                     "emulator": emulator}, v)
    hdr, launches, tasks = read_jsonl(jsonl_path)
    if hdr is None:
        append_line(jsonl_path, current)
    else:
        _check_resume(hdr, json.loads(json.dumps(current)), ("ladder", "n_block", "n_valid", "post_kw"), jsonl_path)
    _override_line(jsonl_path, v, "g3")
    cells, stops = {}, []
    for prior in priors:
        for s in levels:
            j = LEVEL_INDEX[float(s)]
            rows, removed = pilot_surfaces(refs, prior, j)
            theta, eps = training_block(prior, j, 1, n_block)
            val = validation_block(prior, j, n_valid)
            for kind in calibrators:
                cell = {"calibrator": kind, "prior": prior, "j": j, "s": float(s), "n_pilot": len(rows),
                        "removed_references": removed, "rungs": [], "n_star": None, "passed": False,
                        "cost_at_n_star": None}
                runs, preds = {}, {}

                def predicted(nn):                                  # each network's predictions made once
                    if nn not in preds:
                        preds[nn] = predict_rows(runs[nn], rows, **post_kw)
                    return preds[nn]
                for k, n in enumerate(ladder):
                    n2 = 2 * n
                    for nn in (n, n2):
                        if nn not in runs:
                            runs[nn], fresh = get_or_train(kind, prior, j, 1, nn, theta, eps, s, em, val, jsonl_path,
                                                           nets_dir, threads, train_kw)
                            print(f"  {runs[nn]['label']}: {runs[nn]['status']} {runs[nn]['seconds']:.0f}s "
                                  f"{'trained' if fresh else 'reused'}", flush=True)
                    rung = {"k": k, "n": n, "n_check": n2,
                            "budget_n": budget_ok(runs[n], n, budget_surfaces, wall),
                            "budget_2n": budget_ok(runs[n2], n2, budget_surfaces, wall),
                            "status_n": runs[n]["status"], "status_2n": runs[n2]["status"],
                            "seconds_n": runs[n]["seconds"], "seconds_2n": runs[n2]["seconds"]}
                    if rung["budget_n"] and rung["budget_2n"]:
                        pred_n, fails_n, cost_n = predicted(n)
                        pred_2, fails_2, _ = predicted(n2)
                        rung.update(g3_checks(pred_n, pred_2, rows), failures_n=fails_n, failures_2n=fails_2,
                                    cost_n=cost_n)
                    else:
                        rung.update(passed=False, reason=f"budget: run at n {runs[n]['status']} {runs[n]['seconds']:.0f}s, "
                                                         f"at 2n {runs[n2]['status']} {runs[n2]['seconds']:.0f}s")
                    cell["rungs"].append(rung)
                    append_line(jsonl_path, {"kind": "rung", "calibrator": kind, "prior": prior, "j": j, **rung, "utc": utc_now()})
                    print(f"  G3 {kind} {prior} j={j} n={n}: " + ("passed" if rung["passed"] else "not passed")
                          + f" (slope={rung.get('slope')}, |d| {rung.get('median_abs_d_n')} -> {rung.get('median_abs_d_2n')})",
                          flush=True)
                passing = [r for r in cell["rungs"] if r["passed"]]
                if passing:                                         # the largest n at which both hold (D81)
                    top = max(passing, key=lambda r: r["n"])
                    cell["n_star"], cell["passed"], cell["cost_at_n_star"] = top["n"], True, top.get("cost_n")
                else:
                    stops.append(f"{kind}_{prior}_j{j}")
                cells[f"{kind}_{prior}_j{j}"] = cell
    proj = scoring_projection(cells, threads, budget_core_h=budget_core_h)
    passing_cells = sorted(k for k, c in cells.items() if c["passed"])
    commit, clean = git_state(os.path.dirname(os.path.abspath(__file__)))
    report = {"gate": "G3", "passed": not stops, "cells_not_passing": stops, "cells_passing": passing_cells,
              "p5_stops": bool(not passing_cells or not proj["go"]),
              "p5_stops_reason": ("every cell of both calibrators failed (line 144)" if not passing_cells else
                                  "the scoring cost could not be projected for a passing cell (line 143)"
                                  if not proj["complete"] else
                                  "the projected cost of scoring the test set is above the budget (line 143)"
                                  if not proj["go"] else None),
              "scoring_projection": proj, "cells": cells,
              "ladder": list(ladder), "budget_surfaces": budget_surfaces, "budget_wall_s": wall, "slope_min": SLOPE_MIN,
              "stability_tol": STABILITY_TOL, "threads": int(threads), "post_kw": post_kw, "commit": commit,
              "clean_tree": clean, "versions": v, "emulator": emulator, "utc": utc_now(),
              "note": "passed: every cell passed; p5_stops: no cell passed, or the scoring cost is over the budget; "
                      "a failed cell's seeds are not trained and Stage 2 loses that level for every measure needing "
                      "its network (D81)"}
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=1, allow_nan=False)
    return report


# --------------------------------------------------------------------------- #
# After G3: the seeds (line 75) and the scoring of a set
# --------------------------------------------------------------------------- #
def header(mode, threads, extra=None, versions=None):
    commit, clean = git_state(os.path.dirname(os.path.abspath(__file__)))
    v = versions or E.version_check(E.LEAF_VERSIONS, check=False)
    return dict({"kind": "header", "script": os.path.basename(__file__), "mode": mode, "commit": commit,
                 "clean_tree": clean, "hidden": list(HIDDEN), "n_members": N_MEMBERS, "lr": LR, "betas": list(BETAS),
                 "batch": BATCH, "patience": PATIENCE, "max_epochs": MAX_EPOCHS, "var_floor": VAR_FLOOR,
                 "n_valid": N_VALID, "seeds": {"train": SEED_TRAIN, "valid": SEED_VALID, "nre": SEED_NRE,
                                               "torch_base": TORCH_BASE},
                 "ladder": list(LADDER), "draw_n": DRAW_N, "budget_surfaces": BUDGET_SURFACES,
                 "budget_wall_s": BUDGET_WALL_S, "slope_min": SLOPE_MIN, "stability_tol": STABILITY_TOL,
                 "n_seeds": N_SEEDS, "priors": {k: list(v) for k, v in PRIOR_AB.items()},
                 "noise_levels": list(NOISE_LEVELS), "threads": int(threads),
                 "torch": torch.__version__ if _HAS_TORCH else None, "python": platform.python_version(),
                 "numpy": v["numpy"], "scipy": v["scipy"], "versions": v, "platform": platform.platform(),
                 "cpu_count": os.cpu_count(), "utc_created": utc_now()}, **(extra or {}))


def train_seeds(em, report, jsonl_path=None, nets_dir=None, nets_g3=None, threads=THREADS_DEFAULT, n_block=DRAW_N,
                n_valid=N_VALID, seeds=range(2, N_SEEDS + 1), train_kw=None, versions=None, emulator=None):
    """Seeds q = 2..5 at n* per passing cell (q = 1's run is the ladder's), without the wall
    budget of G3 (line 144's budget is the gate's); a cell that failed G3 is not trained
    (D81). A non-finite loss is retrained once per cell from seed value 6, which takes the
    failed seed's place; every later failure in the cell leaves its slot empty, so that a
    second failure leaves the cell with four seeds and a third with three (line 75, D82).
    A resumed training reuses saved networks only when the record's header matches this
    call's (the passing cells, the emulator) (D82). Returns {cell: {n_star, seeds: {slot q:
    run label}, empty_slots}} for the passing cells."""
    jsonl_path, nets_dir, nets_g3 = jsonl_path or PATHS["train"], nets_dir or PATHS["nets"], nets_g3 or PATHS["nets_g3"]
    train_kw = dict(train_kw or {})
    train_kw.setdefault("wall_budget", float("inf"))
    passing = [k for k, c in report.get("cells", {}).items() if c.get("passed")]
    if not passing:
        raise SystemExit("G3 has not passed in any cell: P5 stops and no seeds are trained (line 144)")
    if report.get("p5_stops"):
        raise SystemExit(f"G3 has not passed with a go: P5 stops ({report.get('p5_stops_reason')}); no seeds are trained")
    current = header("train", threads, {"cells_passing": passing, "emulator": emulator}, versions)
    hdr, launches, tasks = read_jsonl(jsonl_path)
    if hdr is None:
        append_line(jsonl_path, current)
    else:
        _check_resume(hdr, json.loads(json.dumps(current)), ("cells_passing",), jsonl_path)
    _override_line(jsonl_path, versions, "train")
    out = {}
    for key in passing:
        cell = report["cells"][key]
        kind, prior, j, n = cell["calibrator"], cell["prior"], cell["j"], cell["n_star"]
        s = cell["s"]
        val = validation_block(prior, j, n_valid)
        done, empty, retrained = {1: run_label(kind, prior, j, 1, n)}, [], False
        for q in seeds:
            theta, eps = training_block(prior, j, q, n_block)
            run, fresh = get_or_train(kind, prior, j, q, n, theta, eps, s, em, val, jsonl_path, nets_dir, threads, train_kw)
            if run["status"] == "ok":
                done[q] = run["label"]
                continue
            if retrained:                                           # the one retraining is spent: four seeds
                empty.append(q)
                append_line(jsonl_path, {"kind": "retrain", "cell": key, "failed_q": q, "replacement_q": None,
                                         "status": run["status"], "note": "second failure; the slot stays empty (line 75)",
                                         "utc": utc_now()})
                continue
            retrained = True
            theta6, eps6 = training_block(prior, j, RETRAIN_SEED, n_block)
            run6, _ = get_or_train(kind, prior, j, RETRAIN_SEED, n, theta6, eps6, s, em, val, jsonl_path, nets_dir,
                                   threads, train_kw)
            append_line(jsonl_path, {"kind": "retrain", "cell": key, "failed_q": q, "replacement_q": RETRAIN_SEED,
                                     "status": run6["status"], "utc": utc_now()})
            if run6["status"] == "ok":
                done[q] = run6["label"]
            else:
                empty.append(q)
        out[key] = {"n_star": n, "seeds": done, "empty_slots": empty}
    return out


def scored_networks(trained):
    """The networks a scoring run covers, [calibrator, prior, level j, slot q], from the
    seeds trained for the passing cells (an empty slot is not listed); the scores' header
    carries the list and Stage 2 checks the scoring against it (D81)."""
    return sorted([key.split("_")[0], key.split("_")[1], int(key.split("_j")[1]), int(q)]
                  for key, cell in trained.items() for q in cell["seeds"])


def score_set(set_name, refs, em, trained, nets_dir=None, nets_g3=None, out_path=None, threads=THREADS_DEFAULT,
              post_kw=None, versions=None, refs_path=None, stage1_dir=None, emulator=None, hashes=None):
    """Every calibrator at every (prior, level, seed) on the set's surfaces at its level:
    per pair the outputs of line 75; `q` is the seed's slot (a seed-6 replacement carries
    q6 in its run label). Resumable by (run label, pair label). The test set is scored only
    after Stage 1 has been declared from its references and has not stopped P5 (line 101,
    D81): the header records the Stage 1 output's digest of the references, checked on
    resume and by Stage 2; it also lists the networks to be scored, every passing cell's
    seeds, against which Stage 2 checks that the scoring is complete (D81). `emulator` is
    G̃'s {path, sha256}, recorded in the header; `hashes` ({source: SHA-256}: G̃, the set's
    references, G3's report), with Stage 1's for the test set, must all agree, else the
    scoring is refused before any network is loaded (line 142, D82)."""
    out_path, post_kw = out_path or scores_path(set_name), dict(post_kw or {})
    stage1 = None
    if set_name == "test":
        import p5_stage2 as S2
        stage1 = S2.require_stage1(refs_path or R.SETS["test"]["jsonl"], stage1_dir)
    hdr, launches, done = read_jsonl(out_path)
    if stage1 is not None and hdr is not None and hdr.get("stage1_refs_digest") != stage1["refs_digest"]:
        raise SystemExit(f"scoring refused: {_rel(out_path)} was begun after another Stage 1 output (line 101)")
    if hashes is not None or stage1 is not None:
        named = dict(hashes or {})
        if stage1 is not None:
            named["Stage 1 output"] = (stage1.get("emulator") or {}).get("sha256")
        E.require_same_emulator(named, f"the scoring of the set {set_name}")
    require_torch()
    torch.set_num_threads(int(threads))
    nets_dir, nets_g3 = nets_dir or PATHS["nets"], nets_g3 or PATHS["nets_g3"]
    expected = scored_networks(trained)
    if hdr is None:
        append_line(out_path, header("score", threads, {"set": set_name, "post_kw": post_kw, "expected": expected,
                                                        "stage1_refs_digest": stage1["refs_digest"] if stage1 else None,
                                                        "emulator": emulator}, versions))
    _override_line(out_path, versions, "score")
    n_new = 0
    for key, cell in trained.items():
        kind, prior, j = key.split("_")[0], key.split("_")[1], int(key.split("_j")[1])
        rows = [r for lab, r in sorted(refs.items()) if r.get("set") == set_name and r["j"] == j - 1]
        for q, label in sorted(cell["seeds"].items(), key=lambda kv: int(kv[0])):
            run = load_run(nets_g3 if int(q) == 1 else nets_dir, label)
            run["label"] = label
            todo = [r for r in rows if f"{label}|{r['task']}" not in done]
            if not todo:
                continue
            if kind == "ensemble":
                pr = ensemble_predict(run, np.stack([np.asarray(r["y"]) for r in todo]))
                for k, r in enumerate(todo):
                    append_line(out_path, {"kind": "task", "task": f"{label}|{r['task']}", "run": label, "pair": r["task"],
                                           "calibrator": kind, "prior": prior, "j": j, "q": int(q), "i": r["i"],
                                           "status": "ok", "H_hat": float(pr["H_hat"][k]), "xi0_hat": float(pr["xi0_hat"][k]),
                                           "lo": float(pr["lo"][k]), "hi": float(pr["hi"][k]),
                                           "var_alea": float(pr["var_alea"][k]), "var_epi": float(pr["var_epi"][k]),
                                           "utc": utc_now()})
                    n_new += 1
            else:
                for r in todo:
                    po = nre_posterior(run, np.asarray(r["y"]), r["i"], **post_kw)
                    append_line(out_path, {"kind": "task", "task": f"{label}|{r['task']}", "run": label, "pair": r["task"],
                                           "calibrator": kind, "prior": prior, "j": j, "q": int(q), "i": r["i"],
                                           **{k2: v for k2, v in po.items()}, "utc": utc_now()})
                    n_new += 1
    return n_new


# --------------------------------------------------------------------------- #
def main(argv=None, check_versions=False):
    ap = argparse.ArgumentParser(description="P5 gate G3 calibrators (see module docstring)")
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--g3", action="store_true", help="the ladder of line 144 and the report")
    mode.add_argument("--train", action="store_true", help="seeds 2 to 5 at the adequate n per passing cell, after G3")
    mode.add_argument("--score", default=None, choices=("g3", "test"), help="the calibrators' outputs on a set")
    ap.add_argument("--emulator", default=None,
                    help="G̃ is the build G1 passes (p5_g1_emulator.require_g1); another file is refused (D82)")
    ap.add_argument("--threads", type=int, default=THREADS_DEFAULT, help="torch threads for training and scoring; recorded")
    ap.add_argument("--allow-other-versions", action="store_true",
                    help="run with numpy, scipy or torch other than p5_g1_emulator.LEAF_VERSIONS and LEAF_TORCH; "
                         "recorded (D81, D82)")
    a = ap.parse_args(argv)
    vers = E.version_check(E.LEAF_VERSIONS, a.allow_other_versions, os.path.basename(__file__), check=check_versions,
                           torch_expected=E.LEAF_TORCH)
    print("=== P5 G3 calibrators ===", flush=True)
    print(f"  trunk {HIDDEN} ReLU; members {N_MEMBERS}; Adam lr={LR} betas={BETAS}; batch {BATCH}; patience {PATIENCE}; "
          f"cap {MAX_EPOCHS}; validation {N_VALID}; seeds train {SEED_TRAIN} valid {SEED_VALID} nre {SEED_NRE}", flush=True)
    print(f"  G3: ladder {LADDER} of a {DRAW_N} draw; slope > {SLOPE_MIN}; |d| stable within {STABILITY_TOL}; "
          f"budget {BUDGET_SURFACES} surfaces, {BUDGET_WALL_S / 3600:.0f} h wall per run; torch: "
          f"{torch.__version__ if _HAS_TORCH else 'absent'}", flush=True)
    if not (a.g3 or a.train or a.score):
        print("\nHOLD: nothing trained. --g3 needs the emulator (G1) and the G3 pilot references (G2), after freezing.",
              flush=True)
        return
    require_torch()
    em_path = E.emulator_for_run(a.emulator)                    # G̃, the build G1 passes (line 142, D82)
    em, info = E.load_emulator(em_path), E.emulator_info(em_path)
    refs_path = R.SETS["g3"]["jsonl"]
    if a.g3 or a.score == "g3" or a.train:
        if not os.path.exists(refs_path):
            raise SystemExit(f"no G3 pilot references at {_rel(refs_path)}: run p5_g2_reference.py --g3-pilot first")
        p_hdr, _, p_refs = read_jsonl(refs_path)
        require_pilot(p_hdr, p_refs, info["sha256"])           # all 48 pairs, made with G̃ (D82)
    st = R.require_g2()                                         # the reference's lengths, for the ratio estimator (D81)
    post_kw = {"burn": st["burn"], "sample": st["sample"]}
    if a.g3:
        rep = run_g3(em, p_refs, threads=a.threads, post_kw=post_kw, versions=vers, emulator=info)
        print(f"=== G3: cells passing {rep['cells_passing']}; not passing {rep['cells_not_passing']}; scoring projected at "
              f"{rep['scoring_projection']['projected_core_hours']:.1f} core-hours"
              f"{'' if rep['scoring_projection']['complete'] else ' (incomplete)'}; "
              + ("P5 stops: " + str(rep["p5_stops_reason"]) if rep["p5_stops"] else "P5 goes on with the passing cells")
              + " ===", flush=True)
        for key, c in rep["cells"].items():
            print(f"  {key}: n* = {c['n_star']} ({len(c['rungs'])} rungs; {len(c['removed_references'])} references removed)",
                  flush=True)
        print(f"  written: {_rel(PATHS['report'])}", flush=True)
        return
    with open(PATHS["report"], encoding="utf-8") as f:
        rep = json.load(f)
    g3_sha = (rep.get("emulator") or {}).get("sha256")
    trained_path = os.path.join(OUT_DIR, "p5_calibrators_trained.json")
    if a.train:
        E.refuse_overwrite([trained_path], "seed training")
        E.require_same_emulator({"G̃": info["sha256"], "G3 report": g3_sha}, "seed training")
        trained = train_seeds(em, rep, threads=a.threads, versions=vers, emulator=info)
        with open(trained_path, "w", encoding="utf-8") as f:
            json.dump(trained, f, indent=1)
        print(f"  trained seeds per cell written: {_rel(trained_path)}", flush=True)
        return
    if a.score:
        with open(trained_path, encoding="utf-8") as f:
            trained = json.load(f)
        set_refs = R.SETS[a.score]["jsonl"]
        if not os.path.exists(set_refs):
            raise SystemExit(f"no references for the set {a.score} at {_rel(set_refs)}")
        s_hdr, _, refs = read_jsonl(set_refs)
        hashes = {"G̃": info["sha256"], "G3 report": g3_sha, f"{a.score} references": (s_hdr or {}).get("emulator_sha256")}
        n = score_set(a.score, refs, em, trained, threads=a.threads, post_kw=post_kw, versions=vers, emulator=info,
                      hashes=hashes)
        print(f"  scored {n} new (run, pair) results -> {_rel(scores_path(a.score))}", flush=True)


if __name__ == "__main__":
    main(check_versions=True)
