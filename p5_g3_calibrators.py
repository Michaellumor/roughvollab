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
first 10,000 × 2^k of the draw, k = 0 to 4; the smallest n at which both hold; the run
at 2n made for the check only (it is the next rung's run, except at the top); budget
320,000 surfaces and 4 h of wall per run, the whole calibrator; a rung whose run at n or
2n exceeds either fails; none passing stops P5.

Nothing here runs before G1 and G2 after freezing; the tests use toy emulators in
memory and skip without torch. The core stays torch-free (requirements-p5.txt).

Usage (from the repository root, in .venv-p5, after G1 and G2):
    py p5_g3_calibrators.py                 the design; holds
    py p5_g3_calibrators.py --g3            the ladder, the checks, output/p5_g3_report.json
    py p5_g3_calibrators.py --train         seeds 2 to 5 at the adequate n per cell, after G3
    py p5_g3_calibrators.py --score SET     the calibrators' outputs on a set (g3 | test)
    options: --emulator PATH, --threads K (torch threads for training and scoring; default 4; recorded)
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
    """(H_hat, xi0_hat) per pilot surface, or None where the ratio estimator failed."""
    if run["kind"] == "ensemble":
        pr = ensemble_predict(run, np.stack([r["y"] for r in rows]))
        return [(float(pr["H_hat"][k]), float(pr["xi0_hat"][k])) for k in range(len(rows))], []
    preds, fails = [], []
    for r in rows:
        po = nre_posterior(run, r["y"], r["i"], burn, sample, rhat_max, ess_min)
        if po["status"] != "ok":
            preds.append(None)
            fails.append(r["i"])
        else:
            preds.append((po["H_hat"], po["xi0_hat"]))
    return preds, fails


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
           train_kw=None, post_kw=None, budget_surfaces=BUDGET_SURFACES, wall=BUDGET_WALL_S):
    """The ladder of line 144 for every (calibrator, prior, level): the runs at n_k and 2n_k
    (the latter the next rung's run, except at the top), the two checks on the pilot
    surfaces, the budgets, the smallest passing n. Resumable: a run in the record with its
    networks saved is reused. Writes the report; returns it."""
    jsonl_path, nets_dir, report_path = jsonl_path or PATHS["ladder"], nets_dir or PATHS["nets_g3"], report_path or PATHS["report"]
    train_kw, post_kw = dict(train_kw or {}), dict(post_kw or {})
    train_kw.setdefault("wall_budget", wall)
    hdr, launches, tasks = read_jsonl(jsonl_path)
    if hdr is None:
        append_line(jsonl_path, header("g3", threads, {"ladder": list(ladder), "n_block": n_block, "n_valid": n_valid}))
    cells, stops = {}, []
    for prior in priors:
        for s in levels:
            j = LEVEL_INDEX[float(s)]
            rows, removed = pilot_surfaces(refs, prior, j)
            theta, eps = training_block(prior, j, 1, n_block)
            val = validation_block(prior, j, n_valid)
            for kind in calibrators:
                cell = {"calibrator": kind, "prior": prior, "j": j, "s": float(s), "n_pilot": len(rows),
                        "removed_references": removed, "rungs": [], "n_star": None, "passed": False}
                runs = {}
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
                        pred_n, fails_n = predict_rows(runs[n], rows, **post_kw)
                        pred_2, fails_2 = predict_rows(runs[n2], rows, **post_kw)
                        rung.update(g3_checks(pred_n, pred_2, rows), failures_n=fails_n, failures_2n=fails_2)
                    else:
                        rung.update(passed=False, reason=f"budget: run at n {runs[n]['status']} {runs[n]['seconds']:.0f}s, "
                                                         f"at 2n {runs[n2]['status']} {runs[n2]['seconds']:.0f}s")
                    cell["rungs"].append(rung)
                    append_line(jsonl_path, {"kind": "rung", "calibrator": kind, "prior": prior, "j": j, **rung, "utc": utc_now()})
                    print(f"  G3 {kind} {prior} j={j} n={n}: " + ("passed" if rung["passed"] else "not passed")
                          + f" (slope={rung.get('slope')}, |d| {rung.get('median_abs_d_n')} -> {rung.get('median_abs_d_2n')})",
                          flush=True)
                    if rung["passed"]:
                        cell["n_star"], cell["passed"] = n, True
                        break
                if not cell["passed"]:
                    stops.append(f"{kind} {prior} j={j}")
                cells[f"{kind}_{prior}_j{j}"] = cell
    commit, clean = git_state(os.path.dirname(os.path.abspath(__file__)))
    report = {"gate": "G3", "passed": not stops, "p5_stops": bool(stops), "cells_not_passing": stops, "cells": cells,
              "ladder": list(ladder), "budget_surfaces": budget_surfaces, "budget_wall_s": wall, "slope_min": SLOPE_MIN,
              "stability_tol": STABILITY_TOL, "threads": int(threads), "commit": commit, "clean_tree": clean,
              "utc": utc_now()}
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=1, allow_nan=False)
    return report


# --------------------------------------------------------------------------- #
# After G3: the seeds (line 75) and the scoring of a set
# --------------------------------------------------------------------------- #
def header(mode, threads, extra=None):
    commit, clean = git_state(os.path.dirname(os.path.abspath(__file__)))
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
                 "numpy": np.__version__, "platform": platform.platform(), "cpu_count": os.cpu_count(),
                 "utc_created": utc_now()}, **(extra or {}))


def train_seeds(em, report, jsonl_path=None, nets_dir=None, nets_g3=None, threads=THREADS_DEFAULT, n_block=DRAW_N,
                n_valid=N_VALID, seeds=range(2, N_SEEDS + 1), train_kw=None):
    """Seeds q = 2..5 at n* per cell (q = 1's run is the ladder's), without the wall budget
    of G3 (line 144's budget is the gate's). A non-finite loss is retrained once per cell
    from seed value 6, which takes the failed seed's place; a second failure leaves that
    slot empty, the cell standing with four seeds (line 75). Returns {cell: {n_star,
    seeds: {slot q: run label}, empty_slots}}."""
    jsonl_path, nets_dir, nets_g3 = jsonl_path or PATHS["train"], nets_dir or PATHS["nets"], nets_g3 or PATHS["nets_g3"]
    train_kw = dict(train_kw or {})
    train_kw.setdefault("wall_budget", float("inf"))
    if not report.get("passed"):
        raise SystemExit("G3 has not passed: the seeds are not trained (line 144)")
    hdr, launches, tasks = read_jsonl(jsonl_path)
    if hdr is None:
        append_line(jsonl_path, header("train", threads))
    out = {}
    for key, cell in report["cells"].items():
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


def score_set(set_name, refs, em, trained, nets_dir=None, nets_g3=None, out_path=None, threads=THREADS_DEFAULT,
              post_kw=None):
    """Every calibrator at every (prior, level, seed) on the set's surfaces at its level:
    per pair the outputs of line 75; `q` is the seed's slot (a seed-6 replacement carries
    q6 in its run label). Resumable by (run label, pair label)."""
    require_torch()
    torch.set_num_threads(int(threads))
    nets_dir, nets_g3 = nets_dir or PATHS["nets"], nets_g3 or PATHS["nets_g3"]
    out_path, post_kw = out_path or scores_path(set_name), dict(post_kw or {})
    hdr, launches, done = read_jsonl(out_path)
    if hdr is None:
        append_line(out_path, header("score", threads, {"set": set_name}))
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
def main(argv=None):
    ap = argparse.ArgumentParser(description="P5 gate G3 calibrators (see module docstring)")
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--g3", action="store_true", help="the ladder of line 144 and the report")
    mode.add_argument("--train", action="store_true", help="seeds 2 to 5 at the adequate n per cell, after G3")
    mode.add_argument("--score", default=None, choices=("g3", "test"), help="the calibrators' outputs on a set")
    ap.add_argument("--emulator", default=EMULATOR_PATH)
    ap.add_argument("--threads", type=int, default=THREADS_DEFAULT, help="torch threads for training and scoring; recorded")
    a = ap.parse_args(argv)
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
    if not os.path.exists(a.emulator):
        raise SystemExit(f"no emulator at {_rel(a.emulator)}: G1 is built after freezing (line 54)")
    em = E.load_emulator(a.emulator)
    refs_path = R.SETS["g3"]["jsonl"]
    if a.g3 or a.score == "g3" or a.train:
        if not os.path.exists(refs_path):
            raise SystemExit(f"no G3 pilot references at {_rel(refs_path)}: run p5_g2_reference.py --g3-pilot first")
    if a.g3:
        _, _, refs = read_jsonl(refs_path)
        rep = run_g3(em, refs, threads=a.threads)
        print(f"=== G3: {'passed' if rep['passed'] else 'NOT passed; P5 stops (line 144)'}; "
              f"cells not passing: {rep['cells_not_passing']} ===", flush=True)
        for key, c in rep["cells"].items():
            print(f"  {key}: n* = {c['n_star']} ({len(c['rungs'])} rungs; {len(c['removed_references'])} references removed)",
                  flush=True)
        print(f"  written: {_rel(PATHS['report'])}", flush=True)
        return
    with open(PATHS["report"], encoding="utf-8") as f:
        rep = json.load(f)
    if a.train:
        trained = train_seeds(em, rep, threads=a.threads)
        with open(os.path.join(OUT_DIR, "p5_calibrators_trained.json"), "w", encoding="utf-8") as f:
            json.dump(trained, f, indent=1)
        print(f"  trained seeds per cell written: {_rel(os.path.join(OUT_DIR, 'p5_calibrators_trained.json'))}", flush=True)
        return
    if a.score:
        with open(os.path.join(OUT_DIR, "p5_calibrators_trained.json"), encoding="utf-8") as f:
            trained = json.load(f)
        set_refs = R.SETS[a.score]["jsonl"]
        if not os.path.exists(set_refs):
            raise SystemExit(f"no references for the set {a.score} at {_rel(set_refs)}")
        _, _, refs = read_jsonl(set_refs)
        n = score_set(a.score, refs, em, trained, threads=a.threads)
        print(f"  scored {n} new (run, pair) results -> {_rel(scores_path(a.score))}", flush=True)


if __name__ == "__main__":
    main()
