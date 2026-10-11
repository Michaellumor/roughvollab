"""
test_p5_g1_emulator.py — mechanics tests for the P5 gate G1 emulator script. They guard
the MACHINERY on cheap settings: the literals written into the draft (lines 39, 54, 141,
142; D62, D64); the Chebyshev mathematics on known functions; the pricing path against
the pilot's and the wrapper's at cheap settings on D38's point (as G0's identity gate);
the held-out draw; the stop on a quote that is not finite; the output schema, resume and
assembly on a toy build. Nothing here builds the emulator or measures the pricer; the
toy build prices a few points at N_riccati 200 on a small box around D38's point.
"""
import csv
import json
from pathlib import Path

import numpy as np
import pytest

import p5_g1_emulator as E
import p5_g0_pilot as P
import p5_g0_main as M
import layer4_calibrate_surface as S

D38 = P.D38_POINT
VUS3 = np.array([-1.0, 0.0, 1.0])
CHEAP_T = {0.10: (100.0, 32), 0.25: (100.0, 32)}
TOY_BOX = {"H": (0.08, 0.12), "nu": (0.30, 0.35), "rho": (-0.75, -0.70), "xi0": (0.04, 0.06)}


def test_literals_are_the_written_ones():
    taken = json.loads(Path(E.OUT_DIR, "p5_g0_main_criterion.json").read_text(encoding="utf-8"))["taken"]
    assert E.BOX["H"] == (0.02, 0.48)
    for d in ("nu", "rho", "xi0"):
        assert list(E.BOX[d]) == taken["bounds"][d]                        # D62's box, read from the record
    assert all(int(v) == E.N_RICCATI for v in taken["resolution"].values())
    assert E.NODES == (16, 8, 6, 12) and int(np.prod(E.NODES[:3])) == 768 and int(np.prod(E.NODES)) == 9216
    assert E.BASE_T is M.BASE_T and E.BASE_T[0.10] == (790.0, 504) and E.BASE_T[2.00] == (180.0, 112)
    assert E.SEED_HELDOUT == 73 and E.N_HELDOUT == 200 and E.RATIO == 5 and E.NOISE_LEVELS == (0.001, 0.003, 0.005)
    assert E.N_QUOTES == 35 and E.POOL_DEFAULT == 4 and E.COST_PER_TRIPLE_S == 288.0
    assert E.SEED_HELDOUT not in (20261002, 7, 61, 62)
    est = E.estimate()
    assert est["n_sets"] == 968 and abs(est["core_hours"] - 968 * 288 / 3600) < 1e-9 and abs(est["wall_hours"] * 4 - est["core_hours"]) < 1e-9


def test_chebyshev_nodes_weights_and_exactness_at_nodes():
    for n in (1, 2, 5, 16):
        x = E.cheb_nodes(n)
        assert x.shape == (n,) and np.all(np.diff(x) > 0) and np.all(np.abs(x) < 1)
        assert np.allclose(np.sort(np.cos((2 * np.arange(n) + 1) * np.pi / (2 * n))), x)   # first kind
        A = E.bary_matrix(x, n)
        assert np.array_equal(A, np.eye(n))                                                 # exact at a node
        rows = E.bary_matrix(np.linspace(-1, 1, 7), n)
        assert np.allclose(rows.sum(axis=1), 1.0)                                           # reproduces constants
    w = E.bary_weights(4)
    assert np.allclose(np.abs(w), np.sin((2 * np.arange(4) + 1) * np.pi / 8)) and np.sign(w[0]) != np.sign(w[1])


def test_interpolation_recovers_smooth_and_polynomial_functions():
    rng = np.random.default_rng(0)
    box, nodes = E.BOX, (9, 7, 5, 8)
    g = E.node_grid(nodes, box)
    H, nu, rho, xi = np.meshgrid(*g, indexing="ij")
    f_smooth = lambda h, n, r, x: np.exp(-3 * h * x) * np.cos(2 * n + r) + 0.1 * np.sin(5 * x)   # entire in each variable
    V = np.stack([f_smooth(H, nu, rho, xi), 2.0 * f_smooth(H, nu, rho, xi)], axis=-1)              # two "quotes"
    lo, hi = E.box_arrays(box)
    pts = rng.uniform(lo, hi, size=(300, 4))
    out = E.evaluate_many(V, pts, box)
    ref = f_smooth(*pts.T)
    assert np.max(np.abs(out[:, 0] - ref)) < 1e-8 and np.allclose(out[:, 1], 2 * out[:, 0])
    # a tensor-product polynomial of degree below the node count is reproduced exactly (to rounding)
    u = E.to_unit(np.stack([H, nu, rho, xi], axis=-1).reshape(-1, 4), box).reshape(nodes + (4,))
    poly = lambda a, b, c, d: (a ** 3 - a) * (b ** 2 + 1) * (c + 0.5) * (d ** 4 - 2 * d)
    Vp = poly(u[..., 0], u[..., 1], u[..., 2], u[..., 3])[..., None]
    up = E.to_unit(pts, box)
    assert np.max(np.abs(E.evaluate_many(Vp, pts, box)[:, 0] - poly(*up.T))) < 1e-12
    # scalar evaluate equals the vectorised one, and a point outside the box raises
    assert np.array_equal(E.evaluate(V, pts[3], box), out[3])
    with pytest.raises(ValueError):
        E.evaluate_many(V, [[0.01, 0.1, -0.8, 0.1]], box)


def test_coefficients_tails_and_refinement_rule():
    n = 6
    x = E.cheb_nodes(n)
    T3, T1 = 4 * x ** 3 - 3 * x, x
    c = E.cheb_coeffs(T3 + 0.5 * T1 + 2.0, axis=0)
    assert np.allclose(c, [2.0, 0.5, 0.0, 1.0, 0.0, 0.0], atol=1e-12)
    V = np.zeros((4, 3, 5, 2, 1))
    xs = [E.cheb_nodes(k) for k in V.shape[:4]]
    V[..., 0] = (2 * xs[0] ** 2 - 1)[:, None, None, None] + 0.3 * (8 * xs[2] ** 4 - 8 * xs[2] ** 2 + 1)[None, None, :, None]
    t = E.tails(V)
    assert abs(t[2] - 0.3) < 1e-12 and max(t[0], t[1], t[3]) < 1e-12                       # only rho's top coefficient
    assert E.refine_dimension(t) == 2 and E.refine_dimension([1.0, 1.0, 0.5, 0.2]) == 0    # lowest index on a tie


def test_pricing_path_equals_the_pilot_and_the_wrapper_bit_for_bit():
    T, N, n = 0.25, 200, 32
    rec = E.price_triple(D38[0], D38[1], D38[2], (D38[3],), N=N, settings={T: (200.0, n)}, Ts=(T,), vus=VUS3)
    cheap = {"A": (200.0, n), "C": (200.0, 2 * n), "D": (400.0, 2 * n)}
    pil = P.run_task(D38[:3], T, ladder=(N,), settings=cheap, xi0s=(D38[3],), vus=VUS3)
    mn = M.run_task_main(D38[:3], T, ladder=(N,), partner=None, settings=cheap, xi0s=(D38[3],), vus=VUS3)
    key = P._xi_key(D38[3])
    assert rec["quotes"][0][0] == pil["rungs"][0]["quotes"]["A"][key]["quotes"]              # the pilot's path
    assert rec["quotes"][0][0] == mn["rungs"][0]["quotes"]["A"][key]["quotes"]               # the main run's path (line 52)
    assert rec["per_T"][str(T)]["xi0"][key]["K"] == pil["anchors"][key]["K"] == mn["anchors"][key]["K"]   # anchored strikes
    assert all(M.BASE_T[t] == M.settings_for(t)["A"] for t in M.TS)                           # the written base settings
    atm = S.atm_by_T(D38, [T], N_riccati=N, n_nodes=n)
    ref = S.model_smile_cf_T(D38, S.fixed_surface_strikes(atm, VUS3, [T])[T], T, N_riccati=N, n_nodes=n)
    assert np.array_equal(np.array(E.flat_quotes(rec, 0), dtype=float), ref)                # exact, not approximate
    assert rec["per_T"][str(T)]["xi0"][key]["sigma_atm"] == atm[T] and rec["all_finite"]
    assert rec["N"] == N and rec["per_T"][str(T)]["U_max"] == 200.0 and rec["per_T"][str(T)]["n_nodes"] == n


def test_heldout_draw_is_the_written_one():
    pts = E.heldout_points()
    ref = np.random.default_rng(73).uniform(*E.box_arrays(E.BOX), size=(200, 4))
    assert pts.shape == (200, 4) and np.array_equal(pts, ref)
    lo, hi = E.box_arrays(E.BOX)
    assert np.all(pts >= lo) and np.all(pts <= hi)
    jobs = E.heldout_jobs()
    assert [j["label"] for j in jobs[:2]] == ["HO000", "HO001"] and jobs[5]["xi0s"] == [float(pts[5, 3])]
    tj = E.train_jobs()
    assert len(tj) == 768 and tj[0]["label"] == "N00_00_00" and len(tj[0]["xi0s"]) == 12
    g = E.node_grid()
    assert [len(x) for x in g] == [16, 8, 6, 12] and g[0][0] > 0.02 and g[0][-1] < 0.48 and g[3][0] > 0.04


def test_nonfinite_quote_is_a_status_that_stops_the_launch(tmp_path, monkeypatch):
    jl = str(tmp_path / "t.jsonl")
    orig = E.price_triple

    def bad_price(H, nu, rho, xi0s, N=None, settings=None, Ts=E.TS, vus=E.VUS):
        rec = orig(0.10, 0.35, -0.70, xi0s, N=200, settings=CHEAP_T, Ts=(0.10,), vus=VUS3)
        if H > 0.11:                                                   # one triple "fails" at the inversion stage
            rec["quotes"][0][0][1] = None
            rec["status"][0][0][1] = "iv_failed"
            rec["all_finite"] = False
        return rec
    monkeypatch.setattr(E, "price_triple", bad_price)
    jobs = E.train_jobs((2, 1, 1, 1), TOY_BOX)
    hdr = E.header(1, (2, 1, 1, 1), TOY_BOX)
    n, stopped = E.launch(1, jobs, jl, hdr, task_kwargs={"Ts": (0.10,), "vus": VUS3})
    assert stopped == "N01_00_00"
    h, launches, train, held = E.read_stage(jl)
    assert train["N01_00_00"]["all_finite"] is False and train["N01_00_00"]["status"][0][0][1] == "iv_failed"
    lines = [json.loads(l) for l in Path(jl).read_text(encoding="utf-8").splitlines()]
    assert lines[-1]["kind"] == "stop" and lines[-1]["task"] == "N01_00_00"
    probs = E.check_complete([j["label"] for j in jobs], train, "train")
    assert probs == ["train N01_00_00: a quote is not finite (iv_failed)"]
    with pytest.raises(SystemExit, match="launch refused: the record holds a quote that is not finite"):
        E.launch(1, jobs, jl, hdr, task_kwargs={"Ts": (0.10,), "vus": VUS3})                  # the stop survives a resume
    with pytest.raises(SystemExit, match="launch refused: .* was started with nodes"):
        E.launch(1, jobs, jl, E.header(1, (3, 1, 1, 1), TOY_BOX), task_kwargs={"Ts": (0.10,), "vus": VUS3})
    with pytest.raises(SystemExit, match="assemble refused"):
        E.assemble("base", (2, 1, 1, 1), base_paths={"jsonl": jl}, stage_paths={"jsonl": jl, "npz": str(tmp_path / "e.npz"),
                   "csv": str(tmp_path / "e.csv"), "report": str(tmp_path / "e.json")}, n_heldout=0, Ts=(0.10,), vus=VUS3)


def test_toy_build_schema_resume_and_assembly(tmp_path, capsys):
    sp = {"jsonl": str(tmp_path / "b.jsonl"), "npz": str(tmp_path / "b.npz"), "csv": str(tmp_path / "b.csv"),
          "report": str(tmp_path / "b.json")}
    nodes = (2, 2, 2, 2)
    kw = {"N": 200, "settings": CHEAP_T, "Ts": (0.10, 0.25), "vus": VUS3}
    jobs = E.train_jobs(nodes, TOY_BOX) + E.heldout_jobs(73, 2, TOY_BOX)
    hdr = E.header(2, nodes, TOY_BOX, settings=CHEAP_T, N=200)
    n1, s1 = E.launch(2, jobs, sp["jsonl"], hdr, task_kwargs=kw)
    n2, s2 = E.launch(2, jobs, sp["jsonl"], hdr, task_kwargs=kw)                   # a second launch finds nothing to do
    assert (n1, n2, s1, s2) == (10, 0, None, None)
    h, launches, train, held = E.read_stage(sp["jsonl"])
    assert h["nodes"] == [2, 2, 2, 2] and h["box"]["H"] == [0.08, 0.12] and h["N"] == 200 and h["base_T"]["0.1"] == [100.0, 32]
    assert [l["n_done_before"] for l in launches] == [0, 10] and all(l["commit"] == h["commit"] for l in launches)
    assert len(train) == 8 and len(held) == 2 and all(r["all_finite"] for r in list(train.values()) + list(held.values()))
    rep = E.assemble("base", nodes, base_paths=sp, stage_paths=sp, n_heldout=2, Ts=(0.10, 0.25), vus=VUS3)
    em = E.load_emulator(sp["npz"])
    assert em.values.shape == (2, 2, 2, 2, 6) and em.nodes == nodes and em.N == 200 and em.stage == "base"
    g = E.node_grid(nodes, TOY_BOX)
    node_pt = [g[0][1], g[1][0], g[2][1], g[3][0]]
    assert np.array_equal(em.evaluate(node_pt), np.array(E.flat_quotes(train["N01_00_01"], 0)))   # exact at a node
    thetas, G = E.heldout_table(held, 2)
    assert rep["e_p99"] == float(np.percentile(np.abs(em.evaluate_many(thetas) - G), 99))      # the record's box, not BOX
    assert rep["n_quotes"] == 12 and rep["n_train"] == 8 and set(rep["verdicts"]) == {"0.001", "0.003", "0.005"}
    assert rep["verdicts"]["0.001"]["bound"] == 0.0002 and rep["refine"] == (rep["e_p99"] > 0.0002)
    assert set(rep["tails"]) == {"H", "nu", "rho", "xi0"} and rep["per_T_p99"].keys() == {"0.1", "0.25"}
    assert (rep["refine_dimension"] is None) == (not rep["refine"]) and ("refined_nodes" in rep) == rep["refine"]
    rows = list(csv.DictReader(Path(sp["csv"]).read_text(encoding="utf-8").splitlines()))
    assert len(rows) == 2 * 2 * 3 and list(rows[0].keys()) == ["point", "H", "nu", "rho", "xi0", "T", "z", "G", "G_tilde", "diff"]
    assert abs(float(rows[0]["G_tilde"]) - float(rows[0]["G"]) - float(rows[0]["diff"])) < 1e-15
    json.dumps(json.loads(Path(sp["report"]).read_text(encoding="utf-8")), allow_nan=False)
    out = capsys.readouterr().out
    assert "10 tasks to do" in out and "0 tasks to do (10 already" in out


def test_g1_verdicts_and_assess_on_hand_built_numbers():
    v = E.g1_verdicts(0.00025)
    assert [v[k]["claimed"] for k in ("0.001", "0.003", "0.005")] == [False, True, True]
    assert E.g1_verdicts(0.0002)["0.001"]["claimed"] is True                              # the bound itself passes
    nodes, box = (3, 3, 3, 3), E.BOX
    g = E.node_grid(nodes, box)
    H, nu, rho, xi = np.meshgrid(*g, indexing="ij")
    V = np.repeat((H + nu + rho + xi)[..., None], 6, axis=-1)                               # linear: interpolated exactly
    thetas = np.random.default_rng(1).uniform(*E.box_arrays(box), size=(5, 4))
    G = np.repeat(thetas.sum(axis=1)[:, None], 6, axis=1) + 0.0009                          # pricer "off" by 0.0009
    rep, Gt, diff = E.assess(V, thetas, G, box, Ts=(0.10, 0.25), vus=VUS3)
    assert abs(rep["e_p99"] - 0.0009) < 1e-12 and abs(rep["max_abs"] - 0.0009) < 1e-12 and rep["n_quotes"] == 30
    assert rep["refine"] is True and rep["refine_dimension"] in E.DIMS and rep["verdicts"]["0.005"]["claimed"] is True
    rep2, _, _ = E.assess(V, thetas, G, box, Ts=(0.10, 0.25), vus=VUS3, stage="refined")
    assert rep2["refine"] is None and rep2["refine_dimension"] is None                       # no second refinement
    c4 = E.cheb_coeffs_4d(V)                                                                 # linear: four first-order terms
    assert np.allclose(np.abs(c4[..., 0]).sum(), abs(c4[0, 0, 0, 0, 0]) + abs(c4[1, 0, 0, 0, 0]) + abs(c4[0, 1, 0, 0, 0])
                       + abs(c4[0, 0, 1, 0, 0]) + abs(c4[0, 0, 0, 1, 0]), atol=1e-12)
