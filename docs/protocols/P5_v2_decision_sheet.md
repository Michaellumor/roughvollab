# P5 v2 decision sheet: the 16 OPEN items

## Status

A working aid for the author. It is not a protocol and it decides nothing. No item below is decided, no checklist box is ticked and nothing is frozen. v1 (`docs/protocols/P5_protocol.md`, OSF https://doi.org/10.17605/OSF.IO/APSF7) remains the registered protocol (D55). Prepared on 4 October 2026; the decisions are the author's.

The sheet takes each item the v2 draft (`docs/protocols/P5_protocol_v2_draft.md`) marks OPEN and sets beside it what is being chosen, what the repository already shows, what the literature does, what each option costs, and the question the author must be able to answer before freezing.

## How it was written

- **Line numbers.** "L" is a line of the v2 draft, "v1 L" a line of v1, "ROADMAP L" a line of `ROADMAP.md`, all as at commit `28e441b`. Code is cited as `file:line`.
- **Repository facts** are read, not run. Nothing was executed against the pricer, and no Jacobian, Fisher information, Cramér–Rao bound, posterior, calibration or emulator was computed.
- **Literature** statements are paraphrases with a pinpoint (section, table or equation). Each was read, while this sheet was prepared, at the source listed under "Sources"; a source read only as an abstract is marked there. No third-party text is quoted. Anything that could not be read at source is under "Leads, not verified" and is not used as support.
- **Arithmetic** is done on numbers already recorded in the draft or the repository and is labelled. It is never a new measurement. Where it rests on an assumption, such as a Gaussian approximation or an optimiser's scatter read as a posterior width, the item says so.
- **Options.** The first option in each table is the draft's own where the draft has one; item 4's three tables have none, because L52 makes no proposal. What an option costs is reasoning from the draft's definitions unless a source is named.
- **Reasons.** Where no reason for a choice is recorded, the sheet says so and supplies none.
- **Views.** None is given on the author's bets: the hypothesis numbers, the priors, the noise levels and the cut-offs. One view is given, in item 7, on a convention that has a published standard.

## Order of decisions

**The sequence of events the draft implies.** Freeze; G0 (pricer over the box: box, grid, cost); the emulator is built once; G1; reference pilots and G2; the amendment that writes in the number of test θ; G3 (training-set size); confirmatory networks; Stage 1 on the reference alone; Stage 2. The draft does not state where Stage 1 falls relative to G3 and network training, and L92's "before any network is trained" does not say whether G3's pilot networks count (item 16).

**How each item is decided.**

| # | Item | Draft line | Decided as | Settle first |
| --- | --- | --- | --- | --- |
| 1 | Parameters: the box | L37 | The box is a value now; a shrink is a rule G0 applies | 13, 2 |
| 2 | Grid: whether to drop T = 2.00 | L39 | A value now, or a rule if left to G0 | 3, 13, 1 |
| 3 | Observable | L41–44 | A value now | 2, 1 |
| 4 | Emulator: form, count, design | L52 | Form and design are values now; the count is a value now or a rule fixed at a gate | 2, 3, 1, 5, 14, 7 |
| 5 | Noise: the three levels | L54 | A value now | 2, 3, 6, 9, 11 |
| 6 | Priors: the Beta parameters | L64 | A value now | 1 |
| 7 | The exact reference | L66 | Settings are values now; the length is a value now or a rule now with its numbers observed at G2 | 4, 6, 1, 2, 3, 5, 9, 12 |
| 8 | Calibrators | L73 | Values now | 2, 3, 5, 6, 4 |
| 9 | Stratification: the cut-offs | L87 | A value now | 6, 7, 12 |
| 10 | Test sets: the number of test θ | L92 | A rule now; the value is fixed at G2 | 7, 15, 5, 1, 2, 8, 9, 11 |
| 11 | Stage 1: the 10% floor | L99 | The share is a value now; its size in pairs is fixed at G2 | 10, 5, 9, 6, 8 |
| 12 | Hypotheses H-a to H-d | L115 | Values now | 6, 9, 5, 8, 7, 10, 11, 16 |
| 13 | Gate G0: tolerance | L139 | A value now | 5, 14 |
| 14 | Gate G1: the ratio | L140 | A value now | 5, 2, 3, 13 |
| 15 | Gate G2: tolerance | L141 | Measure and tolerance are values now; the grid's size waits on the emulator's cost | 7, 4, 9, 12, 5, 6, 1, 2, 3, 10 |
| 16 | Gate G3: tolerance | L142 | The tolerance, statistic, pilot design and budget are values now; the training-set size is fixed at the gate | 15, 8, 4, 5, 6, 9 |

**The dependencies are circular in places**, so some items can only be settled together:

- Items 5, 9 and 11 are one decision in three parts: the noise levels fill the bins, the cut-offs define them, and the floor tests whether they were filled. Item 6 comes first, because the cut-offs are fractions of prior A's variance.
- Item 12 cannot be read apart from items 6 and 9: on a Gaussian approximation the reference's own shift ratio at the edge of the Identified bin is about the size of H-c's Expected value.
- Items 7 and 15 are one decision seen twice: the Monte Carlo error allowed in the reference's mean of H, in c and in m. Both follow from items 9 and 12.
- Items 4 and 7 constrain each other: a gradient-based sampler needs a differentiable emulator.
- Items 13 and 14 share one budget if the pricer's error and the emulator's error both count against the noise level.
- Items 8, 12 and the analysis plan share the definition of a training seed.

**What a gate would fix later.** The box and the grid if G0 fails (items 1, 2, 13). The number of test θ at G2 (item 10). The training-set size at G3 (item 16). The reference's length, if it is left to a rule (item 7). The emulator's training-point count, if it is left to a rule (item 4). For the first three the draft names the gate; for the reference's length and the emulator's count it names none (L66, L52). It states what happens on failure for G0 and G3 (L139, L142) and not for G2 (L141). It gives no order between shrinking the box and shrinking the grid, no formula or budget that turns G2's measured cost into a number of test θ, and no figure for the "budget" of L142. The items say what is missing.

## The 16 items

## 1. Parameters: the box may be shrunk at gate G0 (draft L37)

**The choice.** L37: "**OPEN — the box may be shrunk at gate G0** if the pricer cannot be made finite and converged over all of it at affordable cost. Any change is made before the emulator is built."

**Decided as.** Mixed. The box is a value now. Whether it shrinks, where and by how much is a rule whose outcome G0 fixes.

**What depends on it.**

- The priors, "0.02 + 0.46 · Beta(…)" (L60): 0.46 is the width of the H range.
- The shift-ratio denominator 0.20 (L106), which is the gap between the two prior means.
- The sealed test draw, "θ uniform on the box, seed 20261002" (L91).
- The emulator, built "over the box" (L46), and through it the bin shares behind H-a to H-d (L97–99, L119–122).
- The novelty claim "the identifiability map on a stated grid" (L156), which is a map over this box.

**Settle first.**

- Item 13 (G0 tolerance): "finite and converged" has no test until the tolerance, its unit and the starting `N_riccati` are stated.
- Item 2 (T = 2.00): L139 says "the box or the grid is shrunk here" and gives no order between the two.
- A cost ceiling: "at affordable cost" (L37) names no figure.

**What the repository shows.**

- The box in code is H [0.02, 0.49], ν [0.05, 1.00], ρ [−0.99, 0.00], ξ₀ [0.001, 0.25] (`layer4_calibrate_surface.py:41-42`; the same in `layer4_calibrate.py:44-45`). No comment gives a reason for any bound.
- Why the draft lowers 0.49 to 0.48 is not recorded. v1 already used [0.02, 0.48] for the prior support (v1 L93) while quoting 0.49 for the box (v1 L49).
- κ = 0.30 is set in `layer4_calibrate.py:39`. D38 EXP 4 supports keeping κ fixed (ROADMAP L851). The reason for the value 0.30 is not recorded.
- H = 0.02 is near a numerical floor: "H<0.02 NaNs the long tenor" (D39, ROADMAP L864).
- D55's probe: 9 of 24 uniform points had at least one non-finite maturity at `N_riccati` 1000. The follow-up covered four of the nine, ATM quote only (ROADMAP L1049–1056). The failing points' coordinates are not recorded and the script is not in the repository (ROADMAP L1044–1047), so D55 does not say which part of the box its nine failures came from.
- Code comments place overflow at low H, high ν and long T (`calibrate_btc.py:7-8`) and at small H, high ν and low `N_riccati` (`rough_heston_lifted.py:231-232`).
- D41 records H = 0.02, ν = 0.63 as finite and converged at T = 0.986 with N = 8000 under a per-maturity schedule, and no NaN among the 69 quotes at the fitted optimum, H = 0.02, ν = 0.71 (ROADMAP L882, L885; `analysis/layer4_span_identifiability.py:28`). A code comment records that with T pinned at about 0.989 the one-year tenor overflows at N = 8000 and is finite and bit-converged at the railed θ̂ for N ≥ 8200 (`calibrate_btc.py:61-64`).
- The points named in D39, D41 and D42 and in the `calibrate_btc.py` pre-check go no further than ν of about 0.71, ρ = −0.50 and T of about 0.99 (ROADMAP L860–864, L882–885, L892–893; `calibrate_btc.py:102-103`). ρ = −0.70 is recorded at the D38 truth point (ν = 0.35), finite on all 35 quotes including T = 2.00 at the default knobs (`layer4_calibrate_surface.py:38`, `:44`), and at the CF test point (H = 0.10, ν = 0.20, T = 1; `test_rough_heston_cf.py:20`, `:141-146`). D55 priced T = 2.00 at 24 uniform points and records timing and finiteness, not convergence, and no coordinates (ROADMAP L1050–1056).
- The Riccati solve costs O(N²) (D30, ROADMAP L775; D41, ROADMAP L887).
- P1's live optima sit on the box edge or outside it: H = 0.0200 for BTC and ETH, and ETH's ξ₀ = 0.3973 is above 0.25 (`analysis/layer4_span_identifiability.py:28-29`).
- No test names a point with ρ below −0.70 (`test_rough_heston_cf.py:21`, `:142`). Two test inputs sit near the low-H, high-ν edge of the box, (H, ν) = (0.025, 0.99) and (0.03, 0.95), at ρ = −0.5, ξ₀ = 0.04 and N = 150; the tests check that the output keeps its length and that a non-finite value becomes a penalty, not that the quotes are finite (`test_layer4_calibrate.py:40-43`, `:56-57`).

**What the literature does.**

- [BHL] use a narrower box for rough Heston, with a uniform simulation prior: ν (0.05, 0.45), ρ (−0.95, −0.1), H (0.05, 0.45), and κ and the long-run variance free (Section 5.1.1, Table 1). They generate surfaces by Monte Carlo, not by the characteristic function.
- [HMT] sample rough Bergomi, not rough Heston, with ρ in [−0.95, −0.1] and H in [0.025, 0.5] (Section 4.1.1).
- [ER], the source the repository pins its characteristic function to (D30), states its main theorem for ρ in (−1/√2, 1/√2], that is ρ above about −0.707 (Theorem 4.1). The draft's box runs to −0.99.
- [ALP] treat the Volterra Heston model with ρ in [−1, 1], prove, under conditions on the kernel, that the Riccati–Volterra equation has a unique global solution when Re ψ₁ lies in [0, 1], Re u₂ ≤ 0 and Re f₂ ≤ 0, where ψ₁ = u₁ + 1∗f₁ is the log-price component (Eq. 7.3), and recover rough Heston as the fractional-kernel case (Section 7, Theorem 7.1(ii), Example 7.2).
- [HK] state that accurate prices by the fractional Adams scheme may need thousands to tens of thousands of time steps and that cost is quadratic in the number of steps (Section 3.1).

**Arithmetic on recorded numbers.**

- 9 / 24 = 37.5% of probe points failed; a Wilson 95% interval on that count is about 21% to 57%. One seed, finiteness only, sandbox run.
- If the 1.6 s per surface scaled purely as N², doubling N uniformly would give about 6.4 s at 2000, 26 s at 4000 and 102 s at 8000. Extrapolation, not a measurement.
- The constant-step schedule now in the code (`calibrate_btc.py:69`: h = 1.10e-4, N between 800 and 10000; D41 as recorded used h = 1.23e-4 and a ceiling of 8000, ROADMAP L882) applied to the five P5 maturities gives N of about 909, 2273, 4545, 9091 and 18182 (clipped at 10000): roughly 42 to 88 times the Riccati work of a uniform N = 1000. No run of that schedule at T = 2.00 or at ν up to 1.00 is recorded.
- 28.6% of a uniform ρ on [−0.99, 0.00] lies below −1/√2, outside the range of [ER]'s theorem. [ALP] cover the whole range, so this concerns what the repository's cited source states and what its tests exercise.
- The priors' means are 0.02 + 0.46 × 2.5/11.5 = 0.12 and 0.02 + 0.46 × 7.5/11.5 = 0.32; their gap, 0.20, is 0.4348 × the H width. If the H range is cut, the means, the standard deviations and the 0.20 all move unless the Beta parameters are chosen again.

**Options.**

| Option | What it means | What it costs |
| --- | --- | --- |
| Keep the box and allow a shrink at G0 (the draft) | Freeze the stated box; shrink before the emulator is built if G0 fails | The box used by the priors, the test draw, the emulator and the bin shares is not fixed at freeze. The draft has no rule for which dimension is cut or by how much, so the cut would be chosen after pilot output is seen and recorded as a dated amendment (L161) |
| Keep the box and write the shrink rule down first | As the draft, but state now which of grid and box is cut first, which dimensions may be cut, the pass criterion and cost ceiling, and how priors and the test draw are re-derived | G0 then fixes a value by rule. The rule would be written without the coordinates of D55's failing points, which are not recorded |
| Fix the box with no shrink option; meet G0 by resolution | Raise `N_riccati`, uniformly or per maturity, until every quote is finite and converged; otherwise shrink the grid or stop | Cost is O(N²). Whether any N makes ν up to 1.00, ρ down to −0.99 and T = 2.00 finite together is not recorded |
| Shrink the box now | Choose a smaller box on existing evidence and freeze it | The recorded basis for where to cut is the description of the overflow regime as low H, high ν and long T (ROADMAP L864; `calibrate_btc.py:7-8`); D55 gives a count and no coordinates. A smaller box departs from "keeps P1's box" (L43); raising the H floor would exclude the region P1's fits occupy |
| Keep the box and change the pricer's numerics | Adopt the per-maturity schedule in `calibrate_btc.py`, or a different Riccati solver | No test references `calibrate_btc`. The schedule's stability check runs at ρ = −0.50, ξ₀ = 0.185 (`calibrate_btc.py:102`, `:140-148`) and its current step was set from a probe at the railed θ̂ (`calibrate_btc.py:61-65`), both for maturities up to about 0.99. A new solver is code the "existing verification tests" (L139) do not cover |

The consequences above are reasoning from the draft's definitions and the recorded numbers, not statements in a source.

**Established standard.** None found. [BHL] and [HMT] are examples of practice; both use a narrower range in ρ than the draft.

**To defend it, you would need to be able to say.** If G0 fails somewhere in the box, which part do I cut, by a rule I wrote down before seeing the pilot, and why are my priors, my 0.20 yardstick, my sealed test draw and my hypothesis numbers still the ones I registered after the cut?

**Checklist boxes.** None names the box. It bears on "Gate tolerances declared" (L176) and "Beta parameters accepted or changed" (L171).

**Not confirmed.** Which θ failed in the D55 probe. Whether the per-maturity schedule keeps quotes finite at ν up to 1.00, ρ down to −0.99, ξ₀ up to 0.25 or T = 2.00. The cost per surface on your own machine.

## 2. Grid: whether to drop T = 2.00 (draft L39)

**The choice.** L39: "v2 uses D38's: maturities {0.10, 0.25, 0.50, 1.00, 2.00}, seven standardised moneyness points {−2, −1, −0.5, 0, 0.5, 1, 2}, 35 quotes. **OPEN — whether to drop T = 2.00.** It is where the pricer fails most and, per D41, where H information is least."

**Decided as.** A value now, unless you choose to leave it to G0, in which case it becomes a rule.

**What depends on it.**

- The forward map "G: θ → 35 numbers" (L43); without T = 2.00 it is 28.
- Gate G0 (L139) and the emulator's output dimension and training cost (L46, L52).
- The contraction c, the bin shares and the 10% floor (L77–87, L99), and through them every hypothesis.
- The declared prior knowledge (L18): D38's scatter figures were measured on five maturities including T = 2.00.

**Settle first.** Item 3 (observable): under the self-standardised option each maturity needs a finite ATM quote to place its strikes. Item 13 (G0 tolerance), if the decision is left to the gate. Item 1 (box): shrinking the box and dropping T = 2.00 answer the same G0 failure.

**What the repository shows.**

- The D38 grid in code is `TS = [0.10, 0.25, 0.50, 1.00, 2.00]` with seven standardised moneyness points, plus a leaner five-point strike set for the noise ensemble (`layer4_calibrate_surface.py:35-37`).
- Every multi-maturity set in D38's span sweep contains T = 2.00 (`layer4_calibrate_surface.py:239-240`). EXP 2 has one multi-maturity set without it, {0.10, 0.25} (`:213-214`), but no two sets in either list differ by T = 2.00 alone, so D38 never isolated what T = 2.00 adds.
- D38 records that the shortest maturity gives the largest single gain in conditioning: "the **SHORT maturity (T=0.1) gives the biggest single jump**" (ROADMAP L853).
- D38's H-scatter figures were measured on 25 quotes (five strikes), at N = 900 and 100 nodes, with ten draws (ROADMAP L850, L855), at one parameter point (`layer4_calibrate_surface.py:38`, `:231`).
- **D41 never priced T = 2.00.** Its six maturities run from 0.027 to 0.986 years, and its finding that "the long tenor is the LEAST H-informative maturity" (ROADMAP L880) is about the one-year tenor: sensitivity "64.5 → … → T=0.986 (1yr): 5.2" (ROADMAP L884). It is a local diagnostic at the fitted BTC optimum, where H sits on the box boundary (ROADMAP L887).
- Adding D41's one-year tenor moved |flat[H]| by −0.001 (0.119 to 0.118) and cond by a factor of 0.97 (ROADMAP L883).
- D39's longest tenor was about six months (ROADMAP L859, L864); D42 stops at the one-year tenor (ROADMAP L892–893).
- D55's maturity-level record for T = 2.0 comes from four failing points, ATM quote only: "T = 2.0 was finite at N 2000 in two and still non-finite at N 4000 in the other two" (ROADMAP L1053–1056). No per-maturity failure count exists for the 24-point probe.
- The P1 paper's "least H-informative tenor" statement is about the one-year tenor (`OVERLEAF/P1/calibration_paper_revised.tex:344-346`).
- No test prices T = 2.00; the surface tests use `TS2 = [0.25, 1.0]` (`test_layer4_calibrate_surface.py:13-14`).
- v1 tied the grid to P1's: "**DECISION — the quote grid is P1's.**" (v1 L110).

**What the literature does.**

- [BHL] use a self-standardised 63-point surface with seven maturities from 5/252 to 2 years, so it includes T = 2. Their attribution analysis reports that information about H is spread across maturities, with contributions at both shorter and longer horizons; they caution that these are local, prior-specific attributions (Section 5.1.1; Section 6). In that study κ and the long-run variance are free.
- [HMT]'s grid has eight maturities from 0.1 to 2.0 years and eleven strikes (Section 4.1.1; rough Bergomi, Monte Carlo).
- [ER]'s numerical illustration shows the at-the-money skew growing without bound as maturity goes to zero in the rough case: the rough signature is a short-maturity feature (Section 5.2, Figure 1).
- [CGP] report that at a two-year maturity the Adams method took about four minutes with 150 time steps in their implementation (Section 5.2; Table 3).
- [BL] (Section 1; Section 6) and [BHL] (Section 2.1) say the accuracy of the Adams scheme and of fixed Fourier settings worsens as maturity *shortens*. The repository's recorded failures are the other kind, non-finite values at long maturity, and D55 measured finiteness only. Finite quotes at T ≤ 0.25 are therefore not evidence of accuracy there.

**Arithmetic on recorded numbers.**

- Quotes: 5 × 7 = 35 becomes 4 × 7 = 28, one fifth fewer. Span: log₂(2.00/0.10) = 4.32 octaves becomes 3.32.
- D41's ratios at the one-year tenor: 64.5 / 5.2 = 12.4-fold fall in sensitivity; adding that tenor raised λ_min by 12% and cut cond by 3%. One live optimum, unequal strike counts per maturity; nothing measured about T = 2.00.
- Share of the Riccati work spent on T = 2.00: 20% at a uniform N; about 48% to 75% under a constant-step schedule. Assumes cost proportional to N²; no run of the schedule at T = 2.00 is recorded.

**Options.**

| Option | What it means | What it costs |
| --- | --- | --- |
| Keep T = 2.00 (the grid as the draft states it) | Freeze all five maturities, 35 quotes | The grid stays the one D38 measured on and matches both literature grids, which reach two years. G0 must then show every T = 2.00 quote finite and converged across the box; if it cannot, the remedy shifts to the box |
| Drop T = 2.00 before freezing | Freeze {0.10, 0.25, 0.50, 1.00}, 28 quotes | Removes the maturity that was still non-finite at N 4000 in two of the four points D55 followed up (ROADMAP L1053–1056). The grid is no longer D38's, so "v2 uses D38's" and the D38 figures in L18 describe a different surface, and the repository holds no identifiability number for the four-maturity set. Removing data cannot raise the expected contraction, so on average the Identified bin can only lose pairs and the Flat bin gain them; by how much is not recorded |
| Keep T = 2.00 in the frozen grid and let G0 decide | Drop it at G0 only if the gate fails, as L139 allows | The decision becomes a rule. It needs an order between shrinking the grid and shrinking the box, and a definition of failure, neither of which the draft has. The checklist box "Grid and observable convention chosen" could not be ticked as a value |
| Replace T = 2.00 with another maturity | Keep five maturities by swapping in a shorter one | Leaves D38's grid altogether with no recorded numbers for the new set. Speculative: nothing in the repository measures such a grid |

The consequences are reasoning from the draft's definitions and the recorded numbers.

**Established standard.** None found. Both verified literature grids reach two years.

**To defend it, you would need to be able to say.** What do the seven T = 2.00 quotes add to what the surface says about H on my grid and across my box, as measured there and not inferred from D41's one-year live tenor at a single point, and is that worth making every one of them finite and converged?

**Checklist boxes.** "Grid and observable convention chosen" (L168).

**Not confirmed.** No measurement exists in the repository of a {0.10, 0.25, 0.50, 1.00} surface. Whether [BHL]'s finding carries over to a design with κ fixed is not addressed in their text.

## 3. Observable: self-standardised or fixed absolute grid (draft L41–44)

**The choice.** L41: "D38 anchors strikes once from the target's noiseless ATM vol, which hands a Bayesian observer the exact ATM term structure. v2 must close that leak. **OPEN — one of:**" L43: "1. *Self-standardised (proposed).* Quote i is the implied vol at θ's own standardised moneyness. The forward map is G: θ → 35 numbers. Coherent, leak-free, keeps P1's box." L44: "2. *Fixed absolute grid.* Fixed log-moneyness and maturities, as in the deep-calibration literature, with ξ₀ narrowed so every quote is priceable. Closer to deployed networks, further from P1."

**Decided as.** A value now.

**What depends on it.**

- The forward map G and everything built from it: the emulator's target, the reference likelihood and the calibrators' inputs (L46).
- The box and the sealed test set: option 2 narrows ξ₀ and so changes the box that test θ are drawn from (L37, L44, L91).
- Gates G0 and G1: which quotes exist, and whether an ATM solve precedes them.
- The contraction c, the bins and every hypothesis scored by bin, because contraction is a property of the chosen observable (L79, L81–85, L117–122).

**Settle first.** Item 2 (T = 2.00): L43 says "35 numbers"; without T = 2.00 it is 28. Item 1 (box): option 1's "keeps P1's box" holds only if G0 does not shrink it. Nothing in the hypotheses table has to be settled first.

**What the repository shows.**

- D38's grid is in ATM standard deviations: `VUS = np.array([-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0])`, commented "log-moneyness in std-devs (|vu|<=2)" (`layer4_calibrate_surface.py:36`).
- The anchoring ATM vol is the model implied vol at K = S0, computed per maturity with no noise (`layer4_calibrate_surface.py:71-72`).
- Strikes are built as "K(T) = S0·exp(vu·σ_atm(T)·√T) — same #std-devs at every T; anchored ONCE from the target" (`layer4_calibrate_surface.py:75-77`). `build_target` computes the ATM vols at the true parameters, turns them into strikes and prices the target on those strikes (`:94-98`). The strikes are never recomputed for a candidate θ.
- Noise is added to the quotes only; the same noiseless-anchored strikes are handed to every noisy recalibration (`layer4_calibrate_surface.py:168-172`). Reading of the code: anyone given the strikes can recover the five true ATM vols exactly, while the ATM quote itself carries noise. That is the leak L41 describes. D38's least-squares optimiser does not use that information; an exact Bayesian observer who knew how the grid was built would.
- D38's noise ensemble, the source of L18's scatter figures, ran on the leaner five-strike grid at `N_riccati=900, n_nodes=100`, not on the 35-quote grid (`layer4_calibrate_surface.py:37`, `:223`, `:231`).
- The D55 probe already used option 1's convention: "strikes standardised on each point's own ATM vol" (ROADMAP L1047–1049). The 1.6 s figure is therefore a cost for the self-standardised observable.
- P1's live surfaces sit on fixed, forward-normalised absolute strikes (`deribit_surface.py:19-20`). "Further from P1" (L44) is therefore relative to P1's synthetic study, not to its Deribit study.
- v1's reason for using P1's grid: "Same strikes and maturities, so P5's flat direction is P1's flat direction" (v1 L110). Either option changes the map, so this no longer holds by construction.

**What the literature does.**

- [BHL] use option 1's convention for rough Heston: a surface indexed by maturity and standardised moneyness z = k / (σ_ATM(T)·√T), nine z values by seven maturities, 63 quotes. Their stated reasons are that the normalisation reduces scale differences across maturities, improves the conditioning of the inverse map, and avoids spending features on contracts that are numerically negligible when volatility is low (Section 5.1.1).
- [HMT] learn the map from parameters to implied vols on a fixed grid of 11 strikes by 8 maturities, the same for every parameter draw, with forward variance in [0.01, 0.16] (Section 3.1; Section 4.1.1).
- [BHMST] use the same fixed grid and set out the alternative, a pointwise network that takes strike and maturity as inputs. They report lower accuracy at short maturities and deep out-of-the-money strikes (Sections 3.2.1, 5.1).
- [BS] use the pointwise form, with moneyness and maturity sampled from a density estimated from SPX quotes (Sections 3.1, 4.1).

**Arithmetic on recorded numbers.**

- With σ_ATM ≈ √ξ₀ (the repository's own shorthand, `calibrate_btc.py:102`), the outermost standardised point reaches absolute log-moneyness of about ±0.020 at ξ₀ = 0.001, T = 0.10 and about ±1.414 at ξ₀ = 0.25, T = 2.00: a factor of about 71 across the box.
- A fixed grid that is ±2 standard deviations at the D38 truth (σ ≈ 0.20) would be about ±12.6 standard deviations at ξ₀ = 0.001 and ±0.8 at ξ₀ = 0.25. The draft ties option 2 to a narrowed ξ₀ "so every quote is priceable" (L44) and does not say how narrow.
- Riccati solves per 35-quote surface, counted from the code: 15 if one cache per maturity is shared between the ATM anchor and the strikes (3 per maturity), against 105 to 120 without a cache. How the D55 probe handled the anchor is not recorded; its script is not in the repository.

**Options.**

| Option | What it means | What it costs |
| --- | --- | --- |
| 1. Self-standardised (the draft's proposal) | Strikes are recomputed from each θ's own ATM vol; the observer sees 35 noisy numbers indexed by (T, vu) and is not told the strikes | Stated in the draft: "Coherent, leak-free, keeps P1's box." Stated in [BHL]: used for rough Heston. Reasoning: the leak closes because the strikes are withheld; D38's Jacobian, condition numbers and H-scatter do not transfer as measured; the coordinates are exact while the ATM quote is noisy, a set-up no real observer faces |
| 2. Fixed absolute grid | One grid of absolute log-moneyness for every θ | Stated in the draft: ξ₀ must be narrowed. Stated in [HMT] and [BHMST]: used with variance from 0.01. Reasoning: the box and the test distribution change; the absolute values and the new ξ₀ range must be written in (the draft gives neither); it is closer to P1's Deribit study |
| 3. Anchor on the observed, noisy ATM quote (not in the draft) | Draw the noisy ATM quote first, set strikes from it, then observe the rest | Reasoning only: the leak closes without withholding strikes, but strikes differ from draw to draw, so G is no longer a map to a fixed-length vector and the emulator would need strike as an input, a larger problem than L46 assumes |

**Established standard.** None found. Both conventions are in use: a fixed grid in [HMT] and [BHMST], standardised moneyness in [BHL].

**To defend it, you would need to be able to say.** What exactly does my observer see, 35 numbers at coordinates that move with the true parameters or quotes at fixed strikes, and why is that the right experiment for the claim I want to make?

**Checklist boxes.** "Grid and observable convention chosen" (L168).

**Not confirmed.** How much the five noise-free ATM vols would tell an observer about H, and whether H stays the flat direction under option 1: both would need pricer runs. Whether [HMT]'s strikes are moneyness with spot normalised to 1.

## 4. Emulator: form, training-point count and design (draft L52)

**The choice.** L52: "**OPEN — emulator form, training-point count and design.**" The draft makes no proposal for any of the three. It fixes only that "G̃ approximates G over the box and is built once from the pricer. The calibrators' training data and the reference posterior both use G̃" (L46).

**Decided as.** Mixed. Form and design are values now. The count can be a value now or a rule whose value a gate fixes.

**What depends on it.**

- "Q2 is exact" (L48): this holds only if one unchanged emulator feeds both the calibrators and the reference.
- Q1's scope (L49) and gate G1 (L140): the emulator's accuracy decides at which noise levels Q1 is claimed for rough Heston.
- The reference and G2 (L66, L141): the emulator's evaluation cost and smoothness feed the sampler choice and the test-set size (L92).
- The contraction c of every scored pair, which is computed on G̃ (L77–79).
- Repository placement (L152): the form decides whether the emulator needs the leaf's dependencies.

**Settle first.** Items 2 and 3, which define G's output. Item 1 and G0: "Any change is made before the emulator is built" (L37), and G0 measures the cost that bounds the count. Items 5 and 14, which together set the accuracy the emulator must reach. Item 7: a gradient-based sampler would need a differentiable emulator.

**What the repository shows.**

- No emulator or P5 code exists (D55, ROADMAP L1031–1032).
- Why v2 uses one: "The pricer cannot afford the design directly, so v2 defines the model as a frozen emulator used by both the training data and the exact reference" (D55, ROADMAP L1041–1043).
- `_cached_cf` memoises the characteristic function per maturity (`calibrate_btc.py:43-52`); `surface_model` re-solves the Riccati for every strike (`layer4_calibrate_surface.py:59-63`, `:80-84`). L50's statement is accurate as to mechanism.
- The cache gives three solves per maturity regardless of strike count (D41, ROADMAP L882). The 13× speed-up was recorded on the Deribit grid of 63 points over five maturities (D39, ROADMAP L859, L865).
- The pricer returns NaN where the Riccati overflows or the inversion fails (`layer4_calibrate_surface.py:56-57`), so an emulator's training labels can be missing at some design points unless G0 removes them.
- The default knobs were chosen for finiteness at the truth point, not for a stated accuracy: "cheapest all-finite surface knobs (35/35 at truth)" (`layer4_calibrate_surface.py:44`).
- The core depends on numpy, scipy and matplotlib only (`requirements.txt:1-3`); torch is confined to the Layer 3 environment (`requirements-layer3.txt:1-3`).

**What the literature does.**

- [HMT]: a fully connected network, parameters in and 88 grid implied vols out; 4 hidden layers of 30 nodes; 68,000 training and 12,000 test draws, uniform on a box; labels by Monte Carlo. Accuracy is reported as relative error: average well under 0.5%, maximum up to 25%, computed across the training data. No percentile of absolute error is reported (Sections 3.2.1, 3.2.2, 4.1.1).
- [BHMST]: 3 hidden layers of 30 nodes; four model inputs, the same number as P5's; training set 34,000, test set 6,000; uniform sampling. They note that deterministic grids in parameter space might improve training but were not compared (Sections 3.2.1, 4.1, 5.1; Remark 4).
- [BS]: a pointwise network of 4 hidden layers of 4,096 nodes and about 10⁶ samples; truncated normal marginals for rough Bergomi (Table 1; Sections 4.1, 4.2).
- [BHL] do not emulate the vanilla surface: they simulate 500,000 parameter–surface pairs directly. Their Remark 3.1 notes that in their setting the random design is the prior that defines the joint law being learned (Remark 3.1; Section 5.2).
- [SWMW] model the deterministic output of an expensive code as a realisation of a stochastic process, which gives a statistical basis for choosing the runs, a cheaper predictor and estimates of its uncertainty (abstract). The abstract does not use the term Gaussian process; the listing's keywords include kriging.
- [KOH] present a Bayesian calibration whose predictions allow for all sources of uncertainty and attempt to correct for model inadequacy (abstract). The abstract does not mention emulator error; reading this as the alternative to gating on it, namely carrying it explicitly, is this sheet's reasoning.
- [GGMM]: tensorised Chebyshev interpolation in the parameter space, with explicit error bounds, applied to European options under affine models (abstract).
- [SciPy-qmc]: Sobol' points keep their balance properties for sample sizes n = 2^m; a Latin hypercube places one point in each of n strata of every margin.
- [SPAJH], outside finance: the sampling scheme mattered less than the number of training points, in a problem with many parameters (Section 4.1).

**Arithmetic on recorded numbers.**

- At 1.6 s per surface, one core: 10,000 surfaces cost about 4.4 hours; 40,000 (the size of [BHMST]'s design) about 18 hours; 500,000 (the size of [BHL]'s) about 222 hours. This assumes `N_riccati` 1000 everywhere, which D55 records as insufficient at 9 of 24 box points; at N = 2000 or 4000 everywhere the figures are 4 and 16 times larger.
- On the P5 grid the cache should save a factor of about 7 to 8 in Riccati solves, not 13; the 13× belongs to the denser Deribit grid.
- A tensor grid of n nodes per axis in four dimensions has n⁴ points: 10,000 at n = 10, 65,536 at n = 16.

**Options for the form.**

| Option | What it costs |
| --- | --- |
| Feed-forward network, parameters in and all grid quotes out ([HMT], [BHMST]) | Tens of thousands of labelled surfaces were used. It gives no error estimate of its own, so G1 is the only check. Published accuracy summaries are relative and over training data. In torch it lives in the leaf |
| Gaussian-process emulator (for modelling code output as a stochastic process: [SWMW]) | Returns a prediction uncertainty, a second check beside G1. 35 outputs need either 35 fits or a reduction step. Whether a design small enough for it reaches G1's threshold at s = 0.1 is unknown |
| Tensorised polynomial interpolation ([GGMM]) | The design is fixed by the form: the nodes on each axis are cos(πk/N), k = 0, …, N, so the tensor grid includes the box edges and corners ([GGMM], Section 2.1, Eq. 2.10). D39 records the pricer overflowing at the rough, high-ν, long-T corner (ROADMAP L864); D55 records at least one non-finite maturity at 9 of 24 uniform points, without coordinates (ROADMAP L1051–1053). So it leans hardest on G0. Whether rough Heston implied vols have the analytic extension that [GGMM]'s error bound assumes (Section 2.2, Theorem 2.2) over this box is not established |
| Pointwise network with strike and maturity as inputs ([BS]) | About a million samples. Needed only if strikes vary from draw to draw |

**Options for the design.**

| Option | What it costs |
| --- | --- |
| Independent uniform draws on the box ([HMT], [BHMST], [BHL]) | Matches the test distribution (L91) and serves priors A and B equally. Random gaps leave some regions thinner |
| Space-filling: Latin hypercube or scrambled Sobol' ([SciPy-qmc]) | No new dependency. Sobol' needs n = 2^m |
| Concentrated on a prior ([BS]) | P5 has two priors and a uniform test set; a design concentrated on either would make the emulator more accurate for one prior's calibrators than the other's, an asymmetry in the measures S and E |

**Options for the count.**

| Option | What it costs |
| --- | --- |
| A number fixed before freezing | Leaves no room to tune against G1. Chosen before G0 has measured cost or fixed the box. "Built once" leaves no remedy if G1 then fails at a noise level, other than dropping the rough Heston claim there |
| A rule whose value a gate fixes | Consistent with how the draft treats the test-set and training-set sizes. A rule that takes the smallest count passing G1 turns G1's held-out points into a tuning target, so a separate final held-out set and a stop rule would be needed |

Consequences not attributed to a source are reasoning from the draft's definitions.

**Established standard.** None found for the form or the count. For the design, one documented convention: [SciPy-qmc] warns that a Sobol' sample whose size is not a power of two no longer has the sequence's balance properties.

**To defend it, you would need to be able to say.** Why should a reader accept that this emulator, in this form, built from this many pricer runs laid out this way, is the rough Heston pricer wherever my conclusions about H depend on it, including the corners of the box, and what in the protocol stops it being changed once the reference or the calibrators exist?

**Checklist boxes.** "Emulator form and gate declared" (L170).

**Not confirmed.** A rule of thumb for Gaussian-process design size (Loeppky, Sacks and Welch 2009): the publisher page was blocked. Whether any form reaches G1's threshold over this box: unknowable without building one.

## 5. Noise: the three levels (draft L54)

**The choice.** L54: "**Noise.** Independent Gaussian on each quote, as a design factor: s in {0.1, 0.3, 1.0} vol point. 0.1 is P1's level; 1.0 is the size of the live misfit. Noise level is how the design reaches the flat regime, by construction. Well-specified noise only; no claim about real quotes. **OPEN — the three levels.**"

**Decided as.** A value now.

**What depends on it.**

- Gate G1 (L140): the lowest level sets the emulator accuracy needed.
- The reference's cost: one posterior per test θ per level (L66).
- The number of networks: "Each trained per prior and per noise level" (L68).
- Bin membership (L77) and, through bin occupancy, every row of the hypotheses table and the 10% floor.
- v1 fixed one level, 0.1 vol point, as a ticked DECISION (v1 L106, L203). Three levels is a departure from a registered choice, which v1 L188 makes a dated amendment.

**Settle first.** Items 2 and 3: the only repository numbers on how noise degrades H on a surface are on a 25-quote grid with anchored strikes (D38); the others are single-smile, D37's on ten quotes and D38's baseline on five, both at T = 1 (`layer4_calibrate.py:42`, `:277`; `layer4_calibrate_surface.py:37`, `:230-231`). Item 6: "identifies H well" (L22) is relative to prior A's variance. Items 9 and 11: the levels are the device that fills the bins.

**What the repository shows.**

- D37, single smile: "62% spread at 0.1pp noise → 153% at 0.5pp" (ROADMAP L839). This is the standard deviation of a least-squares optimiser's fitted H across noisy re-fits, divided by the true H. It is optimiser scatter, not a posterior.
- D38, surface: "σ=0.1pp **~106%→~10% (~10×)**; σ=0.3pp ~190%→~29% (~6.5×); σ=0.5pp ~194%→~48% (~4×)" (ROADMAP L850), each pair running from the run's own single-smile baseline to the surface; the surface figures are about 0.010, 0.029 and 0.048 at H = 0.10. Ten draws, one parameter point, 25 quotes, a near-truth start, and the same ten noise seeds reused at every level (`layer4_calibrate_surface.py:223`, `:229`, `:231`, `:175-177`). The ROADMAP itself calls the figures approximate (L854).
- Both noise experiments stop at 0.5 vol point: `sigmas=(0.001, 0.003, 0.005)` (`layer4_calibrate.py:274`; `layer4_calibrate_surface.py:223`). **No synthetic measurement exists at 1.0.**
- Live fit errors: 0.888 vol point for BTC on the reduced span (D39, ROADMAP L860); about 1.1 on the full span (`OVERLEAF/P1/calibration_paper_revised.tex:283-284`); 1.63 for ETH (D42, ROADMAP L892). These are residuals of a fitted model, structured in the put wing and long tenor (ROADMAP L893), not independent quote noise.
- The repository's live-data filter admits bid–ask widths up to 5 vol points (`deribit_surface.py:70-71`).
- D55 records why noise became a design factor (ROADMAP L1039–1041). No reason is recorded for the middle value 0.3, or for leaving out 0.5.
- v1 held open a "second, tenfold-cleaner noise level" (v1 L112). v2 adds noisier levels and does not mention the cleaner one.

**What the literature does.**

- [HMT] remark that the spread on the most liquid options under a year is about 0.2% in implied-vol terms; the remark carries no citation (Section 4.1).
- [BHMST] use a Gaussian likelihood with per-quote error standard deviations; on SPX data those are proxied by a fraction of each quote's bid–ask spread (Sections 4.2.1, 5.3).
- [BHL] impose no independent per-quote noise: their surface errors come from the simulator and are dependent across strikes and maturities. They list an observation model for market surfaces as future work (Section 5.1.1; conclusion).
- [EGM] report Deribit BTC spreads in ticks, not vol points (Section 2.2).

**Arithmetic on recorded numbers.**

- Units: the code's 0.001 is 0.1 vol point. The draft never defines "vol point".
- If D38's scatter standard deviations were posterior standard deviations under prior A, the contraction would be 0.965 at 0.1 vol point (Identified), 0.708 at 0.3 (Partial) and 0.200 at 0.5 (Flat). Read instead as likelihood standard deviations, they give 0.966, 0.774 and 0.556, which puts 0.5 in Partial. **Indicative only**: optimiser scatter at one parameter point on a 25-quote anchored grid is not a posterior standard deviation, and the two readings disagree about the bin at 0.5. Carrying the ten-draw sampling error through the first reading (a 95% chi-square interval on the variance with 9 degrees of freedom, the ten fitted values taken as independent normal draws and the scatter as the code computes it, with divisor n, `layer4_calibrate_surface.py:192`) gives ranges of about [0.87, 0.98], [−0.08, 0.85] and [−1.96, 0.58].
- D38's scatter is close to proportional to s, about 0.097 in H per vol point. A straight-line extension to s = 1.0 gives about 0.10, larger than prior A's standard deviation. That is speculative beyond 0.5: no measurement exists, and the proportionality partly reflects the reuse of the same noise vectors.
- Networks: with K levels there are 2 calibrators × 2 priors × K configurations. K = 3 gives 12 configurations and 60 training runs at five seeds; each added level adds 20 runs, 4 G3 checks and one more reference posterior per test θ.
- 1.0 lies between the two BTC fit errors and below the ETH one; it equals none of them.
- The noise is absolute. With ATM vol roughly √ξ₀, s = 1.0 is about 32% of the vol level at ξ₀ = 0.001 and 2% at ξ₀ = 0.25.

**Options.**

| Option | What it costs |
| --- | --- |
| {0.1, 0.3, 1.0} (the draft) | 12 configurations, 60 training runs. 0.1 and 0.3 each have one recorded reference point (D38); 1.0 has none. G1 needs e ≤ 0.02 vol point at the lowest level |
| A single level, 0.1 (v1's registered choice) | The draft's own statement: "a single noise level would leave little flat regime to study" (L22) |
| {0.1, 0.3, 0.5}, the levels D37 and D38 measured | Every level has a recorded reference point. The link to "the size of the live misfit" is lost. On the indicative arithmetic the top level sits near the Partial/Flat boundary, so the Flat bin is the one at risk of the 10% floor |
| Two levels, {0.1, 1.0} | 8 configurations and 40 runs. The middle level is the only one the indicative arithmetic places in the Partial bin, which carries H-b |
| Four levels, {0.1, 0.3, 0.5, 1.0} | 16 configurations and 80 runs; a third more reference cost for the same number of θ |
| Raise the lowest level | G1's requirement relaxes in proportion. Continuity with v1's registered 0.1 is lost and the Identified bin becomes the one at risk |
| Fix a rule now, the values at a gate | The levels would rest on pilot reference posteriors on the adopted grid, not on one D38 point. Pilots must then compute posteriors at candidate levels. No network output is involved |
| Noise scaled to the quote, as [BHMST] do on market data | Removes the sixteen-fold variation in relative noise across the ξ₀ range. Needs a spread model the draft does not have and goes beyond "no claim about real quotes" |

The consequences are reasoning from the draft's definitions and the recorded numbers.

**Established standard.** None found for the noise levels of a synthetic identifiability study.

**To defend it, you would need to be able to say.** What is each of the three levels there to do in terms of the Identified, Partial and Flat bins, and what do I actually know, beyond ten optimiser fits at one parameter point on a 25-quote grid and no measurement at all at 1.0, that tells me these values will populate all three bins and that the emulator can pass G1 at the lowest one?

**Checklist boxes.** "Noise levels accepted or changed" (L169).

**Not confirmed.** A typical bid–ask width in vol points for index or crypto options from a primary source: not found. The unrounded D38 scatter values: no output log is tracked.

## 6. Priors: the Beta parameters (draft L64)

**The choice.** L60–64: "| H | 0.02 + 0.46 · Beta(2.5, 9) | 0.02 + 0.46 · Beta(7.5, 4) |", "| Mean of H | 0.12 | 0.32 |", "| SD of H | 0.054 | 0.062 |", "**OPEN — the Beta parameters.**"

**Decided as.** A value now.

**What depends on it.**

- "identifies H well" is judged against a prior of standard deviation 0.054, which is prior A's (L22, L62), and the contraction c that defines the Identified bin uses prior A's variance (L77–83).
- The reference's importance weights: "posteriors under A and B by importance reweighting" (L66).
- Every measure: d, S, E and m (L105–108). The 0.20 in S is the gap between the two means.
- Calibrator training and G3, which run per prior (L68, L142).

**Settle first.** Item 1: the formula hard-codes the H range [0.02, 0.48], which L37 leaves open to shrink. v1 fixed the support, the scaled-Beta family and the means 0.12 and 0.32 as ticked DECISION items (v1 L89–98, L201). Choosing the Beta parameters within those means completes v1; changing the means or the family changes a registered commitment.

**What the repository shows.**

- v1 registered "two training priors over H that differ only in where they put their mass" (v1 L89), with support [0.02, 0.48] and means 0.12 and 0.32 (v1 L91–96). It did not fix the Beta parameters or the standard deviations.
- v1's reason for the gap: "It is wide enough that a prior-driven answer cannot hide inside noise, and both means sit inside the range P1 found plausible" (v1 L98). v2 does not repeat this sentence.
- Why Beta(2.5, 9) and Beta(7.5, 4), with a + b = 11.5 in both, is not recorded in the draft, v1, D54 or D55.
- v2 uses prior A alone for the bins and for the interval measure (L77, L108). No reason for A rather than B is recorded.
- H values recorded elsewhere: the synthetic truth is 0.10 (`layer4_calibrate_surface.py:38`); live BTC and ETH fits rail to 0.02 (ROADMAP L860, L892).

**What the literature does.**

- [GVS+] describe refitting the model under several priors as the plainest route to seeing what the priors contribute, and mention importance sampling from one posterior to another when the two are close enough. They give no rule for how far apart alternative priors should be (Section 6.3).
- [KPBV] perturb one prior by raising it to a power near 1 and estimate the effect by importance sampling. Their diagnostic threshold of 0.05 is on a local sensitivity measure: the rate at which a cumulative Jensen–Shannon distance between the base and perturbed posteriors changes with the logarithm of the power, a distance they take from earlier work. It is a local perturbation of one prior, not a method for choosing a pair (Sections 2.1, 2.3, 2.4.2–2.4.4, 2.5).
- [BHL] train under a uniform simulation prior and obtain a posterior under another prior, with support inside the training support, by reweighting with the ratio of prior densities. They contrast this with a directly trained inverse estimator, which they describe as tied to its training design (Section 3.1).
- [EOS+] train one network over a family of prior specifications so that sensitivity can be read at inference time: an alternative to a separate network per prior (abstract).

**Arithmetic on the draft's numbers.**

- The table is correct. Prior A: mean 0.1200, standard deviation 0.0537. Prior B: mean 0.3200, standard deviation 0.0620. The gap is exactly 0.20, which is 3.7 standard deviations of A and 3.2 of B. The variance ratio B/A is 4/3.
- Where the mass sits. A: 5th, 50th and 95th percentiles 0.046, 0.112, 0.220; P(H < 0.10) = 0.41, P(H ≥ 0.25) = 0.020. B: 0.211, 0.324, 0.415; P(H < 0.10) = 0.0002, P(H ≥ 0.25) = 0.86.
- Overlap. The densities cross at H = 0.214; the overlap coefficient is 0.114. The two central 90% intervals overlap only on [0.211, 0.220].
- Against the uniform test set (L91): half the test θ have H ≥ 0.25, where prior A has 2.0% of its mass. P_A(H > 0.40) = 2.8e-6, while 17% of test H exceeds 0.40. For B, 17% of test H is below 0.10, where P_B = 1.7e-4.
- Importance weights from the uniform prior. Under A the weight is below 1% of its uniform value over 31% of the H range (21% for B). For a completely flat surface the effective-sample fraction after reweighting would be 0.39 for A and 0.48 for B. That is one idealised case; real figures depend on the posterior.
- Holding the means and changing a + b: at 5.75 the standard deviations are 0.073 and 0.084 and the overlap 0.244; at 23 they are 0.039 and 0.045 and the overlap 0.029.
- Equal-width pairs with the registered means exist, for example Beta(2.5, 9) with Beta(10.22, 5.45), both 0.0537.
- With unequal prior variances the reference's own shift ratio depends on where the data sit, not on contraction alone (Gaussian approximation): at c = 0.90 it runs from about 0.07 to 0.12 as the likelihood's centre moves across the range of H. With equal variances it would be 1 − c.

**Options.**

| Option | What it costs |
| --- | --- |
| Beta(2.5, 9) and Beta(7.5, 4) (the draft) | Keeps v1's support, family and means and adds the first explicit shapes. The widths differ, so the pair differs in spread as well as location; whether that still meets v1's "differ only in where they put their mass" (v1 L89) is for the author to say. Half the uniform test set lies where prior A has 2% of its mass |
| Same means, equal standard deviations | Matches v1's "differ only in where they put their mass". In the Gaussian approximation the reference's shift ratio becomes a function of contraction alone. Loses the common a + b and the round parameters |
| Same means, wider priors (smaller a + b) | More training mass where the test set lives and gentler importance weights. More overlap; the 0.20 gap is fewer prior standard deviations; with a larger prior variance the same posterior counts as more contracted, moving pairs towards Identified |
| Same means, narrower priors (larger a + b) | Cleaner separation. More of the test set falls where a prior has essentially no mass; importance weights become more extreme; pairs move towards Flat |
| A mirror-image pair, Beta(2.5, 9) and Beta(9, 2.5) | Symmetric with equal widths. It changes v1's registered mean of 0.32 and the 0.20 yardstick, so it amends a confirmed v1 decision |
| Perturb one prior instead of swapping two ([KPBV]) | A published diagnostic exists. It would replace the registered prior-swap design and the measures S and E |
| One network trained on the uniform prior, reweighted to A and B ([BHL]) | For the ratio estimator only, no network is trained under a prior with almost no mass over half the test range. It changes what is tested: the draft's question is about calibrators trained under A and under B |

The consequences are reasoning from the draft's definitions and the arithmetic above.

**Established standard.** None found for choosing a pair of priors for a prior-swap experiment.

**To defend it, you would need to be able to say.** v1 fixed the support and the two means; what fixes a + b = 11.5, that is, why are these the widths, and unequal, and am I content that prior A, the one that also defines every bin, puts under 2% of its mass on the half of the test range above H = 0.25?

**Checklist boxes.** "Beta parameters accepted or changed" (L171).

**Not confirmed.** v1's "range P1 found plausible": no such range for H was located in the P1 paper by keyword search. Whether any published work prescribes the separation or width of priors for a prior-swap test: none found, and the search was not exhaustive.

## 7. The exact reference: sampler, chains, length, R-hat ceiling, effective-sample floor (draft L66)

**The choice.** L66: "**The exact reference.** Per scored surface and noise level: MCMC on G̃ under the uniform box prior; posteriors under A and B by importance reweighting, with a direct re-run where effective sample size falls below the floor. **OPEN — sampler, chains, length, R-hat ceiling, effective-sample floor.**"

**Decided as.** Mixed. The sampler, the number of chains, the R-hat ceiling, and the definition and floor of each effective sample size can be values now. The length can be a number now, chosen with no autocorrelation time yet observed, or a rule now whose numbers are observed at G2.

**What depends on it.**

- Bin membership: c needs the posterior variance of H under prior A, accurately near the cut-offs (L77–85).
- Stage 1, produced "From the reference alone" (L97), and its stop rule (L99).
- H-a, through "d = Ĥ_net,A − Ĥ_exact,A" (L105): the posterior mean under A, to an error small against 0.01.
- H-b, through S and E (L106–107): the posterior mean under B as well. An error δ in the difference of the two means moves E by δ/0.20.
- H-d, through m (L108): the probability of an arbitrary interval under prior A, so the tails of the marginal.
- H-c uses the reference only for bin membership; S_net and the seed floor (L109) are network-only.
- The dose-response (L132), the claim "Q2 is exact" (L48) and the null's "within tolerance" (L126).
- Gate G2 tests these settings, and their measured cost fixes the test-set size (L92).

**Settle first.** Item 4: whether G̃ has gradients decides which samplers are available, and its cost per evaluation decides the affordable length. Item 6: under the drafted scheme the importance weights are the two scaled-Beta densities. Items 1, 2, 3 and 5: they define the domain, the likelihood and the posterior's width. Items 9 and 12: the cut-offs and the hypothesis numbers set how accurate the reference has to be. G2 tests the settings chosen here (item 15). The draft has the reference's sampler and acceptance criteria declared before freezing (the box "Reference sampler and acceptance criteria declared" under "Before freezing", L165, L172), when OPEN items become commitments (L11); it does not otherwise state that the settings are fixed before G2 is run.

**What the repository shows.**

- No sampler, posterior or P5 code exists: "**No code yet.** At the time of the amendment no P5 code, pilot, network or surrogate exists in the repository" (ROADMAP L1031–1032, D55); and of the 3 October pricer probe, "No Jacobian, Fisher information, Cramér–Rao bound, posterior or calibration was computed" (ROADMAP L1056–1057).
- The core environment has no sampling library: `requirements.txt:1-3` lists numpy, scipy and matplotlib. Any sampling library would be a new dependency.
- Under all three priors the other parameters are uniform: "Same support; all other parameters uniform on the box" (L56). So the weight from the uniform-prior posterior to A or to B is a function of H alone.
- Test θ is "uniform on the box" (L91), so true H is uniform on [0.02, 0.48].
- Three of the four reference quantities (c, Ĥ_exact,A and m) are defined under prior A and the fourth (Ĥ_exact,B) under prior B (L77–79, L105–108). Nothing that is scored uses the uniform-prior posterior which L66 samples, so every scored reference quantity passes through the reweighting or the re-run.
- The H–ν figures recorded in the ROADMAP: D37 "degenerate **H~ν (−0.82 full / −0.92 put)**" (ROADMAP L838); D38 "H~ν stays correlated (−0.85)" (ROADMAP L849); D39, live BTC on the reduced span, "corr(H,ν)=−0.96" (ROADMAP L861); D41, live BTC, "corr(H,ν)=−0.876" on the full span and "corr(H,ν)=−0.911" with the one-year tenor dropped (ROADMAP L883); D42, live ETH, "corr(H,ν)=−0.855" (ROADMAP L889). The code computes them as cosines between Jacobian columns, `corr = G / np.outer(d, d)` with `G = J.T @ J` (`layer4_calibrate.py:200-202`; `layer4_calibrate_surface.py:147-149`). They are not posterior correlations.
- The shape evidence for the live fits is local. Of D41's Jacobian comparison the ROADMAP says "This is a LOCAL diagnostic at the fitted optimum θ̂", with the global check "RESERVED for the calibration-paper update" (ROADMAP L887). The P1 paper says "the diagnostics reported are local, evaluated at the fitted optimum", names the cross-run instability among them (D39's two runs that "landed at DIFFERENT (ν,ρ)=(0.54,−0.46) vs (0.64,−0.37) at the SAME ~0.9pp RMSE", ROADMAP L861), and says a global confirmation "is not claimed here" (`OVERLEAF/P1/calibration_paper_revised.tex:444-449`). D37's figure is "JᵀJ at truth" (ROADMAP L838). No posterior shape is recorded anywhere.
- Live fits sit on the lower bound of H: BTC "H=0.0201 (railed to LB)" (ROADMAP L860), railing "in 100% of 24 bootstrap resamples" (ROADMAP L861); "H=0.02 is near the CF-overflow floor" (ROADMAP L864); ETH "H=0.0200 (RAILED)" (ROADMAP L892). These are fits of the model to live data, not well-specified synthetic surfaces.
- v1's account of the precedent: "An MCMC sampler draws the posterior through the network surrogate, under a fixed uniform prior on a truncated box" (v1 L29).
- The pricer took "about 1.6 s on one core" per 35-quote surface (L20).

**What the literature does.**

Convergence of independent chains:

- [VGSCB] recommend using a sample only when R-hat is below 1.01, where R-hat is the larger of the rank-normalised split and the rank-normalised folded-split versions; running at least four chains; and requiring a rank-normalised effective sample size above 400, with the bulk and tail versions both examined. They present these as first-level checks to adapt to the application, and say that in the end the Monte Carlo standard errors of the quantities of interest must be judged small enough (Section 2; Section 3.2; Section 4.2).
- [VGSCB] report that R-hat can fall below 1.1 well before convergence. At 1.01 the split version detects one chain displaced by a third of the marginal standard deviation; at 1.1 only a displacement of about one standard deviation (Section 2; Appendix A).
- [VGSCB] define tail-ESS on the 5% and 95% quantiles, observe that tail quantities are often estimated less efficiently than the mean, and give a method for the Monte Carlo standard error of a quantile (Sections 4.3–4.4). Reading this as bearing on m, the exact posterior probability of the network's 90% interval (L108), is this sheet's reasoning, not a statement in [VGSCB].
- [Stan-RM] attributes the 1.01 threshold to [VGSCB], allows other thresholds depending on the use, and gives the standard error of a posterior mean as the posterior standard deviation over the square root of the effective sample size ("Posterior Analysis"). [Stan-diag] recommends at least four chains, R-hat below 1.01 for results to be trusted and R-hat below 1.1 as often sufficient early in a workflow ("R-hat"), and bulk-ESS above 100 times the number of chains ("Bulk and Tail ESS").
- [VK] state a one-to-one relation between the Gelman–Rubin statistic and effective sample size, which is a route to deriving a ceiling from a chosen effective sample size (abstract only).

The reweighting step:

- [PSIS] give a diagnostic for importance weights: estimates are likely to be unreliable when the fitted Pareto shape exceeds min(1 − 1/log₁₀ S, 0.7) for S draws. They give Kong's 1/Σw̃² as a generic effective sample size that does not depend on the quantity estimated, a function-specific version, and, for draws that come from MCMC, a variance that carries a further factor for the chain's autocorrelation (Algorithm 1; Section 2, Eqs. 5–8; Section 3.1). [loo] documents how the shape is read.
- [EMR] describe the widely used approximation of the effective sample size in importance sampling, whose derivation they place partly in Kong (1992), as resting on assumptions and approximations that make it questionable as an estimate (abstract only). The abstract gives no formula; [PSIS] give Kong's estimate as 1/Σw̃² and cite [EMR] beside it (Section 2.2, Eq. 8).
- [KPBV] estimate properties of posteriors under a power-scaled prior by importance-weighting the base posterior draws, using Pareto-smoothed importance sampling, which they describe as self-diagnosing (Section 2.3).
- [BHL] describe changing the prior at inference time by reweighting posterior samples with the ratio of the new prior to the training prior, for a new prior supported inside the training support (Section 3.1, Eq. 3.7). A search of their text found no diagnostic for that step.

Samplers and precedents:

- [FHLG]: the affine-invariant ensemble sampler's update uses only ratios of the target density, and the method is designed to be insensitive to linear covariances among parameters (Section 1; Section 2, Eq. 9, Algorithm 2). The paper's diagnostic is the integrated autocorrelation time; it suggests a large number of walkers and warns that performance deteriorates on multi-modal targets (Sections 3–4). [emcee-docs] say the Gelman–Rubin statistic should not be computed across the walkers of one ensemble and that an autocorrelation estimate should be trusted only for chains longer than about 50 times it ("Autocorrelation analysis & convergence"), that bounds are imposed by returning −∞ outside them ("FAQ", Parameter limits), and that the default move needs at least twice as many walkers as dimensions ("Moves", RedBlueMove).
- [HG] (abstract only) and [Stan-RM] ("MCMC Sampling"): NUTS takes steps informed by gradients of the log-density. Stan transforms each bounded parameter to an unconstrained scale (log-odds for a two-sided bound) and can use a dense metric to compensate for linear posterior correlations ([Stan-RM], "Constraint Transforms"; "MCMC Sampling").
- [dynesty]: nested sampling specifies the prior as a transform from the unit cube ("Getting Started", Prior Transforms), requires only a log-likelihood, a prior transform and the number of dimensions, and by default picks uniform, random-walk or slice sampling (Initialization; Sampling Options), returns weighted samples with an evidence estimate (Results), and is controlled by the number of live points and a stopping tolerance (Live Points; Running Internally). The three pages read do not mention R-hat or the Gelman–Rubin statistic; the FAQ describes dynesty's samples as nominally independent and groups its random-walk and slice methods with non-gradient samplers ("FAQ", Sampling Questions).
- [LBGGM] generated 10,000 reference posterior samples for each observation of a simulation-based-inference benchmark. Where an analytic posterior exists it was used, with a rejection step where the prior is bounded; one task has a custom scheme built on the model equations; and for three tasks (SLCP, SIR, Lotka–Volterra) the samples come from rejection sampling with a proposal fitted to preliminary draws, used in place of relying on MCMC directly, the reasons given being MCMC's difficulty with multi-modal posteriors and bias from correlated chains. They ran the scheme twice for every task and observation and found the two sets indistinguishable by a classifier two-sample test (Section 2.4; Appendix B.1).
- [BHMST] cite emcee for their Bayesian experiment. A full-text search of the arXiv version found no statement of walkers, length, burn-in or any convergence diagnostic (Sections 4.2.1, 5.3). [BHL] use an MCMC-free sampler and validate by coverage on held-out simulations (Sections 3.1, 5.2).

**Arithmetic on the draft's numbers.**

- Reweighting when the likelihood is flat: the effective fraction is 0.393 under A and 0.483 under B. A reweighted effective size of 400 then needs about 1,017 (A) or 828 (B) effective draws from the uniform chain. This uses Kong's formula, which ignores autocorrelation.
- The flat case is not the worst. With assumed Gaussian posteriors of D38's three widths placed at other centres, a uniform-prior posterior near H = 0.45 keeps 2% to 9% of its draws when reweighted to A, and one near H = 0.03 to 0.05 keeps 3% to 18% when reweighted to B. The shapes are assumptions, not measurements; the calculation shows the direction of the effect only.
- Test θ against the priors: about 35% of test surfaces have true H ≥ 0.32, where prior A's density is under 1% of its peak, and 17% have true H ≤ 0.10, where prior B's is 0.24% of its peak or less. The draft treats the re-run as an exception; on these figures it could be routine. True H is not the posterior, so this is indicative.
- Reweighting directly between A and B has unbounded weights, proportional to x⁵(1 − x)⁻⁵ with x the position in the H range, and infinite variance in the flat case. From the uniform prior the weights are bounded.
- Both priors have zero density at H = 0.02 and at H = 0.48, because all four shape parameters exceed one.
- What an effective size of 400 for the prior-A posterior gives:

| Quantity | Monte Carlo error at 400 | Set against |
| --- | --- | --- |
| Mean of H in the Identified bin (posterior sd ≤ 0.017) | Standard error ≤ 0.00085; a floor of about 0.0006 under the median of abs(d), for normal Monte Carlo error | H-a's 0.01 |
| S_exact across the Partial bin | Per-pair sd about 0.006 at c = 0.9 and 0.014 at c = 0.5, on the Gaussian approximation, if each of the two means had an effective size of 400 and the two erred independently | H-b's 0.05 and 0.15 |
| c | About ±0.007 at c = 0.9 and ±0.035 at c = 0.5 | The cut-offs |
| m near 0.90 | Per-pair sd about 0.015 | H-d's narrowest margin, 0.02 |

- An effective size of about 131 holds the floor under abs(d) at 0.001 in the Identified bin; 3,600 holds the per-pair error in m at 0.005. Zero-mean per-pair errors widen the bootstrap interval of a median without shifting it to first order; a systematic error in the reference does not average out.
- Ridge shape: the square roots of recorded condition numbers give an axis ratio of about 92:1 at D38's one point (8.5e3, ROADMAP L849), 787:1 for D37's single smile (6.2e5, ROADMAP L838) and 99:1 for D41's full-span live BTC fit (9.83e3, ROADMAP L883); D39's reduced-span fit recorded 5.2e4 (ROADMAP L861), about 228:1. Local, in raw parameter units, and not on v2's observable.
- On the pricer, 1.6 s × 10⁴ likelihood evaluations is 4.4 hours and × 10⁶ is 444 hours, per posterior on one core.

**Options.**

| Choice | Option | What it costs |
| --- | --- | --- |
| Target | Uniform prior, reweight to A and B, re-run below a floor (the draft) | One run per pair serves both priors, and the weights are bounded. It needs a reweighting diagnostic as well as the chain diagnostics, and definitions of "effective sample size" and "direct re-run". On the arithmetic the re-run could be routine |
| Target | Sample directly under A and under B for every pair | No reweighting and no fallback rule. Roughly twice the sampling, which lowers the affordable test-set size. The target density vanishes at both ends of the H range |
| Target | Sample under one Beta prior and reweight to the other | Unbounded weights, the situation [PSIS] was designed to diagnose |
| Target | A reference that does not rest on chain convergence: rejection sampling from a fitted proposal, as [LBGGM] used for three of their tasks, nested sampling, or a grid | An R-hat ceiling and a length no longer apply; accuracy is set by other controls, and the OPEN list would need rewording. A full-box grid for every scored pair is out of reach (item 15) |
| Sampler | Affine-invariant ensemble (emcee) | No gradients, so any emulator form works; the sampler [BHMST] cite. Its documented diagnostic is the autocorrelation time, and an R-hat needs several independent ensembles |
| Sampler | NUTS | Needs a differentiable emulator (item 4). The published R-hat and ESS conventions apply as written. Bounds are handled by transformation |
| Sampler | Nested sampling (dynesty) | No gradients; the box is native. No R-hat and no chain length. A Beta prior could be built into the prior transform, removing the reweighting at one run per prior |
| Chains | At least four | The published default. For an ensemble sampler the counterpart would be four or more independent ensembles |
| R-hat ceiling | 1.01 on the rank-normalised maximum | The current published recommendation. Which quantities are monitored (H alone, all four parameters, the log-posterior) is a separate choice the draft does not state |
| R-hat ceiling | 1.1 or another value | [VGSCB] report that 1.1 can be passed well before convergence |
| Effective-sample floor, chain | 400, bulk and tail | Published as the level at which the diagnostics themselves are reliable, not as a guarantee about any quantity. The table above shows what it gives |
| Effective-sample floor, chain | Derived from the errors tolerated in Ĥ_exact, c and m | Ties the floor to the numbers the reference feeds. The tolerated errors are your choice |
| Effective-sample floor, reweighted | Kong's 1/Σw̃², a function-specific version, the Pareto-shape threshold, or a combination | No published numerical floor for Kong's figure was found. With bounded weights the variance is finite, but [PSIS] report that in a finite sample the fitted shape can still exceed zero, and even one, when the bound lies far beyond the largest observed weight, and they use the shape as a diagnostic in that case (Section 3.1; Section 3.2.9; Section 3.3, Example 2). Bounded weights therefore do not rule out a large fitted shape |
| Length | One fixed length for every pair | Simple to register. Some pairs will fail the criteria |
| Length | Run until the criteria are met, with a cap | Spends compute where posteriors are hard. A rule fixed now, with its numbers observed at G2 |
| Pairs that still fail | Re-run, flag, or exclude | The draft has no rule. Excluding them removes the hardest posteriors from the scored set |

Consequences not attributed to a source are reasoning from the draft's definitions and the arithmetic above.

**Established standard.** For independent chains there is one: [VGSCB], repeated in part by [Stan-RM] (the 1.01 ceiling and the maximum rule; "Posterior Analysis") and by [Stan-diag] (the 1.01 ceiling, the maximum rule, at least four chains, and a bulk-ESS floor of 100 times the number of chains, with no number for tail-ESS; "R-hat", "Bulk and Tail ESS"). R-hat below 1.01 on the larger of the rank-normalised split and folded-split versions; at least four chains; bulk and tail effective sample size above 400; and the authors' own caveat that the Monte Carlo error of the quantities used is the final test. For an ensemble sampler the documented practice is different: autocorrelation time, chains longer than about 50 times it, and no R-hat across the walkers of one ensemble ([emcee-docs]). For the reweighting step: the Pareto-shape threshold of [PSIS]. For a numerical floor on Kong's effective sample size, and for the choice of sampler itself: none found.

**View, on the convention only.** If the sampler chosen runs independent chains, the [VGSCB] numbers are the ones a reader will expect, and a looser ceiling would need its own justification. That convention does not cover the reweighted sample, the sampler or the length, and no view is given on those.

**To defend it, you would need to be able to say.** For any one scored surface, how do I know the reference's mean of H, its variance and its probability of the network's interval are right to within an error that cannot change a verdict, under prior A and under prior B and not only under the uniform prior I sampled, and which declared number is the evidence, including for surfaces whose H lies where a prior has almost no mass or against the edge of the box?

**Checklist boxes.** "Reference sampler and acceptance criteria declared" (L172).

**Not confirmed.** Kong (1992) was not fetched; the formula and its attribution are known only through [PSIS]. Gelman and Rubin (1992): only the abstract page was reached, so the paper's own threshold wording was not read. The published version of [BHMST] was not reached, so whether it states sampler settings is unknown; v1's "fixed uniform prior on a truncated box" (v1 L29) could not be confirmed from the passages searched in the arXiv version. No source was found on how either sampler family behaves when the posterior is a ridge cut off by a hard bound. The shape of the H posterior under the v2 design is unrecorded. The emulator's cost per evaluation is unrecorded, so no wall-clock figure can be given. Slice sampling, random-walk Metropolis and sequential Monte Carlo were not examined at source.

## 8. Calibrators: architecture, optimiser, stopping rule, variant, sample count, training seeds (draft L73)

**The choice.** L73: "**OPEN — architecture, optimiser, stopping rule, ratio-estimation variant, posterior sample count, and the number of training seeds (proposed: five per calibrator, prior and noise level).**" The lines above it already fix: "Each trained per prior and per noise level" (L68); "Deep ensemble of five heteroscedastic members, Gaussian negative log-likelihood. Ĥ is the ensemble mean; the 90% interval is the central Gaussian interval at the ensemble's total variance" (L70); "Neural ratio estimator. Ĥ is the posterior mean; the 90% interval is equal-tailed" (L71).

**Decided as.** Values now. G3 forbids tuning architecture against H outputs, so these are declared before G3 runs; G3 adds only the training-set size.

**What depends on it.**

- The seed floor, "S between two networks trained on the same prior with different seeds" (L109), and through it H-c's "above 0.10 and above the seed floor" (L121). The seed count feeds a verdict.
- The bootstrap "over training seeds and, within seed, pairs" (L130): the seed count is the number of top-level units.
- m and H-d (L108, L122). The interval comes from the Gaussian head for the ensemble and from posterior draws or a density for the ratio estimator, so the variant, any sampler and the draw count are part of what m measures.
- Every measure (L105–107): each is a function of Ĥ from a trained calibrator.
- G3 (item 16), which tests whatever is declared here.
- The exploratory question "which ensemble variance component tracks c" (L133), which needs the two components kept separately.
- Item 10: every scored pair needs a ratio-estimator posterior per prior and per seed, a cost L92's rule does not mention.

**Settle first.** Items 2 and 3: they fix the network's input, 35 quotes, or 28 if T = 2.00 is dropped. Items 5 and 6: the number of networks is proportional to the number of noise levels and of priors. Item 4: once G̃ exists training data are cheap, which opens the choice between a fixed training set and fresh draws. The scope line also bears on the variant: "Out: real surfaces, misspecified noise, other models, speed, architecture search, any correction method" (L148).

**What the repository shows.**

- v1's decision on the second calibrator: "it is the method family of Brigo, Huser and Leonte, so the comparison is like-for-like. It is built on RoughVolLab's own pricer with standard simulation-based-inference tooling, not their code, and passes the same G3 gate before it is scored" (v1 L69). Neither v1 nor v2 names a tool.
- v1 on the ensemble: "Each member outputs a mean and a variance for every parameter, trained by Gaussian negative log-likelihood; the ensemble's total variance is the mean of the members' variances plus the variance of their means" (v1 L75); "**DECISION — ensemble size: five members**, the setting of Lakshminarayanan et al." (v1 L177).
- v1 on the ratio estimator: "Under its training prior it is calibrated by construction, so the standard simulation-based-calibration check should pass at home" (v1 L71). v2 does not repeat the sentence.
- v1: "They are falsification thresholds, not targets: no code may be tuned to reach them" (v1 L143).
- The reason for five training seeds is not recorded: L73 gives only the parenthesis.
- No P5 code or network exists (ROADMAP L1031–1032, D55).
- The repository's one neural-network precedent is Layer 3: a network of two hidden layers of width 32 with Tanh, Adam at learning rate 1e-3, a fixed number of epochs with no validation split (`layer3_deep_hedging.py:264-289`), and eight seeds (`layer3_deep_hedging.py:523`; ROADMAP L873).
- What that precedent recorded about convergence: "statistical significance (z=7.2) + seed-consistency (8/8) + budget-reproducibility (500ep AND 1500ep) ALL FAILED to guarantee convergence" (ROADMAP L903, D43); "more capacity without more data overfits" (ROADMAP L875, D40).
- The isolation precedent L152 points to: `requirements-layer3.txt` holds `torch>=2.12` for a separate environment, and the core tests run without torch.

**What the literature does.**

The ensemble:

- [LPB]: each network outputs a mean and a variance and is trained with the Gaussian negative log-likelihood. The ensemble is a uniformly weighted mixture, approximated for regression by one Gaussian with the mixture's mean and variance. Members are trained on the whole training set and differ by random initialisation and shuffling. Five networks is given as the recommended default. Adversarial training is listed as an optional step and is reported not to help significantly on the regression benchmarks. The regression experiments use one hidden layer of 50 ReLU units (100 for the two larger data sets), 40 epochs, batch size 100 and Adam at a fixed learning rate (Sections 2.1–2.4; Algorithm 1; Sections 3.1, 3.3; Section 4; Appendix A.1, Table 2).

Ratio-estimation variants:

- [HBL]: an amortised estimator of the likelihood-to-evidence ratio is learned and embedded in MCMC to draw posterior samples (abstract only). [DMP]: classifier-based ratio estimation and direct posterior density estimation are unified under one contrastive-learning scheme (abstract only). The description of the first as a binary classifier and of the second as multi-class is from [MWF] (abstract; Section 2) and [LBGGM] (Appendix A.7).
- [MWF] generalise both. They state that the multi-class variant carries an unknown term depending on the data, which makes a ratio-based diagnostic unreliable; they recommend many contrastive classes with γ near 1. Their benchmark trains five estimators with five seeds per task, on a residual network of 128 hidden units and three blocks (abstract; Sections 2.2, 3, 3.3; Conclusion).
- [DHRWL]: the balanced variant adds a penalty so that posteriors tend to be more conservative at finite simulation budgets. Costs stated in the paper: a loss of statistical performance that is recovered with more simulations, and, for a strong penalty at a small budget, an approximation that reduces to the prior (Sections 3, 4).
- [sbi] implements those four variants; no telescoping estimator appeared on the pages read. Its documented defaults are a residual classifier with 50 hidden features, Adam at learning rate 0.0005, batch size 200, a 10% validation split, stopping after 20 epochs without validation improvement, and posterior samples by MCMC (vectorised slice sampling, 20 chains, 200 warm-up steps). Its installation page requires Python 3.10 or higher and recommends 3.12.

How trained estimators behave, and how others set them up:

- [HDR+]: every algorithm benchmarked, amortised ratio estimation included, can give overconfident posteriors, more so at small simulation budgets, and a large budget is no guarantee. The design trains five estimators per budget over budgets from 2¹⁰ to 2¹⁷. Ensembles of five estimators had higher expected coverage than single ones. An estimator that returns the prior has nominal expected coverage, so coverage cannot measure information gain (Sections 2, 3; Observations 1–4).
- [LBGGM]: the benchmark's ratio estimator is the multi-class formulation as implemented in sbi, a residual network of two layers of 50 units, with posterior samples by slice-sampling MCMC (100 chains, 250 warm-up steps, ten-fold thinning, 10,000 samples). The paper warns that the MCMC step can limit the performance of ratio estimators (Section 3; Appendix A).
- [TBSVG]: for simulation-based calibration the examples use 100 posterior draws per replication. For Markov-chain samplers the paper's Algorithm 2 thins the chain by effective sample size to that many effective draws, noting that some autocorrelation remains; one Markov-chain example is run without thinning, and the variational example needs none (Sections 5.1, 6, 6.2, 6.3).
- [BHMST]: the pricing network has three hidden layers of 30 nodes with ELU, trained with Adam on 34,000 parameter combinations. No stopping rule was found by text search of the arXiv version (Section 4.1; Section 5.1; Appendix A).
- [BHL]: telescoping ratio estimation, one binary classifier head per conditional parameter on a shared encoder, with densities evaluated by polynomial approximation and sampled without MCMC. 500,000 pairs split 90/5/5; early stopping with a patience of 50 epochs; AdamW; the architecture chosen by a grid search over 192 configurations; and a post-training calibration of the classifiers, before which they were under-confident and after which coverage was close to nominal. No repeated training seeds were found by text search (Section 3; Section 5.2; Appendices E.2.3, E.3; Table 23).

**Arithmetic on the draft's definitions.**

- Networks. Two priors and three levels give six cells per calibrator. If a training seed is a whole five-member ensemble, five seeds mean 150 member networks and 30 ratio estimators, 180 in all (60 trained calibrators); three seeds mean 108, eight 288, ten 360. If the five members are themselves the five seeds, 60.
- Seed-floor pairs per prior and cell: 3 at three seeds, 10 at five, 45 at ten. Prior-swap pairings at five seeds: 5 if matched by seed index, 25 if all crossings are used. The draft does not say which.
- Top level of the bootstrap: five seeds give 126 possible seed compositions and on average 3.4 distinct seeds per resample; three seeds give 10 compositions and an 11% chance that a resample is a single seed.
- Draws from the ratio estimator. The Monte Carlo standard error of a posterior mean in the Identified bin is about 0.0005 at 1,000 effective draws and 0.0017 at 100, against H-a's 0.01. The mass of an equal-tailed 90% interval built from L independent draws has a sampling standard deviation of about 0.03 at L = 100, 0.01 at 1,000 and 0.003 at 10,000; H-d's nearest limit is 0.03 from 0.90. A density evaluated on a grid has no such noise.
- The Gaussian interval. If the exact posterior equalled prior A and the ensemble returned exactly its mean and variance, the central Gaussian 90% interval, [0.032, 0.208], would hold 0.9225 of prior A's mass; under prior B the figure is 0.904. So the construction at L70 can move m away from 0.90 with no network error. A limiting case, not a forecast for the Flat bin.
- Scoring load: with N test θ, three levels, two priors and S seeds there are 3N × 2 × S ratio-estimator posteriors to produce: 15,000 at N = 500 and five seeds.

**Options.**

| Choice | Option | What it costs |
| --- | --- | --- |
| Seeds | Five per calibrator, prior and noise level (the draft) | 180 networks if a seed is a whole ensemble. Ten seed-floor pairs; 126 seed compositions in the bootstrap. Five is the number of repeated estimators [HDR+], [DHRWL] and [MWF] used; none states it as a recommendation |
| Seeds | Three | 108 networks. Three seed-floor pairs and a coarse top level in the bootstrap |
| Seeds | Eight or ten | 288 or 360 networks. Eight is what Layer 3 used. Cost per network is unrecorded |
| Seeds | The five ensemble members are the five seeds | 60 networks. The ensemble then exists once per cell, so the seed floor and the seed level of the bootstrap are undefined for it |
| Variant | Binary classifier ([HBL], as described in [MWF]) | Estimates the ratio itself. [BHL]'s heads are also binary classifiers, so it is the nearest sbi class to the family v1 names, though their method is the telescoping one |
| Variant | Multi-class ([DMP], as described in [MWF] and [LBGGM]) | The variant [LBGGM] benchmarked. [MWF] state it fixes the ratio only up to a function of the data, which breaks a ratio diagnostic |
| Variant | The generalisation of [MWF] | Two more hyperparameters to declare. The paper's preferred number of contrastive classes (large; K = 99 in its benchmark according to Section 3.3, 100 according to Appendix C.2) differs from sbi's default of 5; its γ near 1 matches sbi's default of 1.0 |
| Variant | Balanced ([DHRWL]) | Conservative by design, and at a strong penalty and small budget it tends to the prior. Extra prior dependence would then describe the method's design. L148 excludes "any correction method" without saying whether this is one |
| Variant | Telescoping with post-training calibration, as [BHL] | The only choice that makes v1's "like-for-like" literal. Not among the sbi classes read, so it needs its own implementation. Their calibration step changed coverage, so including or omitting it changes m |
| Posterior of H from the ratio estimator | MCMC through a joint estimator, with a declared draw count | Sampler error enters d and m as if it were network error. One sampling run per pair, prior and seed |
| Posterior of H from the ratio estimator | A marginal estimator for H, normalised on a grid | No sampling error; "posterior sample count" becomes a grid resolution. It gives no ξ₀ estimate, which G3's slope check needs |
| Posterior of H from the ratio estimator | Rejection or importance sampling | Independent or weighted draws without chain diagnostics. Efficiency falls where the posterior is much narrower than the proposal |
| Stopping rule | Early stopping on a validation split | The criterion is a held-out loss and touches no H output. Patience, split and tolerance become declared numbers |
| Stopping rule | A fixed number of epochs | One number. D43 records fixed budgets under-training, which only escalation revealed; G3's doubling test would be the sole check |
| Optimiser | Adam or AdamW | No source read compares them for this purpose. Learning rate, batch size and any schedule or weight decay are declared with it |
| Architecture | Small defaults: sbi's classifier; one hidden layer of 50 or 100 units per member as in [LPB] | Fully specified by a tool or a cited paper. [MWF] report potential architecture bottlenecks, the larger network's advantage coming mainly from one difficult task, SLCP (Sections 3.1, 3.3); under L142 a failure at G3 stops the study and does not permit a change |
| Architecture | Larger networks | More capacity. [BHL] chose theirs by a grid search, which L148 rules out, so it would be declared without search |
| Ensemble | With or without the adversarial step of [LPB] | The source marks it optional. Neither protocol says which |
| Training data | A fixed set, its size chosen at G3 | What L142 implies. Whether quote noise is drawn once per training surface or afresh each epoch is not stated |
| Training data | Fresh draws from G̃ every batch | Removes data as a limit, as [MWF] do in one experiment. "Training-set size" and G3's doubling would have to be restated as training steps |
| Ratio estimator | Trained once and reweighted to A and B, as [BHL] describe | Not the draft's design: L68 trains per prior. It would remove the per-prior training the draft's question is about |

Consequences not attributed to a source are reasoning from the draft's definitions and the arithmetic above.

**Established standard.** For the ensemble: [LPB] give five members as the recommended default, trained on the full data and differing by initialisation and shuffling, which agrees with v1 L177. For training a ratio estimator: no methods-paper standard was found; the documented defaults of [sbi] are listed above. For the number of training repetitions: no stated recommendation was found. For the posterior sample count: no single standard; 100 ([TBSVG]), 1,000 (the default `num_posterior_samples` of `run_sbc` in [sbi]) and 10,000 ([LBGGM]) all appear.

**To defend it, you would need to be able to say.** For each calibrator, what exactly is trained, how do I know training has finished, and how is its 90% interval produced; and if a referee swapped in another equally standard choice, a different variant, sampler, draw count or number of seeds, which of H-a to H-d could change verdict, and why is my choice the one whose result I am prepared to call amortised calibration of rough Heston?

**Checklist boxes.** "Architectures and training procedure declared" (L173).

**Not confirmed.** Whether sbi installs and runs in the Layer 3 environment: not tested. The depth of sbi's default residual classifier: not shown on the pages read. [HBL] and [DMP] were read as abstracts only. Other libraries that might count as standard tooling were not examined. No source was found that states how many training seeds an amortised-inference study should use, or how a hierarchical bootstrap behaves with five top-level units. Appendix H of [LBGGM] was not read.

## 9. Stratification: the contraction cut-offs (draft L87)

**The choice.** L77–87: "Each scored (surface, noise level) pair is binned by the exact posterior's contraction of H under prior A: c = 1 − Var_post(H | y) / Var_prior(H)", with Identified at c ≥ 0.9, Partial at 0.5 ≤ c < 0.9 and Flat at c < 0.5. "**OPEN — the cut-offs.**"

**Decided as.** A value now.

**What depends on it.**

- Which bin each hypothesis is scored in: H-a and H-c in Identified, H-b in Partial, H-d in Flat (L119–122).
- L124's statement "In the flat bin the reference itself has S near one".
- The 10% floor (item 11) and the null (L126).
- Novelty claim (ii): "calibrator fidelity measured against an exact posterior, stratified by contraction" (L156).
- The dose-response at L132 does not depend on it: it uses c itself, not the bins.

**Settle first.** Item 6: a cut-off in c is a posterior standard deviation only once prior A is fixed. Item 7: Var_post is a Monte Carlo estimate, so pairs near a cut-off can be misfiled. Item 12: H-c's numbers and L124 are tied to where the cut-offs sit.

**What the repository shows.**

- v1 stratified by quintiles of CRB_H: five equal-occupancy groups whose boundaries were "computed from CRB_H on the sealed test set before any network is scored, so the bins cannot be drawn around the answer" (v1 L116, L173). v2 replaces this with three bins at fixed values of a different quantity. The reason for moving from equal occupancy to fixed values is not recorded.
- No reason is recorded for 0.9 and 0.5.
- No posterior, contraction, Jacobian or Fisher information has been computed for P5 (D55, ROADMAP L1031–1032, L1056–1057), so the cut-offs are set without any distribution of c in hand.

**What the literature does.**

- [Bet] defines posterior shrinkage as 1 − posterior variance / prior variance and recommends plotting the posterior z-score against it. The regions are labelled on a figure; no numerical cut-offs are given (Section 3.4.3; Figure 6).
- [SBV] define posterior contraction by the same formula, propose no formal cut-offs and say the plot is not a test. In a worked example they describe contraction below 0.5 as weak and values approaching 1 as very strong (Eq. 5, p. 27; pp. 29–30; pp. 55–58).
- [GVS+] mention comparing prior and posterior spread as one shortcut for judging a prior's influence, with no formula and no thresholds (Section 6.3).
- [BayesFlow] provides a `posterior_contraction` diagnostic. The user guide describes contraction as the share by which the posterior variance has fallen relative to the prior variance (Section 9.4). The function's source listing computes 1 − posterior variance / prior variance and clips the result to [0, 1], so negative values become zero; the function's documented text does not mention the clipping. None of these pages gives a threshold for contraction.
- [BHL] measure contraction by divergences between each marginal posterior and its prior, not by a variance ratio, and stress that it is specific to the chosen prior (Section 4).

**Arithmetic on the draft's numbers.**

- Under the proposed prior A (standard deviation 0.0537), c ≥ 0.9 means a posterior standard deviation of H of at most 0.017; c ≥ 0.5 means at most 0.038.
- In a Gaussian reading, c = 0.5 is where the data and prior A carry equal precision about H, and c = 0.9 is where the data carry nine times the prior's.
- The reference's own shift ratio at the bin edges, in a Gaussian approximation with the draft's unequal prior variances and the likelihood centred anywhere in the draft's range for H, 0.02 to 0.48 (L37):

| c | Reference's S, approximately |
| --- | --- |
| 0.95 | 0.03 to 0.06 |
| 0.90 | 0.07 to 0.12 |
| 0.75 | 0.175 to 0.29 |
| 0.50 | 0.39 to 0.56 |
| 0.25 | 0.66 to 0.80 |
| 0.10 | 0.86 to 0.92 |

  So on this approximation a perfectly faithful network has S of about 0.07 to 0.12 at the lower edge of the Identified bin, the size of H-c's "above 0.10". In the Flat bin the reference's S runs from about 0.39 to 0.56 just under the cut-off and approaches one only as c approaches zero. This is a one-dimensional Gaussian approximation; Beta priors, non-Gaussian posteriors and four parameters will change the numbers. It shows that the cut-offs and the hypothesis numbers are coupled, not what the reference will show.
- Monte Carlo error in c for a roughly Gaussian posterior with n effective draws is about (1 − c)·√(2/(n − 1)): ±0.007 at c = 0.9 and ±0.035 at c = 0.5 with n = 400. Misfiling matters about five times more at the lower cut-off.
- H-a's 0.01 and 0.03 are 0.59 and 1.77 of the largest posterior standard deviation the Identified bin allows.

**Options.**

| Option | What it costs |
| --- | --- |
| 0.9 and 0.5 (the draft) | On the Gaussian reading the Flat bin includes pairs whose reference shift ratio is only about 0.39 to 0.56, and the lower edge of Identified has a reference shift ratio the size of H-c's expected value |
| Other fixed cut-offs, for example 0.95 and 0.5, or 0.75 and 0.25 | Moves pairs between bins and with them the reference's own S at the edges. Each choice needs the hypothesis numbers read against it. A stricter bin is more exposed to the 10% floor |
| Equal-occupancy bins, as v1 did | Boundaries computed before scoring, every bin populated, no floor needed. The bins lose a fixed meaning, and L124's statement would no longer refer to a fixed range of c |
| No bins: hypotheses stated on c as a continuous variable | No cut-off and no floor to defend. The four hypotheses would have to be re-stated as trends, a larger rewrite of the table |
| Two bins | One fewer value to defend. H-b is defined on the Partial bin, which would cease to exist |
| A different contraction measure, as in [BHL] | A divergence responds to a shift in location as well as to a change in spread, so it is not the same quantity and would need new cut-offs |
| Keep the cut-offs but state the edge conventions | Say what happens when c is negative and how Monte Carlo error near a cut-off is handled. Without it, negative c falls into Flat by the table's inequality |

The consequences are reasoning from the draft's definitions.

**Established standard.** None found for cut-offs. The quantity itself is standard: [Bet], [SBV] and [BayesFlow] all use the variance ratio. One convention read at source: [BayesFlow]'s implementation clips negative values to zero (source listing; the function's documented text does not say so).

**To defend it, you would need to be able to say.** Why does "identified" begin where the posterior variance is a tenth of prior A's and "flat" where it is more than half, and at those edges do the numbers I have attached to each bin, 0.10 for S in Identified and "S near one" in Flat, still say what I mean them to say?

**Checklist boxes.** "Contraction cut-offs accepted or changed" (L174).

**Not confirmed.** Whether any simulation-based-inference paper classifies parameters by contraction thresholds: one search found none; it was not exhaustive.

## 10. Test sets: the number of test θ (draft L92)

**The choice.** L92: "**OPEN — the number of test θ.** Set from the reference's measured cost at G2, written in as an amendment before any network is trained." With it, L91: "θ uniform on the box, seed 20261002 (retained from v1); one noise draw per θ per noise level, seeds declared."

**Decided as.** A rule now; the value is fixed at G2.

**What depends on it.**

- Stage 1 (L97): the distribution of c and the share of pairs in each bin.
- The 10% floor (L99). The floor is a share; the number of pairs behind it is set here.
- Every hypothesis (L119–122): each is a median over the pairs in one bin, with a 95% interval whose width depends on the number of pairs (L130).
- The reference's total cost (L66): one run per scored surface and noise level, plus re-runs.

**Settle first.** Items 7 and 15: the sampler settings set the cost per posterior, and the value cannot exist before G2. Item 5: pairs are the number of θ times the number of levels. Items 1 and 2: the test θ are uniform on the box, so the sealed draws can be generated only after G0. Item 8: seeds and the ratio estimator's sampler set a scoring cost per pair that the rule does not count. Items 9 and 11: they decide how many pairs land in each bin and how few is too few.

**What the repository shows.**

- v1 sealed a seed, not a size: "**DECISION — the confirmatory test set is sealed now.** Its random seed is 20261002, in the repository's date-style convention; pilots draw their surfaces from different seeds" (v1 L161). v1 gave no test-set size either.
- v1 bootstrapped "over test surfaces" (v1 L165). v2 bootstraps "over training seeds and, within seed, pairs" (L130).
- G2 is where the cost is measured: "Cost per posterior is measured and fixes the test-set size" (L141).
- "Any later change is a dated amendment stating what changed, why, and what had been seen" (L161). Writing in the number is such an amendment.
- The only timing in the draft is for the pricer, "about 1.6 s on one core" per surface (L20). The cost of a reference posterior on the emulator is unrecorded.
- No compute budget appears in the draft. Machine details on record include "cap the pool at **physical cores (4)**" (ROADMAP L855, D38) and "a memory-tight machine" (ROADMAP L843, D37); neither line names the machine. The 3 October probe ran on one core in a chat sandbox and not on the author's machine (ROADMAP L1044–1050, D55).
- No noise seeds appear in the draft, although L91 calls them "declared".

**What the literature does.**

- [LBGGM], the one benchmark read that compares each approximate posterior with a reference posterior, uses 10 observations per task, with 10,000 reference samples each (Section 2.4). [MWF] score the same ten (Section 3.3).
- [HDR+]: coverage studies need no reference posterior and use far more test points: at least 5,000 unseen samples for amortised estimators, and 300 observations for non-amortised ones, a number the authors tie to cost (Section 3).
- [TBSVG]: for simulation-based calibration the number of replications limits the sensitivity of the check; the paper's rule of thumb is about 20 replications per histogram bin, and its examples use 1,000 or 10,000 (Section 4.1; Section 6).
- [sbi]'s calibration tutorial uses 200 runs with the comment that the number should be in the hundreds, and notes that repeated inference is cheap for amortised methods but not for sequential methods or for any method that samples by MCMC or variational inference. The tutorial does not name ratio estimators; sbi's ratio-estimator classes sample by MCMC by default (API reference, `build_posterior`).
- [BHL] hold out 5% of 500,000 pairs, which is 25,000, and assess coverage on it. Coverage is estimated, for each held-out pair, from draws of the learnt conditional posteriors set against that pair's true parameters; no reference posterior enters the check (Section 5.2; Appendix E.2.3, Eqs. E.9–E.10).

**Arithmetic on the draft's definitions.**

| Number of test θ | Pairs (three levels) | A bin at the 10% floor, pairs pooled | Reference runs | Ratio-estimator posteriors at five seeds |
| --- | --- | --- | --- | --- |
| 100 | 300 | 30 | 300 to 900 | 3,000 |
| 200 | 600 | 60 | 600 to 1,800 | 6,000 |
| 500 | 1,500 | 150 | 1,500 to 4,500 | 15,000 |
| 1,000 | 3,000 | 300 | 3,000 to 9,000 | 30,000 |

- The upper end of the reference range is the worst case in which every pair needs a direct re-run under both A and B. For every reference run there are ten ratio-estimator posteriors to produce at five seeds.
- A 95% interval for a median of n independent values runs roughly from the (50 − 98/√n)th to the (50 + 98/√n)th percentile of those values: the 37th to the 63rd at n = 60, the 42nd to the 58th at n = 150, the 44th to the 56th at n = 300. This ignores seed-to-seed variation and the dependence between pairs that share a θ, and it cannot be turned into a width in H until per-pair values exist.
- A bin share near 10% is itself uncertain: at 200 test θ its standard deviation is between 0.012 (pairs treated as independent) and 0.021 (only θ independent). So whether a bin near the floor is declared untestable is partly chance.

**Options.**

| Option | What it costs |
| --- | --- |
| Set from the reference's measured cost at G2, by amendment (the draft) | For the rule to leave no discretion, the frozen text would need the compute budget on a stated machine, the formula from cost to number including how re-runs are counted, a minimum below which P5 stops or the reference is changed, a cap if the reference proves cheap, and the generator and draw order under seed 20261002. Without them the number is chosen after pilot outputs have been seen, and L161 requires the amendment to say so |
| Fix the number now | No amendment and no discretion. If the cost proves unaffordable the choice is to amend after seeing pilot data or to stop; if cheap, the study is smaller than it could have been. No cost figure exists today |
| Fix a minimum now from the precision wanted; let G2 decide only go or stop | Ties the number to what the hypotheses can resolve. The minimum pairs per bin is itself a judgement, since per-pair distributions of d, E and m do not exist yet |
| A budget rule that also counts the calibrators' scoring cost | At five seeds there are ten ratio-estimator posteriors per reference run. If they are sampled by MCMC, a rule based on the reference alone would overstate the affordable number. Depends on timings that do not exist |
| A larger reference-only set for Stage 1, a subset scored in Stage 2 | A finer map for the same scoring cost. Any subset chosen with knowledge of c changes the uniform design. Not in the draft |

The consequences are reasoning from the draft's definitions.

**Established standard.** None found. Practice in the sources read spans three orders of magnitude and depends on whether each test point needs an exact reference: 10 where one is computed ([LBGGM], [MWF]), hundreds to thousands for calibration and coverage checks ([sbi], [HDR+], [TBSVG]), 25,000 for coverage in [BHL].

**To defend it, you would need to be able to say.** Before G2 is run, what is the compute budget that turns the measured cost per posterior into a number of test θ, what is the formula, and what is the smallest number at which I would still run the study; and at that smallest number, how many pairs does a bin at the 10% floor hold, and why is that enough for a 95% interval to fall clear of the numbers in H-a to H-d?

**Checklist boxes.** None names the test-set size. It is settled by the amendment L92 describes, after G2, and bears on "Gate tolerances declared" (L176).

**Not confirmed.** The cost of one reference posterior on the emulator: it does not exist until G2. The bin shares over the box. No source was found that prescribes a number of test points for a per-pair comparison against an exact reference, or a minimum number of pairs per stratum for a bootstrap interval on a median. Whether the three noise draws for one θ are meant to be independent is not stated in the draft.

## 11. Stage 1: the 10% floor (draft L99)

**The choice.** L99: "Descriptive, with one stop rule: a bin holding fewer than 10% of pairs makes its hypotheses untestable, and they are declared so, not adjusted. **OPEN — the 10% floor.**"

**Decided as.** Mixed. The share is a value now; its size in pairs is fixed only when the number of test θ is set at G2.

**What depends on it.**

- Whether each hypothesis is testable: the Identified bin carries H-a and H-c, Partial carries H-b, Flat carries H-d.
- The verdicts (L131). The floor adds a fourth outcome, "untestable", decided before any network is scored.
- The null (L126), which speaks of "every bin".

**Settle first.** Item 10 (number of test θ), item 5 (the number of levels sets the pairs per θ), items 9 and 6 (they define the bins), and item 8 with the bootstrap design (how many pairs a bootstrap of a median needs).

**What the repository shows.**

- Pairs arise as "one noise draw per θ per noise level" (L91), and the number of θ is not yet fixed (L92).
- Hypotheses are scored "pooled over noise levels" (L124), by a "95% hierarchical bootstrap over training seeds and, within seed, pairs" (L130).
- v1 needed no floor: its bins were equal-occupancy quintiles (v1 L116). v1's verdict rule already sends a weakly supported result to "inconclusive" (v1 L171).
- Nothing in the repository says what share of pairs falls in any bin. The reason for 10% is not recorded.

**What the literature does.**

- [Hes] states that the common percentile bootstrap interval under-covers badly when the sample is small (abstract) and is too narrow there (Section 5.2); that for a median or other quantile of a small sample the bootstrap distribution can differ greatly in shape and scale from the sampling distribution, while the percentile interval for the median is nonetheless described as acceptable (Section 3.3); and that resampling does not make up for a sample too small to represent its population (Section 3.5). It gives no minimum size for a stratum.

**Arithmetic on the draft's definitions.**

| Number of test θ | Pairs (three levels) | Floor, all pairs pooled | Floor, within one level |
| --- | --- | --- | --- |
| 50 | 150 | 15 | 5 |
| 100 | 300 | 30 | 10 |
| 500 | 1,500 | 150 | 50 |
| 1,000 | 3,000 | 300 | 100 |

- The draft does not say which of the two readings of "pairs" is meant. Under the pooled reading, a bin that receives one whole noise level and nothing else always clears the floor.
- A share near 10% is itself estimated with a 95% binomial half-width of about ±3.4 points on 300 pairs and ±1.5 on 1,500, treating pairs as independent. Pairs that share a θ are not independent, so the true uncertainty is larger. The draft's rule is applied to the observed share, not to an interval.
- At the floor, each median rests on 0.3 × (number of θ) pairs. The same pairs are reused across the five seeds, so the number of distinct surfaces in the bootstrap is the pair count, not five times it.

**Options.**

| Option | What it costs |
| --- | --- |
| 10% of pairs (the draft) | A relative rule decided from the reference alone, before any network is scored. Its size in pairs is unknown until G2. It can remove a hypothesis outright |
| State which pairs: pooled or per noise level | Pooled: a bin fed by one whole level always passes. Per level: a bin can pass at one level and fail at another, which fits the per-level breakdown but multiplies the floor decisions |
| An absolute minimum number of pairs, alone or with a percentage | Ties the rule to what the bootstrap of a median needs. [Hes] supports the concern for the bootstrap distribution of a median in a small sample, describes the percentile interval for the median as acceptable, and supplies no number, so the count would be your own choice. Makes this a rule now with its value at a gate |
| A different percentage | 5%: fewer hypotheses lost, thinner bins scored. 20%: sturdier medians, a hypothesis lost whenever its bin has under a fifth of pairs |
| No floor: rely on the three-verdict rule | Nothing is declared before the networks are scored. A very thin bin could still return "confirmed" or "refuted" on a handful of pairs. The difference between "untestable by design" and "inconclusive on the evidence" is lost |
| Secure occupancy by design | Equal-occupancy bins, or noise levels and number of θ chosen at G2 from pilot posteriors. The floor becomes unnecessary or rarely binding. Equal-occupancy bins give up fixed cut-offs; choosing levels from pilots makes item 5 a rule fixed at a gate |

The consequences are reasoning from the draft's definitions.

**Established standard.** None found for a minimum share or count per stratum in a pre-registered stratified analysis.

**To defend it, you would need to be able to say.** Ten per cent of what exactly, all pairs pooled across the noise levels or the pairs at each level, and why is that share both enough pairs to bootstrap a median and a rule I am prepared to lose a hypothesis to?

**Checklist boxes.** None names the floor. It bears on "Contraction cut-offs accepted or changed" (L174) and "Every number in the hypotheses table defensible" (L175).

**Not confirmed.** Whether any pre-registration guidance gives a minimum stratum size for bootstrap intervals of a median: not searched beyond [Hes]. What shares the three bins will hold: known only from pilot reference posteriors.

## 12. Hypotheses H-a to H-d: directions and numbers (draft L115, table L117–122)

**The choice.** L115: "Named H-a to H-d to avoid collision with papers P1–P4. Directions and numbers are the author's bets and are **OPEN**." The four rows, L119–122:

- "| **H-a Faithful when informed** | median of abs(d) | Identified | below 0.01 | above 0.03 |"
- "| **H-b Excess prior dependence** | median E | Partial | above 0.15 | below 0.05 |"
- "| **H-c Leak into the identified regime** | median S_net | Identified | above 0.10 and above the seed floor | below 0.05 |"
- "| **H-d Interval infidelity** | median m | Flat | outside [0.80, 0.95] | inside [0.85, 0.93] |"

The columns are Measure, Bin, Expected and Refuted if (L117).

**Decided as.** Values now, as drafted. For H-c's comparison value a rule-now form is also available, with the value fixed at Stage 1, which uses the reference alone (see the options). The seed floor is network-only (L109), so a rule for it would take its value from trained networks.

**What depends on it.**

- The verdict rule: "**Confirmed**: whole interval on the expected side of the expected value. **Refuted**: whole interval beyond the refutation threshold. **Inconclusive**: otherwise" (L131). Four hypotheses and two calibrators give eight pooled verdicts.
- The null (L126). The draft gives no rule from the eight verdicts to the null.
- The reporting commitment: "Every hypothesis reported with verdict and interval, for both calibrators" (L160).
- Novelty claim (ii), "calibrator fidelity measured against an exact posterior, stratified by contraction" (L156).
- The amended registration: the table is what it commits to (L11, L161).

**Settle first.** Items 6 and 9: the 0.20 in S and the prior standard deviation come from the Beta parameters, and the cut-offs decide the bin each row is scored in. If either moves, every row changes meaning without a digit in the table changing. Item 5: the levels decide which bins are populated and by which networks. Item 8: the seed count sets the top level of the bootstrap, the seed floor and the pairing of A-trained with B-trained networks. Item 7: the reference's Monte Carlo error sits under H-a's 0.01 and H-d's margins. Items 10 and 11: they decide whether a row is testable and how wide its interval is. Item 16: G3 selects the training-set size on abs(d), H-a's measure.

**What the repository shows.**

- No reason is recorded for any of the ten numbers in the table's eight Expected and Refuted if cells, in the draft, the ROADMAP or the commit that added the draft. The only reasoning attached to the table is L124: "In the flat bin the reference itself has S near one, so excess prior dependence is not testable there; only interval fidelity is."
- The reason for the amendment: "v1's predictions followed from Bayes' rule and the parameter box, so an exact posterior would have confirmed them as readily as a network" (L9).
- "Every comparison is per pair against the reference, so nothing rests on frequentist coverage within strata" (L111). H-a, H-b and H-d compare with the reference. H-c's measure, S_net, does not.
- v1 used the same shift ratio with the same normaliser: "the difference in their answers divided by the 0.20 gap between their priors' means" (v1 L122). v1's P3 expected the median shift ratio "below 0.1" in the most identifiable quintile (v1 L136). H-c expects median S_net "above 0.10" in the Identified bin. The bins differ and v2 pools three noise levels, but the direction of the bet in the identified regime is reversed. L9 gives a general reason for the amendment; a reason for this reversal is not recorded.
- v1 statements with no counterpart in v2's hypotheses section: "They are falsification thresholds, not targets: no code may be tuned to reach them" (v1 L143); "If all four predictions are refuted, the null stands" (v1 L151); a home-prior control (v1 L139); and "A result inside it is reported as **inconclusive**, not stretched into a confirmation" (v1 L141).
- v1's interval was a "95% bootstrap interval over test surfaces" (v1 L165). v2's is a "95% hierarchical bootstrap over training seeds and, within seed, pairs" (L130).
- The research question has a direction, "does the gap grow as the surface loses information about H?" (L30). The dose-response line states none: "Spearman correlation of c with abs(d), with interval" (L132).
- "tolerance" appears at L126 and in gates G0, G2 and G3, each OPEN. None is defined for the null.
- Neither protocol says anything about multiplicity or power. On sample size, v2 sets the number of test θ from the reference's measured cost (L92, L141) and proposes five training seeds (L73); v1 gave no test-set size and left the number of noise realisations per test point in E2 to be "set in the G3 pilot, as the smallest number at which the spread estimate stops moving" (v1 L177). v2: "No aggregate verdict" (L124).

**What the literature does.** Nothing read bears on the numbers themselves. What was read bears on the machinery around them.

- [Hes]: the percentile bootstrap interval under-covers in small samples. For five units its Table 5 gives a narrowness factor of 0.894, a ratio of normal to t quantiles of 0.706 and a one-sided miss rate of 0.077 against a nominal 0.025 (abstract; Section 5.2, Table 5).
- [CGM]: with few clusters, which the abstract puts at five to thirty, standard asymptotic tests can over-reject; the authors study cluster bootstrap-t procedures (abstract).
- [SBS]: for nested data a hierarchical bootstrap applied level by level keeps the Type-I error within its bound (abstract). [Owe]: for crossed random effects a naive bootstrap can be seriously misleading, and an alternative resamples rows and columns separately (abstract). In the draft every seed is scored on the same pairs, so seeds and pairs are crossed.
- [CSO]: a tutorial explaining the link between the number of random seeds and the probabilities of statistical error, with guidelines for choosing that number when the performance of two algorithms is compared (abstract). [ASC+]: with a handful of training runs, point estimates of aggregate performance can mislead, and interval estimates are advocated (abstract).
- [HDR+]: the failure reported for simulation-based inference algorithms, ratio estimation among them, is overconfidence, that is intervals too narrow, and ensembling is reported as more reliable (abstract). H-d as drafted is two-sided.
- [SBKR]: in amortised inference, posterior errors grow with the distance between the data and the typical set of the training simulations (abstract). The test θ are uniform on the box while each network is trained under A or B.
- [EOS+] suggest deep ensembles as a way of flagging sensitivity whose origin is an unreliable approximation, for example under model misspecification (abstract). The abstract does not contrast this with prior sensitivity. The draft's seed floor (L109) is S between networks that differ only in training seed; the draft does not state its purpose.
- [BayesFlow]'s user guide reads contraction together with a posterior z-score, which it describes only as a standardised gap between the posterior mean and the true parameter value (Section 9.4).
- [OSF]'s guidance for preregistration asks for the statistical tests and the decision criteria to be described explicitly.

**Arithmetic on the draft's definitions.**

- Units. S = 0.10 is a difference of 0.020 in H between the two networks' answers; 0.05 is 0.010. H-b's 0.15 and 0.05 are 0.030 and 0.010 in H. H-a's 0.01 is 0.19 of prior A's standard deviation.
- An identity. Writing d_B for the prior-B analogue of d, which the draft does not define, E = (d_B − d_A)/0.20 on every pair, and S_net = S_exact + (d_B − d_A)/0.20. So H-b is a statement about the difference of the two networks' mean errors, and where both are within 0.01 of the reference, abs(E) is at most 0.10. Medians do not add, so this does not carry over exactly to bin medians. E is positive whenever the B-network's error exceeds the A-network's for any reason.
- The reference's own S, on a Gaussian approximation with the draft's unequal prior variances and the likelihood centred anywhere in the draft's range for H (0.02 to 0.48, L37): about 0.03 to 0.06 at c = 0.95, 0.07 to 0.12 at c = 0.90, 0.21 to 0.35 at c = 0.70 and 0.39 to 0.56 at c = 0.50. With equal variances it would be 1 − c. An approximation, not a computed posterior.
- H-c against that. A network with no excess has S_net equal to the reference's S. If the Identified bin's pairs cluster near c = 0.9, such a network has a median S_net near 0.10, the Expected value. If the bin's median c lies between about 0.90 and 0.95, it lands between 0.05 and 0.10 and cannot be refuted. Only if the median c exceeds about 0.95 does it fall below 0.05. Which of these holds is a Stage 1 fact.
- H-a in posterior standard deviations: 0.01 is 0.59 at c = 0.9 and 1.86 at c = 0.99; 0.03 is 1.77 and 5.59. An absolute threshold is stricter the better H is identified.
- H-b on the Gaussian picture with equal prior variances, where the reference's S is 1 − c: E = c − (1 − S_net), the shortfall of the contraction the network behaves as if it had. On that simplification a network returning only its prior mean would score E = c, that is 0.5 to 0.89 across the Partial bin, and the same E of 0.15 is 136% of the reference's S at c = 0.89 and 30% at c = 0.50.
- H-d as a ratio of interval width. For a Gaussian posterior and a network interval with the same centre, m = 0.80, 0.85, 0.93 and 0.95 correspond to a network standard deviation of 0.78, 0.875, 1.10 and 1.19 times the exact one. The bands look asymmetric about 0.90 in m and are roughly symmetric in that ratio. A centre displaced by half a posterior standard deviation with the right width gives m = 0.858.
- H-d and the ensemble. In the limit where the exact posterior is prior A, a Gaussian interval with exactly the right mean and variance scores m = 0.9225, which is 0.0075 below the top of the refutation band. The ratio estimator's equal-tailed interval has no such offset.
- The seed floor. As defined at L109 it is a signed per-pair quantity whose median is zero by symmetry. If two same-prior networks differed by independent normal noise of standard deviation τ each, the median of its absolute value would be 4.77τ: 0.05 at τ = 0.0105. An illustration.
- Five seeds. For a mean of five independent normal values, a nominal 95% percentile bootstrap interval is about 63% of the width of the t interval and covers about 85%; the factors agree with [Hes]'s Table 5. This is for a mean, not the draft's statistic. Because L131 requires the whole interval to clear a number, an interval that is too narrow returns Confirmed and Refuted more readily than a 95% interval should.
- Counts. Eight pooled verdicts; 24 more results if each noise level carried its own. If eight 95% intervals were independent with exact coverage, the chance that at least one misses would be 0.34; neither assumption holds.
- Test set against training priors: about 46% of test H lies above prior A's 99th percentile and about 32% below prior B's 1st percentile. d and m use the A-trained network only.
- Bins and levels. Reading D38's three scatters as likelihood standard deviations puts 0.1 vol point in Identified and 0.3 in Partial, and a straight-line extension puts 1.0 in Flat. If that held across the box, each pooled hypothesis would be scored mostly on the networks of one noise level. One θ, an optimiser's scatter, and an extrapolation: indicative only.

**Options.** These are your bets; the sheet gives no view on any of them.

| Option | What it costs |
| --- | --- |
| Freeze the table as drafted | Four things cannot be computed from the text alone: H-c's seed-floor clause, the pairing of A- and B-trained seeds, the bootstrap's units and interval type, and the null's "tolerance". H-c's verdict would depend on where c sits inside the Identified bin as well as on the network |
| Keep every number; add the missing definitions only | The bets are unchanged and each verdict becomes reproducible from the registered text. The definitions are: the seed-floor statistic; the seed pairing; the bootstrap; the null's tolerance and which verdicts amount to the null; the sign and status of the dose-response; whether per-level results carry verdicts |
| Score H-c on excess, E, not on S_net | H-c would then test the network's own contribution, in line with L9 and L111. It becomes H-b's measure in another bin, and through the identity it is tied to H-a |
| Keep S_net for H-c; fix its comparison value by rule | The value would come from Stage 1's results, produced "From the reference alone" (L97); the draft does not say whether Stage 1 comes before or after network training. A number set after reference results are seen; the draft already uses that pattern for the test-set size |
| Express H-a relative to each pair's exact posterior standard deviation | Uniform in posterior-standard-deviation terms. No longer reads in units of H, and inherits the Monte Carlo error of the reference's variance |
| Make H-d directional, or state it separately for the two calibrators | "The expected side" becomes unambiguous under L131. As drafted, intervals that are too wide and too narrow both confirm, and a pooled median can sit in the band while pairs err both ways |
| A verdict per noise level, or pooled verdicts with the breakdown descriptive | Per level: each verdict lines up with one set of trained networks, at three times as many verdicts, smaller bins and possibly empty ones. Pooled: eight verdicts, but a bin may be dominated by one level |
| Change the interval or the number of seeds | A t-based or bootstrap-t interval on seed-level statistics, more seeds, or resampling seeds and pairs as crossed factors. Wider intervals make Inconclusive more likely; more seeds scale the training |
| Add a statement on multiplicity | Read individually without adjustment and say so, or at an adjusted level, or with one hypothesis named primary. With no statement the text is silent, as v1 was |
| Regroup the rows | For example H-a as a control, or H-c merged or dropped. Fewer verdicts. H-a is the only row whose Expected is fidelity; H-a and H-c share a bin and are linked by the identity |
| State or demote the dose-response | Give it a sign and a verdict rule, or label it descriptive. A negative rank correlation would also arise with no prior-leaning if errors simply scale with posterior width |

The consequences are reasoning from the draft's definitions and the arithmetic above.

**Established standard.** None found for any of the table's ten numbers, for the directions, or for the three-verdict interval rule. On the machinery: [Hes] and [CGM] on intervals with few top-level units, [Owe] on crossed data, and [OSF] on stating tests and decision criteria.

**To defend it, you would need to be able to say.** For each row, in units of H and in exact-posterior standard deviations for its bin: what would the reference itself score on this measure, what would a perfectly faithful network score once Monte Carlo error and seed noise are added, and why do my Expected and Refuted values sit where they do against those two numbers? In particular, is H-c a bet about the network's total shift or about its excess over the reference, what number is "the seed floor", what would count as the null, and can five seeds give an interval honest enough to clear gaps of 0.02 in H, 0.10 in E, 0.05 in S and 0.02 to 0.05 in m?

**Checklist boxes.** "Every number in the hypotheses table defensible in the author's own words" (L175).

**Not confirmed.** The published convention nearest to the whole-interval rule (interval against a region of practical equivalence) could not be read: the publisher pages were blocked, and the references are listed under leads. The journal version of [CGM] was not reached. [ASC+] was read as an abstract only. The reference's S under the actual scaled-Beta priors, whether c can be negative on this model, and m for any case other than the limit above were not computed. How pairs distribute over c within each bin is not knowable before Stage 1.

## 13. Gate G0: the tolerance (draft L139)

**The choice.** L139: "**G0 — pricer over the box.** Existing verification tests pass. On pilot points across the box, including its corners, every quote is finite and changes by less than a stated tolerance when `N_riccati` doubles. Cost per surface is measured. If this fails, the box or the grid is shrunk here. **OPEN — tolerance.**"

**Decided as.** A value now.

**What depends on it.**

- Items 1 and 2: the tolerance decides what counts as failing, and so whether the box or the grid is cut.
- Q1's claim for rough Heston (L49, L140). G1 measures the emulator against the pricer. G0's tolerance is the only bound on the pricer against the model.
- The emulator design (item 4), through "Cost per surface is measured".

**Settle first.** Item 5 (noise levels): a tolerance in vol points only has meaning against s. Item 14 (G1 ratio): if pricer error and emulator error both count against s, the two share one budget. Which pricer G0 tests: a uniform `N_riccati`, as in the D55 probe, or the per-maturity schedule in `calibrate_btc.py`.

**What the repository shows.**

- The existing tests G0 says must pass are not named in the draft. The candidates are `test_rough_heston_cf.py`, `test_layer4_calibrate.py`, `test_layer4_calibrate_surface.py` and `test_layer4_smile_gate.py`; `test_rough_heston_lifted.py` also calls the CF pricer, at ν = 0.40 (`:75-93`). They check the H = ½ limit, CF properties at one rough point, calibration machinery at and near the truth point, and that two low-H, high-ν inputs, (H, ν) = (0.025, 0.99) and (0.03, 0.95) at N = 150, return a fixed-length output without raising (`test_layer4_calibrate.py:40-43`, `:56-57`).
- The only test that refines N is at H = ½, with N = 100, 200, 400 (`test_rough_heston_cf.py:102-106`). No test refines N in the rough regime.
- The CF property tests use one rough point: H = 0.10, ν = 0.20, ρ = −0.70, N = 600, at T = 1 except for one small-T check at T = 0.05, 0.01 and 0.002 (`test_rough_heston_cf.py:142-146`, `:168-171`). The other candidate files price at and near the truth point (0.10, 0.35, −0.70, 0.04) with N = 1000 or 900 (`layer4_calibrate.py:41`; `test_layer4_calibrate.py:16`; `test_layer4_calibrate_surface.py:13`) and check that the CF smile is finite at H = 0.10, ν = 0.40 (`test_layer4_smile_gate.py:37-40`).
- The repository already has one convergence tolerance for this pricer: an absolute implied-vol difference of `atol=2e-3` (0.2 vol point), ATM only, between N(T) and N(T) + 2000, at four (H, ν) corners (`calibrate_btc.py:132`, `:144`, `:149`).
- One convergence order is recorded in the rough regime, at a single mild point: "order p ≈ 1.60 = 1+H+½" at H = 0.10, ν = 0.20 (D30, ROADMAP L774).
- Convergence at the hard corner is recorded as agreement of the printed implied vol across larger N: "IV 0.4285 bit-identical through N=14000" (D41, ROADMAP L882).
- The code names the Fourier inversion's own settings as a separate error source from the Riccati grid (`rough_heston_cf.py:136-139`). They are fixed, at `U_max` 200 and 128 nodes on the surface (`rough_heston_cf.py:128`; `layer4_calibrate_surface.py:44`); one test checks their convergence on the classical Heston characteristic function (`test_rough_heston_cf.py:67-74`). G0 as drafted varies only `N_riccati`.
- A quote can be non-finite by several routes in the code: a non-finite call price; a call price at or below 1e-12; for a strike below the forward, a put obtained by put–call parity that is non-finite or at or below 1e-12; a failed implied-vol root search; or any exception raised while pricing, which is caught and leaves NaN (`layer4_calibrate_surface.py:62-67`; `rough_heston_cf.py:75-86`). "Finite" in G0 does not distinguish them.
- The draft names `calibrate_btc._cached_cf`-style memoisation (L50). The cached surface model built on it is described as bit-identical to the engine (`calibrate_btc.py:38-41`; D39, ROADMAP L865), but no test file references it. `_cached_cf` itself takes `N_riccati` as an argument (`calibrate_btc.py:43`); the surface model built on it, `make_cached_surface_model`, replaces any global `N_riccati` with the per-maturity schedule (`calibrate_btc.py:90-99`), whereas the D55 probe used per-maturity memoisation with a single `N_riccati` of 1000 (ROADMAP L1048–1049).

**What the literature does.**

- [ER] use the fractional Adams scheme and state convergence of the Riccati solution at grid points for a real Fourier argument, with error of order o(Δ^(2−α)) away from the origin, α = H + ½ (Section 5.1). At H = 0.10 that exponent is 1.4. D30's measured 1.60 is a fitted order for the pricer's "reference error" as `N_riccati` grows (ROADMAP L774; the line does not state whether that error is a price or an implied vol, or its strike and maturity). It is not stated to be the error of the Riccati solution at grid points, so the two exponents are not shown to be the same quantity.
- [CGP] measure error as the difference between implied vols from two solvers and set a maximal difference of 1%, which they themselves call very large. They report short-maturity cases where no step count brought the Adams price within that tolerance (Section 5.2; Table 3 and its caption).
- [HK] report the largest relative implied-vol error over the strike grid for each (H, T) (Section 3.2; Tables 1 and 2).
- [BHL] validate their generator by quantiles of the absolute implied-vol difference from a characteristic-function benchmark over all grid points and 500 parameter draws; at their adopted setting the table gives 0.10 at the 95th and 0.15 at the 99th percentile (Section 5.1.3, Table 2).
- [BRST] treat the time-discretisation error and the Fourier quadrature error as two parts of one prescribed tolerance (abstract).
- [BL] argue that Fourier inversion with settings fixed in advance produces spurious smiles, and that the same settings cannot serve every strike and maturity (abstract; Section 6).
- [DFF04] state that the fractional Adams method's error has asymptotic expansions usable with Richardson extrapolation (abstract).

**Arithmetic on recorded numbers.**

- G1's allowance s/5 is 0.02, 0.06 and 0.20 vol point at s = 0.1, 0.3 and 1.0. The repository's existing 0.2 vol-point gate is twice the smallest noise level and ten times that level's s/5.
- If the error behaves as C·N^(−p), the error left at 2N is the doubling change divided by (2^p − 1): equal to the change at p = 1, 0.61 of it at p = 1.4, 0.49 at p = 1.6, a third at p = 2. This holds only once the scheme is in its asymptotic regime; D41 and D55 record non-finite values just below a threshold N, where no order applies.
- If pricer and emulator errors add, the distance from the emulator to rough Heston is bounded by e + τ, not by e alone. With e ≤ s/5, a pricer tolerance τ of s/5 gives 0.4·s in total; s/10 gives 0.3·s.
- A relative tolerance is not commensurate with a noise level in vol points: with ATM vol roughly √ξ₀, 1% relative is about 0.03 vol point at ξ₀ = 0.001 and 0.5 vol point at ξ₀ = 0.25.

**Options.**

| Option | What it means | What it costs |
| --- | --- | --- |
| Every quote, absolute change in vol points, one number (L139 read literally; the draft names no unit) | Keep L139's wording and state one number | The unit matches s and G1's e. One number must be small against the smallest noise level; against s = 1.0 it is then ten times stricter than needed |
| A tolerance tied to each noise level | State it as a fraction of s, mirroring G1 | G0 passes or fails per noise level. A failure at s = 0.1 would restrict the Q1 claim there without stopping Q2. A second ratio to defend alongside G1's |
| One budget shared with G1 | Require pricer change plus emulator error together to stay within the G1 allowance | Closes the gap that G1 compares the emulator with the pricer, not with the model. Leaves less room for the emulator |
| Relative change, maximum over strikes | abs(ΔIV) / IV, as [HK] report | Not commensurate with a noise level in vol points across the ξ₀ range |
| Reuse the repository's existing gate | `calibrate_btc.precheck`: below 0.2 vol point between N and N + 2000, ATM only | Already coded and used for P1. It checks one quote per maturity and is not a doubling test, so it departs from L139's wording |
| A quantile or root-mean-square over the surface | Replace "every quote" with a high quantile, as G1 does and as [BHL] report | Tolerates isolated wing quotes that move more. Departs from "every quote" |

**Established standard.** None found. No source read states a convention for a step-doubling tolerance on the rough Heston Riccati solve. A related statement read at source, which is not a convention: [DFF04]'s abstract says asymptotic expansions of the method's error may be used with Richardson extrapolation to obtain faster-converging variants (abstract).

**To defend it, you would need to be able to say.** When `N_riccati` doubles and a quote moves by less than my tolerance, how large can the pricer's remaining error still be, in what unit and over which quotes, and why is that small enough beside my smallest noise level and the G1 allowance for me to call the emulator rough Heston?

**Checklist boxes.** "Gate tolerances declared" (L176).

**Not confirmed.** The error order of the fractional Adams method: the body of [DFF04] is behind a paywall and only the abstract was read. The unit, strike and maturity behind D30's fit. Whether the existing tests pass today: they were read, not run.

## 14. Gate G1: the ratio e ≤ s / 5 (draft L140)

**The choice.** L140: "**G1 — emulator.** On held-out pricer points, the 99th-percentile absolute implied-vol error e is reported. Q1 is claimed for rough Heston only at noise levels with e ≤ s / 5. **OPEN — the ratio.**"

**Decided as.** A value now.

**What depends on it.**

- The scope of Q1 and of novelty claim (i), the identifiability map (L32, L49, L156). At a noise level that fails G1 the map describes the emulator only.
- Item 4: the ratio and the smallest s set the accuracy the emulator must reach.
- Item 13: held-out pricer points are only as accurate as the pricer's own convergence.
- The pooled hypotheses (L124). Q2 is unaffected by G1 (L48), but pooled results would mix levels where the emulator is shown to be rough Heston to tolerance and levels where it is not.

**Settle first.** Item 5 (the thresholds are s times the ratio), items 2 and 3 (they define the quotes the percentile is taken over), and item 13 (it sets the floor below which e cannot be measured).

**What the repository shows.**

- The code's implied-vol units are decimals: `sigmas=(0.001, 0.003, 0.005)` for 0.1, 0.3 and 0.5 vol point (`layer4_calibrate_surface.py:223`, `:237`). So s/5 at the smallest level is 0.0002 in code units.
- The only pricer-convergence tolerance already in the repository for implied vols is `atol=2e-3`, 0.2 vol point (`calibrate_btc.py:132`, `:149`).
- A convergence law is recorded at one mild point (H = 0.10, ν = 0.20): "`err ≈ 0.68·(T/N_riccati)^1.60`" (`docs/gate_checks/layer4_convergence_gate_check.md:48-51`). The line does not state whether the error is in price or implied-vol units.
- The reason for the value 5 is not recorded in the draft or the ROADMAP.

**What the literature does.**

- [BHL] report the same kind of statistic, for their simulator against a characteristic-function benchmark and not for an emulator: quantiles of the absolute implied-vol difference over 500 parameter draws, pooled over all grid points. At their adopted setting the printed values are 0.02, 0.07, 0.10 and 0.15 at the 50th, 90th, 95th and 99th percentiles, which the text reads in basis points of implied vol, 0.15 being 15 basis points or 0.15 vol point (Section 5.1.3, Table 2; Appendix B.1).
- [HMT] judge emulator accuracy against the Monte Carlo error of the labels and against market spreads, giving a spread of about 0.2% in implied-vol terms on the most liquid options under a year as a yardstick. No percentile of absolute error and no ratio to a noise level is given (Section 4; Section 4.1).
- [BHMST] report calibration-fit figures, not emulator-error percentiles, and in their Bayesian experiment treat a posterior that concentrates on the true parameters as the evidence that the network is accurate enough (Sections 5.2, 5.3).
- [SPAJH], in cosmology, define emulation error relative to the data noise and report 68th, 95th and 99th percentiles over a test set. They also state that test-set residuals are not a sufficient accuracy test, that an emulator with small residuals can still give biased posteriors, and that the accuracy threshold is set by the inference the emulator serves (Section 4.1; Appendix C; Section 5.2).
- [VGB]: a Bayes linear approach known as history matching, which uses an iterative succession of emulators and takes account of function uncertainty, observational error and other sources of uncertainty in one unified treatment (abstract).

**Arithmetic on the draft's numbers.**

| Ratio | Threshold at s = 0.1 | at s = 0.3 | at s = 1.0 | Variance inflation if the error were independent noise | Worst-case coherent shift over 35 quotes |
| --- | --- | --- | --- | --- | --- |
| 1/3 | 0.033 vol point | 0.10 | 0.33 | +11% | about 2.0 noise standard deviations |
| 1/5 (the draft) | 0.02 | 0.06 | 0.20 | +4% | about 1.2 |
| 1/10 | 0.01 | 0.03 | 0.10 | +1% | about 0.6 |

- The variance column is an illustration only: emulator error is a fixed function of θ, so it acts as a bias, not as noise. The last column is an upper bound from the definitions (every quote at the bound with the worst alignment). It shows that a per-quote percentile does not by itself bound the effect on the likelihood.
- If errors were independent across quotes, up to about 30% of 35-quote surfaces could contain at least one quote beyond the 99th percentile (1 − 0.99³⁵).
- A 99th-percentile disagreement of the size [BHL] report between two rough Heston pricers, 0.15 vol point, would pass the draft's gate only at s = 1.0. This compares different objects: their figure is simulator against benchmark on their box and grid, not an emulator error and not a forecast for P5.
- The repository's existing 0.2 vol-point tolerance equals G1's threshold at s = 1.0 and is ten times the threshold at s = 0.1.

**Options.**

| Option | What it means | What it costs |
| --- | --- | --- |
| 1. e ≤ s/5 at the 99th percentile (the draft) | L140 as written | Where it fails, Q1 is not claimed for rough Heston at that level (stated in the draft). It leaves 1% of quotes unbounded, and at s = 0.1 it needs the pricer itself converged well inside 0.02 vol point |
| 2. A looser ratio, s/3 | Thresholds 0.033, 0.10, 0.33 | Easier for the emulator and for G0. The claim that the emulator's contraction is rough Heston's is weaker |
| 3. A tighter ratio, s/10 | Thresholds 0.01, 0.03, 0.10 | At s = 0.1 it requires 1 basis point of vol at the 99th percentile, which the pricer's own convergence would have to beat by a margin. More likely to leave s = 0.1 without a rough Heston claim |
| 4. Keep a ratio but change the statistic | A whole-surface measure, for example a percentile over θ of ‖G̃(θ) − G(θ)‖₂ relative to s | The Gaussian likelihood depends on the emulator only through the sum of squares over a surface's quotes, so a surface norm maps directly onto log-likelihood error. It needs its own number |
| 5. Gate at the posterior level on pilot surfaces | Compare the posterior of H through the emulator with one through the pricer | Tests the quantity Q1 reports, as [SPAJH] argue it should. Each pricer posterior costs hours to days, so only a handful could be checked; it adds a tolerance to declare |
| 6. No threshold: report e and claim Q1 for the emulator only | Keep "Q1 describes G̃" and drop the conditional claim | Nothing to defend numerically. Novelty claim (i) would then be a statement about an emulator |
| 7. Carry emulator error in the likelihood | Add a predictive variance to s². [KOH] and [VGB] treat several sources of uncertainty within one analysis (abstracts only); neither abstract describes this mechanism, which is this sheet's reasoning | Changes the object of study and complicates "Q2 is exact". A larger design change than the OPEN item asks for |

Consequences not attributed to a source are reasoning from the draft's definitions.

**Established standard.** None found. No source read states a standard ratio of emulator error to observation noise.

**To defend it, you would need to be able to say.** Why is an emulator error of one fifth of the noise, at the 99th percentile of quotes, small enough that the contraction I measure on the emulator is the contraction rough Heston would give, and what do I say about the one quote in a hundred beyond it and about errors that line up across a surface?

**Checklist boxes.** "Emulator form and gate declared" (L170); "Gate tolerances declared" (L176).

**Not confirmed.** The units of [BHL]'s Table 2: read as vol points from the text (Appendix B.1), since the table itself does not label them. No wider search for a published standard was made.

## 15. Gate G2: the tolerance for MCMC against a grid posterior (draft L141)

**The choice.** L141: "**G2 — reference.** On pilot surfaces the MCMC posterior of H matches a brute-force grid posterior on G̃. Cost per posterior is measured and fixes the test-set size. **OPEN — tolerance.**"

**Decided as.** Mixed. The measure of agreement and its tolerance can be values now. The grid's size depends on the emulator's cost per evaluation, which is unrecorded until the emulator exists.

**What depends on it.**

- The test-set size: "Set from the reference's measured cost at G2, written in as an amendment before any network is trained" (L92). Through it, the width of every bootstrap interval (L130) and the number of pairs behind a bin at the 10% floor (L99).
- The standing of every measure that uses the reference: c (L77–79), d (L105), S and E (L106–107) and m (L108). G2 is the only gate that checks the reference.
- The claim "Q2 is exact" (L48) and the null's "within tolerance" (L126).
- Item 7: G2 is where the settings declared there are shown to be adequate or not.
- Stage 1, produced "From the reference alone" (L97).

**Settle first.** Item 7: G2 tests those settings. If they are not declared first, the gate and the settings can be adjusted to each other on pilot output. Item 4 and G1: the grid is computed on G̃, so the emulator is frozen first, and its cost fixes the affordable grid. Items 9 and 12: a tolerance in H, c or m has meaning only against the cut-offs and the hypothesis numbers. Item 5: which noise levels the pilot surfaces cover cannot be stated until the levels are fixed. Item 6, if the gate is to cover the posteriors under A and B. Items 1 to 3: they define the domain and the likelihood the grid tabulates. Item 10: no compute budget is stated that turns a cost into a number of test θ.

**What the repository shows.**

- The gate is new in v2. v1's G2 was a pricer check: "**G2 — The pricer is right.** RoughVolLab's characteristic-function engine passes its existing verification tests" (v1 L158).
- G0 and G3 state what happens on failure: "If this fails, the box or the grid is shrunk here" (L139); "If none within budget passes, P5 stops" (L142). G2 does not.
- "Pilots use other seeds and never touch the sealed set" (L93); "No pilot output enters the results" (L137).
- Under the proposed observable "The forward map is G: θ → 35 numbers" (L43). That is a fixed function of θ, so one table of G̃ on a θ-grid would serve every surface and noise level.
- The box the grid must cover: "H in [0.02, 0.48], ν in [0.05, 1.00], ρ in [−0.99, 0.00], ξ₀ in [0.001, 0.25]" (L37), widths 0.46, 0.95, 0.99 and 0.249.
- How narrow the other three directions may be: D37, single smile, "ξ₀/ρ **rock-solid** (rel spread ~1–2%), ν **stable** (2–10%)" (ROADMAP L839); D38, surface, "ν/ρ/ξ₀ stay tight throughout (surface ≤10%)" (ROADMAP L850). These are optimiser scatters at one θ = (0.10, 0.35, −0.70, 0.04) (`layer4_calibrate_surface.py:38`), not posterior widths.
- No grid posterior, sampler or emulator exists (ROADMAP L1031–1032, D55).

**What the literature does.**

- No source read gives a numerical tolerance for agreement between an MCMC posterior and a grid posterior.
- [VGSCB] present their convergence targets as first-level checks and say the Monte Carlo standard errors of the quantities of interest must be judged small enough for the application. They give the standard error of a mean and a method for that of a quantile (Section 2; Section 3.2, Eq. 5; Sections 4.3–4.4). [Stan-RM] gives the same formula for the mean ("Posterior Analysis").
- [LBGGM] checked their reference posteriors without a grid: they generated the reference samples twice for every task and observation and found the two sets indistinguishable by a classifier two-sample test, the metric they prefer. They report that another distance, MMD, was sensitive to its hyperparameters (Section 2; Appendix B.1).
- Neither rough-volatility Bayesian paper checked its sampler against a grid or reported sampler diagnostics: [BHMST] show marginal plots only (Sections 4.2.1, 5.3); [BHL] validate by coverage of credible regions on held-out simulations and by simulation-based calibration checks (Section 5.2; Appendix E.2.3).
- If the gate is also to cover the reweighted posteriors, the published diagnostic for that step is the Pareto-shape threshold of [PSIS] (Algorithm 1).

**Arithmetic on recorded numbers.**

- A uniform grid over the whole box with n points on each of four axes:

| n per axis | Emulator evaluations | Spacing in H | Table size (35 outputs, 8 bytes) |
| --- | --- | --- | --- |
| 10 | 10,000 | 0.051 | 0.003 GB |
| 20 | 160,000 | 0.024 | 0.04 GB |
| 50 | 6.25 million | 0.0094 | 1.75 GB |
| 100 | 100 million | 0.0046 | 28 GB |
| 200 | 1.6 billion | 0.0023 | 448 GB |

  Doubling n multiplies the cost by 16. The emulator's time per evaluation is unrecorded, so no wall-clock figure can be given.
- Grid points per posterior standard deviation of H: at sd 0.017 (c = 0.9), 1.8 for n = 50 and 3.7 for n = 100; at sd 0.038 (c = 0.5), 4.0 and 8.2. A full-box grid that is affordable is comfortable only for Flat posteriors.
- The other axes are tighter. D37's 1% to 2% at ξ₀ = 0.04 is a spread of 0.0004 to 0.0008 on a range of 0.249, which would take roughly 300 to 600 grid points for one point per spread; D38's "≤10%" would take about 60. Order of magnitude only: optimiser scatter, not a posterior width. So at low noise a brute-force grid is feasible only if it is local to the posterior or non-uniform, which presupposes knowing where the posterior is.
- What a disagreement means in the hypotheses' units. In the mean of H: 0.001, 0.002 and 0.005 are 10%, 20% and 50% of H-a's 0.01, and 0.005, 0.01 and 0.025 in S (dividing by 0.20) if one of the two prior-specific means is affected. In the standard deviation: a relative error r moves c by about 2r(1 − c), so 5% moves c by 0.01 at c = 0.9 and by 0.05 at c = 0.5. In interval mass: compare directly with H-d's margins of 0.02 and 0.05. The example sizes are illustrations.
- Chance disagreement. Even a correct sampler differs from the grid by about one standard error, sd/√ESS. For a tolerance τ on the mean to exceed twice that, the pilot needs ESS ≥ (2·sd/τ)²: for τ = 0.001, about 1,156 at sd 0.017 and 5,776 at sd 0.038. So the pilot run length and the tolerance have to be set together. The factor of two is an illustration.

**Options.**

| Choice | Option | What it costs |
| --- | --- | --- |
| As drafted | "matches", tolerance OPEN | Not yet a gate that can pass or fail: no measure, tolerance, grid, pilot set or prior is named. The comparison is of the posterior of H only |
| Measure | Difference in the posterior mean of H | Maps directly onto d and H-a and, divided by 0.20, onto S and H-b. Blind to errors in spread and tails |
| Measure | Difference in the posterior variance, expressed as a difference in c | Maps onto the cut-offs. Needs the grid posterior under prior A, which is the uniform-prior grid posterior multiplied by the Beta density, at no extra emulator cost |
| Measure | Interval end points, or the mass the MCMC sample gives to the grid posterior's central 90% interval | Maps onto m and H-d. Tail estimates carry larger Monte Carlo error than the mean |
| Measure | A distance between the two marginals of H, such as the classifier test [LBGGM] use | One number for location, spread and shape, in units no hypothesis uses, so its tolerance needs a separate justification |
| Form of tolerance | Absolute, in H, c or m | Ties the gate to the hypotheses. Can fail by chance if the pilot chain is short |
| Form of tolerance | Relative to Monte Carlo error | Tests that the sampler is unbiased. An arbitrarily imprecise chain passes it |
| Form of tolerance | Both | Separates "correct" from "precise enough". Two numbers to declare |
| Scope | Uniform prior only | The reweighting to A and B and the re-run of L66 are then checked by no gate, although every scored reference quantity is under A or B |
| Scope | Priors A and B as well | No extra emulator evaluations on the grid side |
| Grid | Uniform over the whole box | n⁴ evaluations, reusable across surfaces. On the arithmetic it under-resolves Identified posteriors in H and more so in ξ₀ |
| Grid | Local to each pilot posterior | Resolves narrow posteriors. Depends on first locating the posterior and on truncating it, which is an approximation to bound. Either way the grid's own error has to be shown smaller than the tolerance, for instance by refinement |
| Pilot surfaces | How many, at which θ, at which noise levels | A gate passed only on easy posteriors says nothing about hard ones. The record has an H–ν ridge at the synthetic truth point (ROADMAP L838, L849) and live fits railed at the lower bound of H (ROADMAP L860, L892); item 7's arithmetic adds true H in the tail of prior A or B. Whether any of these is in fact hard on G̃ is not known before the pilot |
| Complement | Two independent reference runs compared, as [LBGGM] did | Cheap and available where a grid is infeasible. Detects irreproducibility, not a bias common to both runs |
| On failure | State what happens | The draft is silent. G0 and G3 each state a consequence |

Consequences not attributed to a source are reasoning from the draft's definitions and the arithmetic above.

**Established standard.** None found for a numerical tolerance between an MCMC posterior and a grid posterior. The nearest guidance at source is [VGSCB]'s statement that the Monte Carlo error of the quantities of interest is what must finally be judged, and [LBGGM]'s precedent of validating a reference by repeating it.

**To defend it, you would need to be able to say.** What is the largest disagreement between the MCMC reference and the grid, in the mean of H, in c and in the probability of a 90% interval, under the priors actually scored, that would leave every verdict in the hypotheses table unchanged; why is my tolerance below it; and how do I know the grid itself is fine enough to be the arbiter on the pilot surfaces where the posterior is narrowest?

**Checklist boxes.** "Gate tolerances declared" (L176). It bears on "Reference sampler and acceptance criteria declared" (L172).

**Not confirmed.** That no published numerical standard exists: none was found in the sources read, which is not proof. Textbook treatments of grid posteriors were not fetched. How many grid points per posterior standard deviation make a grid trustworthy is reasoning, not a sourced figure. Whether the posterior of H on G̃ is unimodal, how curved the ridge is, and how often mass piles against H = 0.02 or 0.48 on synthetic surfaces are all unrecorded. Kolmogorov–Smirnov, Wasserstein and total-variation distances were not checked at source for this use; only the classifier test in [LBGGM] was.

## 16. Gate G3: the tolerance for "stable to a doubling" (draft L142)

**The choice.** L142: "**G3 — training adequacy.** Per calibrator, prior and noise level: ξ₀ recovered with slope above 0.95, and abs(d) on pilot surfaces stable to a doubling of the training set. Training-set size is the smallest that passes. If none within budget passes, P5 stops; architecture is not tuned against H outputs. **OPEN — tolerance.**"

**Decided as.** Mixed. The tolerance, the statistic, the pilot design and the budget are values now. The training-set size is the value the gate then fixes.

**What depends on it.**

- The training-set size of every confirmatory network: "Training-set size is the smallest that passes."
- The study's stop rule: "If none within budget passes, P5 stops".
- H-a (L119). G3 selects the training-set size on abs(d), the quantity H-a scores.
- H-b, through E = S_net − S_exact, and H-c, through S_net (L106–107, L120–121). Any training-size effect left in d passes into S_net and so into E.
- The null's "within tolerance" (L126). G3's is the only tolerance on abs(d) in the draft.
- The order of events at L92, "before any network is trained": G3 pilots train networks.

**Settle first.** Item 15: d needs the reference posterior on every G3 pilot surface, so the reference passes G2 first and its cost is paid on the pilot surfaces too. Item 8: architecture, optimiser, stopping rule, variant and sampler are declared before G3. Item 4 and G1. Items 5 and 6: the gate is applied in twelve cells under the draft's proposals. Item 9, if stability is assessed by bin.

**What the repository shows.**

- v1's G3 used no H output: "Trained on prior A, it must recover the variance level with a tracking slope above 0.95 in every quintile. If it cannot learn a parameter the surface fixes cleanly, its behaviour on H tells us nothing. G3 also fixes the training-set size" (v1 L159). v2 adds the test on abs(d), and drops "Trained on prior A" and "in every quintile".
- v1: "**DECISION — training-set size is set by a pilot, not guessed.** It is the smallest size at which the identifiable parameters are recovered to saturation" (v1 L102).
- d is defined for prior A only: "d = Ĥ_net,A − Ĥ_exact,A" (L105). G3 applies "Per calibrator, prior and noise level".
- The word "budget" occurs once in the draft, at L142. No maximum training-set size, starting size, ladder or compute limit is stated anywhere.
- "Pilots use other seeds and never touch the sealed set" (L93); "No pilot output enters the results" (L137).
- The repository's precedent for stability under a larger budget as the test of convergence: "a real effect is budget-STABLE; one that dissolves as you train harder is a convergence artifact" (ROADMAP L902, D43); and "the EDGES converging does NOT guarantee the finer roughness CONTRAST (a difference-of-differences) converged" (ROADMAP L903).
- Why ξ₀ is the easy parameter: "ξ₀/ρ **rock-solid** (rel spread ~1–2%)" (ROADMAP L839, D37); "ν/ρ/ξ₀ stay tight throughout (surface ≤10%)" (ROADMAP L850, D38).
- One existing stability tolerance in the repository is absolute, against a higher-resolution reference: `atol=2e-3` in implied vol, between N(T) and N(T) + 2000 (`calibrate_btc.py:132`, `:144`, `:149`).

**What the literature does.**

- No source read gives a numerical criterion for stability under a doubling of the training set.
- [HDR+] and [DHRWL] vary the simulation budget in powers of two from 2¹⁰ to 2¹⁷ and train five estimators at each. [HDR+] report that overconfidence is worse at small budgets and that a large budget is no guarantee; they also note that an estimator returning the prior passes a coverage check (Sections 2, 3). [DHRWL] report that the gaps between the plain and balanced estimators shrink as the budget grows (Section 4).
- [MWF] separate data limits from architecture limits by drawing new simulations for every mini-batch and training until saturation (Section 3.1).
- [LBGGM] report results at budgets of 1,000, 10,000 and 100,000 simulations and give no criterion for convergence in training-set size (Section 3).
- [sbi]'s documented stop is a validation-loss criterion. Its calibration tutorial states that passing simulation-based calibration is necessary but not sufficient, and that using the prior as the posterior would pass.
- [BHL] train on 90% of 500,000 pairs, which is 450,000, and judge training by early stopping on validation loss and by coverage checks. No test of sensitivity to training-set size was found by text search (Section 5.2; Appendix E.3).

**Arithmetic on the draft's definitions.**

- H-a's numbers in other units: 0.01 is about 0.19 of prior A's standard deviation and 0.6 of the widest posterior standard deviation in the Identified bin; 0.03 is about 0.56 and 1.8.
- How a tolerance on d passes into E. If d under prior B is defined like d under A, then E = (d_B − d_A)/0.20. If each d can still move by τ with more training, E, and S_net with it since S_exact does not depend on training, can move by up to 10τ: 0.01 at τ = 0.001, 0.05 at τ = 0.005, 0.10 at τ = 0.01. H-b's two numbers are 0.10 apart and H-c's are 0.05 apart. This is the worst case, with opposite signs under the two priors, and G3 as drafted constrains abs(d), not the signed d.
- What one doubling says about the error still to come. If the training-size error fell as a power n^(−p), a change of τ on doubling would leave a remaining error of τ at p = 1, 2.4τ at p = 0.5 and 5.3τ at p = 0.25. The power law is an assumption; the point is that a tolerance on the change is not a bound on what remains.
- Pilot networks. One training-set size and one seed over all twelve cells is 36 networks. Two sizes and one seed, the smallest possible gate, is 72; four sizes and three seeds is 432. The draft states neither the ladder nor the number of pilot seeds.

**Options.**

| Choice | Option | What it costs |
| --- | --- | --- |
| As drafted | "stable to a doubling", tolerance OPEN | Until "stable" has a statistic, a threshold, a pilot design and a budget, the gate cannot be applied mechanically, and the training-set size would be chosen after pilot values of abs(d), H-a's own measure, have been seen |
| Form | Absolute: the median of abs(d) changes by no more than τ between n and 2n | Directly comparable with H-a's 0.01 and 0.03, and through 10τ with H-b and H-c. The pilot must be large enough to measure a change of size τ |
| Form | Relative: a percentage change | Scale-free. Where abs(d) is tiny a percentage is mostly noise; where it is large the same percentage admits a large absolute change |
| Form | Set by the seed-to-seed spread at fixed size | Adapts to the noise present. Needs several seeds at every size. D43 records a case where seed-consistency did not guarantee convergence |
| Form | Equivalence: a bootstrap interval for the paired change lies wholly inside ±τ | A noisy pilot fails instead of passing. Needs a stated number of pilot surfaces and their reference posteriors |
| What is tested | The training objective and the ξ₀ slope only, as in v1 | Keeps every H output out of the gate. D43's risk: the coarse quantity converges while the finer contrast has not |
| What is tested | abs(d) (the draft) | Uses an H output to choose the training-set size. Stability is not the same as smallness, but the choice is made with pilot abs(d) in view |
| What is tested | E and m as well | Addresses D43's point that a difference of differences converges later than its parts. Doubles the pilot's use of H outputs and needs prior-B networks and references under B on the pilot surfaces |
| ξ₀ slope | Against the true ξ₀ (the draft, as far as it says) | A posterior mean shrinks towards the prior mean, so at high noise the exact reference itself could have a slope below one. Against truth the check cannot tell shrinkage from under-training. Whether the reference's slope at s = 1.0 exceeds 0.95 is not known |
| ξ₀ slope | Against the reference's posterior mean of ξ₀ | Compares the calibrator's ξ₀ with the reference's, as d does for H (L105). Needs the reference's ξ₀ posterior on the pilot surfaces |
| Budget | State the first size, the ladder and the largest size | Makes "within budget" and "the smallest that passes" checkable. The studies read use doubling ladders from 2¹⁰ to 2¹⁷ |
| Size across cells | One common size, the largest any cell needs | Keeps prior-A and prior-B networks on equal training, which S and E compare |
| Size across cells | A size per cell | Cheaper. Puts unequal training into the comparison between priors |
| Partial failure | State what happens when some cells, or one calibrator, fail | The draft covers only "none within budget passes" |

Consequences not attributed to a source are reasoning from the draft's definitions and the arithmetic above.

**Established standard.** None found. Nearby practice at source: early stopping on validation loss ([sbi]); budget ladders in powers of two with five estimators per budget ([HDR+], [DHRWL]); training on fresh simulations until saturation ([MWF]). The repository's own precedent is D43: escalate the budget until the fine quantity stabilises.

**To defend it, you would need to be able to say.** What change, in which statistic of abs(d), measured on how many pilot surfaces and seeds, will I call stable; why is that tolerance small enough that a leftover training-size effect could not by itself carry H-a across 0.01 or 0.03, or E across the numbers in H-b, or S_net across those in H-c; what is the largest training set I will try; and what happens if one of the twelve cells, or one of the two calibrators, fails?

**Checklist boxes.** "Gate tolerances declared" (L176). It bears on "Architectures and training procedure declared" (L173).

**Not confirmed.** That no convention exists for a training-size adequacy tolerance: none was found, which is not proof. The rate at which abs(d) falls with training-set size is unknown. Whether the reference's ξ₀ slope against truth exceeds 0.95 at the highest noise level is unknown; nothing was computed. Training cost per network is unrecorded, so no budget in hours can be derived from the repository.

## Points in the draft: undefined wording, statements beyond the record, and conflicts

These are places where the draft's wording is undefined, goes further than the record it cites, or conflicts with another line. They are listed for the author; the sheet changes nothing in the draft. Each is a reading of the text against the repository, and several are reasoning, not recorded fact.

### What was known before freezing (L17–22)

This section has its own checklist box, "Prior-knowledge section checked against the ROADMAP entries it cites" (L167).

- **L20 against D55.** "all at T ≥ 0.5" and "`N_riccati` 2000 to 4000 cured T ≤ 1" go further than D55 records. D55 gives maturity detail only for a follow-up on four of the nine failing points, ATM quote only (ROADMAP L1053–1056).
- **L19.** "The longest tenor is the least informative about H" is recorded for the one-year tenor on live BTC and ETH (D41, D42). It is not a statement about T = 2.00 or about the synthetic grid.
- **L18 against L39.** D38's scatter figures come from its 25-quote noise grid; v2 adopts the 35-quote grid. "v2 uses D38's" does not say which of D38's two strike sets the evidence belongs to.
- **L17.** D37's −0.82 and D38's −0.85 (ROADMAP L838, L849) are cosines between Jacobian columns, not posterior correlations. L17's "degenerate with ν" quotes neither number.
- **L22.** "probably identifies H well" rests on D38: one parameter point, 25 quotes, ten optimiser fits, strikes anchored on the noiseless target. No recorded number is on the grid and observable the draft would freeze.
- **L39, "per D41".** D41 never priced T = 2.00; its longest tenor is 0.986 years (ROADMAP L884). D38's grid contains T = 2.00 (`layer4_calibrate_surface.py:35`) and D38 did not isolate it; D55's probe also priced T = 2.0, for finiteness only (ROADMAP L1053–1056). "It is where the pricer fails most" rests on four failing points, ATM only; D55 gives no per-maturity failure count for the 24-point probe.
- **The 3 October probe and G0.** The probe ran in a sandbox and not on the author's machine, on 24 uniform points without the corners, for timing and finiteness only, and its script is not in the repository (ROADMAP L1044–1056). It did not test what G0 tests. The draft does not say how the probe stands to G0's pilot or whether its points may be reused.

### Parameters, grid and observable (L37–44)

- L37 "at affordable cost" and L139 "Cost per surface is measured": no ceiling is stated.
- L37 "the box may be shrunk": no rule for which dimension or by how much. L139 "the box or the grid is shrunk here": no order between the two.
- L37 against L60, L91 and L106: nothing says what happens to "0.02 + 0.46 · Beta", to the 0.20 denominator or to the sealed draw if the box changes.
- L37 "with the H upper bound lowered from 0.49": no reason is given. "κ fixed at 0.30, per D38 EXP 4": EXP 4 supports keeping κ fixed; the reason for the value 0.30 is not recorded.
- L43 "θ's own standardised moneyness" does not say which volatility standardises, and does not say that the strikes are withheld from the observer and the networks. "Leak-free" rests on that unstated condition.
- L43 says "35 numbers" while L39 leaves T = 2.00 OPEN. L43 "keeps P1's box" while L37 already lowers the H bound and allows a shrink.
- L44 gives no values for the fixed grid and no narrowed ξ₀ range, and "priceable" is undefined. Option 2 cannot be chosen as written.
- L44 "as in the deep-calibration literature" is true of [HMT] and [BHMST]. [BHL], the rough Heston paper v1 calls the state of the art, use standardised moneyness.
- "vol point" is never defined. The code's convention is 0.1 vol point = 0.001 in decimal implied vol.

### The emulator and G1 (L46–52, L140)

- L52 makes no proposal: form, count and design are three separate decisions.
- "Frozen" and "built once" (L46) are not made operational: whether G1 runs before the emulator is declared frozen, whether a rebuild is allowed if G1 fails, and how many attempts.
- L140 "held-out pricer points" are not distinguished from any validation set used while fitting the emulator, and L91–93 protect the sealed test θ from pilots but say nothing about the emulator's design points.
- The draft does not say what the emulator does with points that remain non-finite after G0.
- L50 names `calibrate_btc._cached_cf`. In the code the cached model also replaces `N_riccati` by a per-maturity schedule, while the probe used a uniform 1000. The draft does not say which pricer G0 tests or builds the emulator.
- L48 "Q2 is exact" is exact with respect to the emulator. L49 and L140 claim Q1 for rough Heston on G1's e alone, but G1 compares the emulator with the pricer; the pricer's own error is bounded only by G0.
- L140 does not define the population of the percentile, the number of held-out points, how they are drawn, or the unit. Nothing is said about the 1% of quotes above e. The consequence of failing is only partly stated: whether the level is still run, whether it enters the pooled hypotheses, whether the emulator may be rebuilt.
- L49 "small against s" and L140 "e ≤ s / 5": the reason for 5 is not recorded.
- L139 and L140 aggregate differently, "every quote" against a 99th percentile, without saying why.

### Noise (L54)

- "0.1 is P1's level": P1's experiments ran 0.1, 0.3 and 0.5. v1 L106 calls 0.1 the level that corrupts H (the single smile); L22 says that at 0.1 the surface probably identifies H well (the surface). The phrase does not say which finding it means.
- A reason is given for 0.1 and for 1.0, none for 0.3 and none for leaving out 0.5. No synthetic measurement exists at 1.0.
- "1.0 is the size of the live misfit": live fit errors are residuals of a fitted model, structured in the put wing and long tenor, while L54's noise is independent Gaussian. The sentence can support a magnitude only.
- L91 "one noise draw per θ per noise level" does not say whether one standard-normal vector is rescaled across levels, as D38's code does, or drawn afresh. No noise seeds appear although L91 calls them "declared".
- The noise is absolute across a box where ξ₀ runs from 0.001 to 0.25; the draft does not comment.
- v1 L112 held open a "second, tenfold-cleaner noise level"; v2 does not mention it.

### Priors (L56–64)

- The reason for Beta(2.5, 9) and Beta(7.5, 4), and for a + b = 11.5, is not recorded.
- The standard deviations are unequal and the draft does not comment. v1 L89 registered priors that "differ only in where they put their mass".
- L56 "Same support": true, but prior A gives 2.0% of its mass to H ≥ 0.25 while the test set is uniform.
- L77 and L108 use prior A alone for the bins and for m. No reason for A is recorded, and contraction under B differs for the same surface.
- L106 hard-codes 0.20 and L60 hard-codes the H range while L37 leaves the box OPEN.
- v2 drops v1 L98's sentence on why the gap is 0.20 without saying so.

### The exact reference and G2 (L66, L141)

- "effective sample size" is undefined: the chain's figure, the importance weights' figure, or a combination; for which quantity and which prior.
- "a direct re-run" is undefined: its target, settings and acceptance criteria, and what happens to a posterior that still fails.
- The OPEN list names an "R-hat ceiling" before a sampler is chosen; the R-hat of [VGSCB] compares between-chain and within-chain variance and comes with a recommendation of at least four chains (Section 2; Section 3.1), although [VGSCB] note that it could be computed from a single chain where one is known to suffice (Section 3.1) and [VK] say their version can be calculated for a single chain (abstract only).
- Nothing scored uses the uniform-prior posterior that L66 samples. The draft treats the re-run as an exception, with no estimate of how often it fires.
- Which parameters the criteria apply to, H alone or all four, is not stated.
- L79: Var_prior(H) is not said to be the analytic value or an estimate. L108: the draft does not say how m is computed from the reference or that tail accuracy is checked.
- "exact" (L48, L66, L77, L108) and "within tolerance" (L126): the reference is a Monte Carlo estimate, and the draft has no statement of the error it is allowed.
- L141: "matches" has no measure; the comparison is of the posterior of H only; the prior under which it is made is not stated; "brute-force grid" has no resolution, extent or refinement check; "pilot surfaces" are not specified; no consequence of failure is stated.
- L141 and L92: no compute budget or formula turns "cost per posterior" into a number of test θ.
- Apart from L11 and the box "Reference sampler and acceptance criteria declared" under "Before freezing" (L165, L172), the draft does not state that the reference's settings are fixed before G2 is run.

### Calibrators and G3 (L68–73, L142)

- "five per calibrator" is ambiguous for the ensemble: a training seed is a whole five-member ensemble, or the five members are the five seeds.
- L68–71 do not say what the calibrators output. G3 needs a ξ₀ estimate from each; a ratio estimator can be joint over four parameters or marginal in H.
- L71: a posterior mean and an equal-tailed interval need draws or a density, but no sampler is named for the ratio estimator. Its error would be counted as calibrator error.
- L70: a central Gaussian interval is applied to a bounded, skewed quantity. In the limit where the posterior is prior A, an interval with exactly the right mean and variance has m = 0.9225, inside H-d's refutation band.
- L71 uses an equal-tailed interval and L70 a central Gaussian one; [HDR+] compute expected coverage on highest-posterior-density regions (Section 2.2). The reason for the draft's two forms is not recorded.
- Whether the training set is fixed or redrawn, and whether quote noise is drawn once per training surface or afresh each epoch, is not stated.
- L148 excludes "any correction method" without saying whether a conservative-by-design variant, post-training calibration of a classifier, or ensembling of ratio estimators is one.
- Neither protocol names the "standard simulation-based-inference tooling" (v1 L69), or says whether the ensemble uses the optional adversarial step of [LPB].
- L142 "stable to a doubling" is undefined: the statistic, absolute or relative, the threshold, the pilot surfaces, the seeds per size. "within budget": no budget, largest size, starting size or ladder is stated.
- L105 defines d for prior A only, but G3 applies per prior.
- L142 "ξ₀ recovered with slope above 0.95": the regression is not stated. v1 had "Trained on prior A" and "in every quintile" (v1 L159); v2 drops both. Against the true ξ₀ the criterion mixes Bayesian shrinkage with under-training.
- "Training-set size is the smallest that passes": per cell or one common size is not stated.
- The gate says architecture is not tuned against H outputs, yet it selects the training-set size using abs(d), H-a's own measure. It tests abs(d) only, while E and m are the finer quantities the hypotheses use.
- "If none within budget passes, P5 stops": the outcome when some cells pass and others do not is not stated.
- G3 needs reference posteriors on its pilot surfaces; neither L141 nor L142 says how many pilot surfaces there are.

### Stratification, test sets and Stage 1 (L77–99)

- No reason is recorded for 0.9 and 0.5, or for moving from v1's equal-occupancy quintiles to fixed cut-offs.
- L79: nothing bounds c below zero. If a pair had posterior variance above prior A's, the table would file it under Flat without saying so; whether that can happen on this model was not computed.
- L87 "c depends only on the observed surface and the reference": it also depends on prior A and on the emulator. The sentence presumably means "on no network".
- L92 "before any network is trained": G3 pilots train networks.
- L91: the seed is fixed but the number is not; the generator and the order of draws are not stated, so which θ a set of N draws contains is not yet defined. If the box shrinks at G0 the draws change.
- The number of pairs is never stated; with three levels it is three times the number of θ.
- L99 "10% of pairs" does not say whether pairs are pooled over noise levels or counted per level. It introduces "untestable", but L131 defines three verdicts and L160 promises every hypothesis "with verdict and interval".
- L99 "not adjusted" does not say what may not be adjusted, or whether bin shares seen in the G2 pilot may inform items 5, 9 or 10.
- No stop rule covers a number of test θ that is affordable but too small.

### Measures, hypotheses and the null (L105–126)

- L109 with L121: "above the seed floor" has no numerical meaning as written. The seed floor is a signed per-pair quantity, so its median is zero by symmetry. Not stated: an absolute value or a quantile, which prior, which seed pairs, and what happens when the interval is above 0.10 but not above the floor.
- L106: S_net and E need one A-trained and one B-trained network. Whether seeds are matched by index or crossed is not stated.
- L105, L108: d and m are defined for prior A only; d under prior B is never defined, although E is the difference of the two mean errors divided by 0.20. Ĥ_exact is not explicitly defined as the posterior mean.
- L121 against L9 and L111: H-c is scored on S_net, which is not a comparison with the reference. The reason for S_net and not E in this row is not recorded.
- L124 "In the flat bin the reference itself has S near one": that holds as c tends to zero. The Flat bin is every c below 0.5, where on a Gaussian approximation the reference's S starts at about 0.39 to 0.56.
- L124 pools over noise levels although L68 trains a separate network per level. Whether a seed is one unit across levels and whether the per-level breakdown carries verdicts are not stated.
- L126: "tolerance" is undefined; the null says "in every bin" while each hypothesis scores one bin; no rule links the eight verdicts to the null. A perfectly faithful calibrator would not necessarily be Refuted on H-c.
- L122: the bands are asymmetric about 0.90 and no reason is recorded; H-d states no direction.
- L131 against L122 and L121: for H-d the "expected value" is a two-sided region and the "refutation threshold" a band; H-c has two expected values. The rule speaks of one value and one threshold. That a tight interval lying wholly in a gap is Inconclusive was said in v1 (v1 L141, L171) and is not repeated.
- L77 with L106–107: the bins use contraction under A for all measures, including those involving the B-trained network.
- L124: "No aggregate verdict", and no statement on multiplicity.

### Analysis plan (L130–133)

- L130: the interval type and the number of resamples are not stated. Every seed's network is scored on the same pairs, so pairs are crossed with seeds, not nested "within seed"; pairs that share a θ are not independent. The resampling unit, how seed-level values become one point estimate, and what is resampled for S and E are not stated. The reference's own Monte Carlo error is not resampled.
- L132: no expected sign, threshold or verdict for the Spearman correlation, and no statement of whether it is computed per calibrator, per level or pooled.

### Gate G0 (L139)

- "Existing verification tests pass": the tests are not named. None prices T = 2.00 or ρ below −0.70, and none refines `N_riccati` in the rough regime; the two inputs near the low-H, high-ν edge are checked for output length, and one of them also for the penalty that replaces a non-finite residual; neither is checked for finite quotes (items 1 and 13).
- "changes by less than a stated tolerance when `N_riccati` doubles": no unit, no starting value, and no statement of which value then builds the emulator. Only `N_riccati` is varied, although the code names the inversion settings as a separate source of error.
- "pilot points across the box, including its corners": the box has 16 corners; the number of points, and how many failing quotes fail the gate, are not stated. "If this fails" does not say whether "this" is the tests, finiteness, the tolerance or the cost.
- "every quote is finite": the code returns a non-finite quote by several different routes.

### Differences from v1, and statements set against sources

L161 and v1 L188 require an amendment to state what changed and why. The first six points are differences from v1 that were found; of them D55 records only the first, as noise made a design factor (ROADMAP L1039–1041). The last three set a statement of v1, or the box's ρ range, against a source:

- Three noise levels in place of the single registered level of 0.1 (v1 L106, a ticked DECISION).
- Fixed contraction cut-offs with a floor in place of quintiles of CRB_H computed before scoring (v1 L116, L173).
- A hierarchical bootstrap over seeds and pairs in place of a bootstrap over test surfaces (v1 L165).
- H-c reverses the direction of v1's P3 in the identified regime: v1 expected the shift ratio "below 0.1" in the most identifiable quintile (v1 L136); H-c expects it "above 0.10" in the Identified bin.
- G3 adds a test on abs(d) and drops "Trained on prior A" and "in every quintile" (v1 L159).
- v2 has no counterpart to: "no code may be tuned to reach them" (v1 L143); "If all four predictions are refuted, the null stands" (v1 L151); the home-prior control (v1 L139); the cleaner noise level held open at v1 L112; v1 L98's reason for the 0.20 gap.
- v1 L71 says the ratio estimator is "calibrated by construction" under its training prior. [HDR+] report overconfident amortised ratio estimators at finite budgets (Sections 2, 3; Observations 1–4).
- v1 L41 records that "neither Bayesian paper varies its prior". [BHL] describe applying a different prior at inference time by reweighting (Section 3.1, Eq. 3.7); a text search found the device described and no experiment applying it. That rests on a text search, not on a full reading.
- The correlation range, for the prior-work section: [ER]'s main theorem is stated for ρ above about −0.707, while the box runs to −0.99; [ALP] cover the whole range.

## Leads, not verified

These could not be read at source. Where a page was blocked it was not worked around. None is used as support in the items.

**Pricer and numerics**

- El Euch and Rosenbaum, the published version (Mathematical Finance 29(1), 3–38, 2019): the publisher page was blocked. Whether the published theorem carries the same correlation restriction as the arXiv version is not verified.
- Gatheral and Radoičić, "Rational approximation of the rough Heston solution": publisher and SSRN pages blocked.
- Li and Tao (2009), "On the fractional Adams method", Computers & Mathematics with Applications 58(8): blocked; seen only in [ER]'s bibliography.
- Roache's Grid Convergence Index: publisher page blocked.
- The error order in the body of [DFF04]: paywalled; abstract only.

**Emulation and deep calibration**

- Horvath, Muguruza and Tomas, journal details (Quantitative Finance 21(1), 11–27, 2021): publisher page blocked; seen only in Crossref metadata.
- Whether Bayer et al. (arXiv:1908.08806) was published in 2025, as v1 L29 states: not checked. Only the 2019 arXiv version was read.
- Baschetti, Bormetti and Rossi, "Deep calibration with random grids": cited by [BHL]; not fetched.
- Loeppky, Sacks and Welch (2009), Technometrics 51(4), on design size for Gaussian-process emulators: blocked.
- McKay, Beckman and Conover (1979), Technometrics, on Latin hypercube sampling: failed to load.
- Bastos and O'Hagan (2009), "Diagnostics for Gaussian Process Emulators", Technometrics: not fetched.
- The journal version of [GGMM]: not reached. The arXiv version (v2) was read for its abstract and Sections 2.1 and 2.2 only.

**Noise, priors and contraction**

- Journal details of [KPBV] (Statistics and Computing) and of [SBV] (Psychological Methods): publisher pages not reached.
- Betancourt's case study "Towards A Principled Bayesian Workflow": not read.
- Nott et al. (2020) on prior-to-posterior shrinkage checks, cited by [GVS+]: not fetched.
- The published version of [Hes] (The American Statistician, 2015): not fetched.
- A primary-source figure for bid–ask width in vol points, for index or crypto options: not found.

**Samplers and diagnostics**

- Kong (1992), "A note on importance sampling using standardized weights", Technical Report 348, University of Chicago, and Kong, Liu and Wong (1994): not fetched; known only through [PSIS].
- Gelman and Rubin (1992), Statistical Science 7(4): only the abstract page was reached.
- Skilling (2006) on nested sampling: the page returned a bot check and was not pursued.
- Chatterjee and Diaconis (2018) on importance-sampling diagnostics, cited by [PSIS]: not fetched.
- emcee's journal details, and the journal version of [VK]: not verified.

**Calibrators**

- [HBL] and [DMP] were read as abstracts only; nothing about their architectures is reported from them.
- Appendix H of [LBGGM], on hyperparameters for ratio estimators: not read.
- Other libraries that might count as standard tooling: no documentation fetched.
- Whether sbi installs in the Layer 3 environment: not tested.

**Hypotheses and analysis**

- Kruschke (2018), on deciding with an interval and a region of practical equivalence, and Kruschke and Liddell (2018): publisher pages blocked. These would be the nearest published convention to the whole-interval verdict rule at L131.
- Lakens (2017) on equivalence tests: not attempted.
- Ren et al. (2010), Journal of Applied Statistics, on bootstrapping hierarchical data: blocked.
- The journal version of [CGM] (doi:10.1162/rest.90.3.414): blocked; verified on the working-paper page only.
- Davison and Hinkley (1997), Bootstrap Methods and their Application: the publisher page confirms the book and its chapter list only.
- The OSF Preregistration template's wording on directional and multiple hypotheses: seen only in a search summary.
- The full text of [ASC+]: not read; abstract only.

## Checklist map

The draft's "Before freezing" list has twelve boxes (L167–178). None is ticked, and this sheet ticks none.

| Box | Draft line | Items it waits on |
| --- | --- | --- |
| Prior-knowledge section checked against the ROADMAP entries it cites | L167 | No OPEN item. See "Points in the draft", the section on L17–22 |
| Grid and observable convention chosen | L168 | 2, 3 |
| Noise levels accepted or changed | L169 | 5 |
| Emulator form and gate declared | L170 | 4, 14 |
| Beta parameters accepted or changed | L171 | 6 |
| Reference sampler and acceptance criteria declared | L172 | 7 |
| Architectures and training procedure declared | L173 | 8 |
| Contraction cut-offs accepted or changed | L174 | 9 |
| Every number in the hypotheses table defensible in the author's own words | L175 | 12 |
| Gate tolerances declared | L176 | 13, 14, 15, 16 |
| Prior-work section written from the papers themselves | L177 | No OPEN item. To be written from the author's own reading (L156) |
| OSF registration amended | L178 | No OPEN item. The author's act at freeze (D55) |

**Three boxes no OPEN item covers:** the prior-knowledge check (L167), the prior-work section (L177) and the OSF amendment (L178).

**Three OPEN items no box names:** the box and its shrink rule (item 1), the number of test θ (item 10) and the 10% floor (item 11). Item 10 is settled by the amendment L92 describes, after G2.

**For the prior-work box.** L156 says the section must cover Bayesian prior-sensitivity analysis and the simulation-based-inference literature on unfaithful and misspecified amortised posteriors. Sources read for this sheet that fall under those headings: [GVS+], [KPBV] and [EOS+] on prior sensitivity; [HDR+], [DHRWL] and [SBKR] on unfaithful or misspecified amortised posteriors; and [LBGGM], a benchmark of approximate posteriors against reference posteriors. They were read for the items above, some as abstracts only. The section itself is to be written from the author's own reading.

## Sources

Each was read at the address given while this sheet was prepared. "Abstract only" means the statement in the items rests on the abstract or listing page. arXiv pages are used as in `OVERLEAF/_refs/new_references_verified.tex`; journal details are given only where the page read shows them.

**Rough volatility: models, pricing and calibration**

- [ALP] Eduardo Abi Jaber, Martin Larsson, Sergio Pulido. Affine Volterra processes. arXiv:1708.08796v3 (Annals of Applied Probability 29(5), 3155–3200, 2019, per the arXiv page). https://arxiv.org/abs/1708.08796
- [BHL] Damiano Brigo, Raphaël Huser, Dan Leonte. Uncertainty and Explainability in Deep Rough Volatility: A Neural Information-Theoretic Posterior Approach. arXiv:2609.31570v1, 25 Sep 2026. https://arxiv.org/abs/2609.31570
- [BHMST] Christian Bayer, Blanka Horvath, Aitor Muguruza, Benjamin Stemper, Mehdi Tomas. On deep calibration of (rough) stochastic volatility models. arXiv:1908.08806v1, 22 Aug 2019. https://arxiv.org/abs/1908.08806
- [BL] Svetlana Boyarchenko, Sergei Levendorskiĭ. Correct implied volatility shapes and reliable pricing in the rough Heston model. arXiv:2412.16067v1, 20 Dec 2024. https://arxiv.org/abs/2412.16067
- [BRST] Chiheb Ben Hammouda, Abderrahmene Ben Romdhane, Michael Samet, Raúl F. Tempone. Single- and Multilevel Quadrature with Error Control for Fourier Pricing under the Rough Heston Model. arXiv:2609.00438, 31 Aug 2026. https://arxiv.org/abs/2609.00438 (abstract only)
- [BS] Christian Bayer, Benjamin Stemper. Deep calibration of rough stochastic volatility models. arXiv:1810.03399v1, 8 Oct 2018. https://arxiv.org/abs/1810.03399
- [CGP] Giorgia Callegaro, Martino Grasselli, Gilles Pagès. Fast Hybrid Schemes for Fractional Riccati Equations (Rough is not so Tough). arXiv:1805.12587v4, 18 Feb 2020. https://arxiv.org/abs/1805.12587
- [DFF04] Kai Diethelm, Neville J. Ford, Alan D. Freed. Detailed Error Analysis for a Fractional Adams Method. Numerical Algorithms 36, 31–52, 2004. doi:10.1023/B:NUMA.0000027736.85078.be. https://link.springer.com/article/10.1023/B:NUMA.0000027736.85078.be (abstract only)
- [EGM] Mnacho Echenim, Emmanuel Gobet, Anne-Claire Maurice. Unbiasing and robustifying implied volatility calibration in a cryptocurrency market with large bid-ask spreads and missing quotes. arXiv:2207.02989v1, 6 Jul 2022. https://arxiv.org/abs/2207.02989
- [ER] Omar El Euch, Mathieu Rosenbaum. The characteristic function of rough Heston models. arXiv:1609.02108v1, 7 Sep 2016. https://arxiv.org/abs/1609.02108
- [HK] Paul P. Hager, Dörte Kreher. Expanding the rough Heston model in H. arXiv:2606.16619, 15 Jun 2026. https://arxiv.org/abs/2606.16619
- [HMT] Blanka Horvath, Aitor Muguruza, Mehdi Tomas. Deep Learning Volatility. arXiv:1901.09647v2, 22 Aug 2019. https://arxiv.org/abs/1901.09647

**Emulation and design**

- [GGMM] Maximilian Gaß, Kathrin Glau, Mirco Mahlstedt, Maximilian Mair. Chebyshev Interpolation for Parametric Option Pricing. arXiv:1505.04648v2, 8 Jul 2016. https://arxiv.org/abs/1505.04648 (abstract; Sections 2.1 and 2.2)
- [KOH] Marc C. Kennedy, Anthony O'Hagan. Bayesian Calibration of Computer Models. Journal of the Royal Statistical Society Series B 63(3), 425–464, 2001. doi:10.1111/1467-9868.00294. https://academic.oup.com/jrsssb/article/63/3/425/7083367 (abstract only)
- [SciPy-qmc] SciPy documentation, `scipy.stats.qmc.Sobol` and `scipy.stats.qmc.LatinHypercube`. https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.qmc.Sobol.html ; https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.qmc.LatinHypercube.html
- [SPAJH] A. Spurio Mancini, D. Piras, J. Alsing, B. Joachimi, M. P. Hobson. COSMOPOWER: emulating cosmological power spectra for accelerated Bayesian inference from next-generation surveys. arXiv:2106.03846v2 (MNRAS 511(2), 1771–1788, 2022, per the arXiv page). https://arxiv.org/abs/2106.03846
- [SWMW] Jerome Sacks, William J. Welch, Toby J. Mitchell, Henry P. Wynn. Design and Analysis of Computer Experiments. Statistical Science 4(4), 409–423, 1989. doi:10.1214/ss/1177012413. https://projecteuclid.org/journals/statistical-science/volume-4/issue-4/Design-and-Analysis-of-Computer-Experiments/10.1214/ss/1177012413.full (abstract only)
- [VGB] Ian Vernon, Michael Goldstein, Richard G. Bower. Galaxy formation: a Bayesian uncertainty analysis. Bayesian Analysis 5(4), 619–669, 2010. doi:10.1214/10-BA524. https://projecteuclid.org/journals/bayesian-analysis/volume-5/issue-4/Galaxy-formation-a-Bayesian-uncertainty-analysis/10.1214/10-BA524.full (abstract only)

**Bayesian workflow, priors and contraction**

- [BayesFlow] BayesFlow documentation, v2.0.12: `bayesflow.diagnostics.posterior_contraction`, its source listing, and the user guide, "Diagnostics and Visualizations". https://bayesflow.org/v2.0.12/api/bayesflow.diagnostics.posterior_contraction.html ; https://bayesflow.org/v2.0.12/_modules/bayesflow/diagnostics/metrics/posterior_contraction.html ; https://bayesflow.org/v2.0.12/user_guide/diagnostics.html
- [Bet] Michael Betancourt. Calibrating Model-Based Inferences and Decisions. arXiv:1803.08393v1, 22 Mar 2018. https://arxiv.org/abs/1803.08393
- [EOS+] Lasse Elsemüller, Hans Olischläger, Marvin Schmitt, Paul-Christian Bürkner, Ullrich Köthe, Stefan T. Radev. Sensitivity-Aware Amortized Bayesian Inference. arXiv:2310.11122v6 (TMLR, 08/2024, per the arXiv page). https://arxiv.org/abs/2310.11122 (abstract only)
- [GVS+] Andrew Gelman, Aki Vehtari, Daniel Simpson, Charles C. Margossian, Bob Carpenter, Yuling Yao, Lauren Kennedy, Jonah Gabry, Paul-Christian Bürkner, Martin Modrák. Bayesian Workflow. arXiv:2011.01808, 3 Nov 2020. https://arxiv.org/abs/2011.01808
- [KPBV] Noa Kallioinen, Topi Paananen, Paul-Christian Bürkner, Aki Vehtari. Detecting and diagnosing prior and likelihood sensitivity with power-scaling. arXiv:2107.14054v4, 26 May 2023. https://arxiv.org/abs/2107.14054
- [SBV] Daniel J. Schad, Michael Betancourt, Shravan Vasishth. Toward a principled Bayesian workflow in cognitive science. arXiv:1904.12765v3, 28 Feb 2020. https://arxiv.org/abs/1904.12765

**Samplers and diagnostics**

- [dynesty] dynesty documentation, version 3.1.0: index, "Getting Started", "FAQ". https://dynesty.readthedocs.io/en/stable/
- [emcee-docs] emcee documentation (stable): "Autocorrelation analysis & convergence", "FAQ", "Moves". https://emcee.readthedocs.io/en/stable/tutorials/autocorr/ ; https://emcee.readthedocs.io/en/stable/user/faq/ ; https://emcee.readthedocs.io/en/stable/user/moves/
- [EMR] Víctor Elvira, Luca Martino, Christian P. Robert. Rethinking the Effective Sample Size. arXiv:1809.04129 (International Statistical Review, 2022, per the arXiv page). https://arxiv.org/abs/1809.04129 (abstract only)
- [FHLG] Daniel Foreman-Mackey, David W. Hogg, Dustin Lang, Jonathan Goodman. emcee: The MCMC Hammer. arXiv:1202.3665v4. https://arxiv.org/abs/1202.3665
- [HG] Matthew D. Hoffman, Andrew Gelman. The No-U-Turn Sampler: Adaptively Setting Path Lengths in Hamiltonian Monte Carlo. Journal of Machine Learning Research 15(47), 1593–1623, 2014. https://jmlr.org/papers/v15/hoffman14a.html (abstract only)
- [loo] loo R package reference, version 2.10.1, "Diagnostics for Pareto smoothed importance sampling (PSIS)". https://mc-stan.org/loo/reference/pareto-k-diagnostic.html
- [PSIS] Aki Vehtari, Daniel Simpson, Andrew Gelman, Yuling Yao, Jonah Gabry. Pareto Smoothed Importance Sampling. Journal of Machine Learning Research 25(72), 1–58, 2024; arXiv:1507.02646v9. https://arxiv.org/abs/1507.02646 ; https://jmlr.org/papers/v25/19-556.html
- [Stan-diag] Stan, "How to Diagnose and Resolve Convergence Problems". https://mc-stan.org/learn-stan/diagnostics-warnings.html
- [Stan-RM] Stan Reference Manual, version 2.40: "Posterior Analysis", "Constraint Transforms", "MCMC Sampling". https://mc-stan.org/docs/reference-manual/analysis.html ; https://mc-stan.org/docs/reference-manual/transforms.html ; https://mc-stan.org/docs/reference-manual/mcmc.html
- [VGSCB] Aki Vehtari, Andrew Gelman, Daniel Simpson, Bob Carpenter, Paul-Christian Bürkner. Rank-normalization, folding, and localization: An improved R-hat for assessing convergence of MCMC. arXiv:1903.08008v5; Bayesian Analysis 16(2), 667–718, 2021, doi:10.1214/20-BA1221. https://arxiv.org/abs/1903.08008
- [VK] Dootika Vats, Christina Knudson. Revisiting the Gelman-Rubin Diagnostic. arXiv:1812.09384. https://arxiv.org/abs/1812.09384 (abstract only)

**Calibrators and simulation-based inference**

- [DHRWL] Arnaud Delaunoy, Joeri Hermans, François Rozet, Antoine Wehenkel, Gilles Louppe. Towards Reliable Simulation-Based Inference with Balanced Neural Ratio Estimation. arXiv:2208.13624v1 (NeurIPS 2022, per the NeurIPS proceedings page, https://proceedings.neurips.cc/paper_files/paper/2022/hash/7e6288bfb68182db7d6e328b0aefa89a-Abstract-Conference.html; the arXiv page shows no venue). https://arxiv.org/abs/2208.13624
- [DMP] Conor Durkan, Iain Murray, George Papamakarios. On Contrastive Learning for Likelihood-free Inference. arXiv:2002.03712v2 (ICML 2020, per the arXiv page). https://arxiv.org/abs/2002.03712 (abstract only)
- [HBL] Joeri Hermans, Volodimir Begy, Gilles Louppe. Likelihood-free MCMC with Amortized Approximate Ratio Estimators. arXiv:1903.04057v5 (ICML 2020, per the arXiv page). https://arxiv.org/abs/1903.04057 (abstract only)
- [HDR+] Joeri Hermans, Arnaud Delaunoy, François Rozet, Antoine Wehenkel, Volodimir Begy, Gilles Louppe. A Trust Crisis In Simulation-Based Inference? Your Posterior Approximations Can Be Unfaithful. arXiv:2110.06581v3 (TMLR version, per the arXiv page). https://arxiv.org/abs/2110.06581
- [LBGGM] Jan-Matthis Lueckmann, Jan Boelts, David S. Greenberg, Pedro J. Gonçalves, Jakob H. Macke. Benchmarking Simulation-Based Inference. arXiv:2101.04653v2 (AISTATS 2021, per the arXiv page). https://arxiv.org/abs/2101.04653
- [LPB] Balaji Lakshminarayanan, Alexander Pritzel, Charles Blundell. Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles. arXiv:1612.01474v3 (NIPS 2017). https://arxiv.org/abs/1612.01474
- [MWF] Benjamin Kurt Miller, Christoph Weniger, Patrick Forré. Contrastive Neural Ratio Estimation for Simulation-based Inference. arXiv:2210.06170v3 (NeurIPS 2022, per the arXiv page). https://arxiv.org/abs/2210.06170
- [sbi] sbi documentation (stable; latest release v0.27.0): API reference for NRE_A, NRE_B, NRE_C, BNRE, MCMCPosterior, classifier_nn and run_sbc; the installation page; the advanced tutorial "Simulation-based Calibration in SBI". https://sbi.readthedocs.io/en/stable/
- [SBKR] Marvin Schmitt, Paul-Christian Bürkner, Ullrich Köthe, Stefan T. Radev. Detecting Model Misspecification in Amortized Bayesian Inference with Neural Networks. arXiv:2112.08866. https://arxiv.org/abs/2112.08866 (abstract only)
- [TBSVG] Sean Talts, Michael Betancourt, Daniel Simpson, Aki Vehtari, Andrew Gelman. Validating Bayesian Inference Algorithms with Simulation-Based Calibration. arXiv:1804.06788v2. https://arxiv.org/abs/1804.06788

**Resampling, seeds and registration**

- [ASC+] Rishabh Agarwal, Max Schwarzer, Pablo Samuel Castro, Aaron Courville, Marc G. Bellemare. Deep Reinforcement Learning at the Edge of the Statistical Precipice. arXiv:2108.13264. https://arxiv.org/abs/2108.13264 (abstract only)
- [CGM] A. Colin Cameron, Jonah B. Gelbach, Douglas L. Miller. Bootstrap-Based Improvements for Inference with Clustered Errors. NBER Technical Working Paper 0344, September 2007. doi:10.3386/t0344. https://www.nber.org/papers/t0344 (abstract only)
- [CSO] Cédric Colas, Olivier Sigaud, Pierre-Yves Oudeyer. How Many Random Seeds? Statistical Power Analysis in Deep Reinforcement Learning Experiments. arXiv:1806.08295. https://arxiv.org/abs/1806.08295 (abstract only)
- [Hes] Tim Hesterberg. What Teachers Should Know about the Bootstrap: Resampling in the Undergraduate Statistics Curriculum. arXiv:1411.5279v1, 19 Nov 2014. https://arxiv.org/abs/1411.5279
- [OSF] OSF Support, "Welcome to Registrations & Preregistrations!". https://help.osf.io/article/229-select-a-registration-template
- [Owe] Art B. Owen. The pigeonhole bootstrap. Annals of Applied Statistics 1(2), 386–411, 2007. doi:10.1214/07-AOAS122; arXiv:0712.1111. https://arxiv.org/abs/0712.1111 (abstract only)
- [SBS] Varun Saravanan, Gordon J. Berman, Samuel J. Sober. Application of the hierarchical bootstrap to multi-level data in neuroscience. arXiv:2007.07797. https://arxiv.org/abs/2007.07797 (abstract only)
