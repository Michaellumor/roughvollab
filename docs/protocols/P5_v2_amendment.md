# P5 v2 — the OSF amendment: statement, order of acts and record

Michael Lumor

## Status

A working aid for the author (D77, 11 October 2026). It decides nothing and freezes nothing, and nothing on OSF has been touched. The text under "The amendment statement" is for the author to paste on OSF at freeze, with the placeholders in angle brackets filled from the record table; the acts under "The order of acts at freeze" are the author's where so marked. The protocol is `docs/protocols/P5_protocol_v2_draft.md`; v1 is `docs/protocols/P5_protocol.md`, registered as OSF APSF7 (https://doi.org/10.17605/OSF.IO/APSF7).

## What the registration is, and what OSF allows

Read on 11 October 2026 at the addresses given; paraphrases, nothing quoted.

- **The registration.** `apsf7` is an Open-Ended Registration of the OSF project `gdsv7`, registered on 2 October 2026 at 19:08 UTC, public, not embargoed, not withdrawn, its review state accepted and last modified at registration (API, `https://api.osf.io/v2/registrations/apsf7/`). It has one schema response, the original, approved, with an empty justification (API, `https://api.osf.io/v2/registrations/apsf7/schema_responses/`, total 1); the registration's own `revision_state` field read `approved` at a first reading on 11 October, `in_progress` at a second while a draft update the author had started existed, and `approved` again after the author deleted that draft the same day, nothing having been submitted; the schema response count stayed at one and nothing on the registration has changed since 2 October. The response holds a summary, which names the attached protocol as the authoritative document and cites commit `13bca7e`, and one uploaded file, `P5_protocol.md`, whose recorded SHA-256 is `5b7a67e5ca83b62308907b699f0587fe1febc4894a8335ce3b8acfe3e7890132` (API, `https://api.osf.io/v2/schema_responses/6ac001210231cc8223157561/`). The git blob of `docs/protocols/P5_protocol.md` at `13bca7e`, and at `main` on the day of reading, hashes to the same value; the working copy differs only by the CRLF line endings that `core.autocrlf` applies on checkout. The commit was made at 18:22 UTC on 2 October 2026 and the registration 45 minutes later: v1's order was the commit first and OSF second, and the ROADMAP's sentence that the registered file is byte-identical to the committed protocol holds, of the blob.
- **Updates.** From OSF Support, "Welcome to Registrations & Preregistrations!" (`https://help.osf.io/article/229-select-a-registration-template`, the sections on updating and the FAQ): an admin contributor starts an update from the registration's overview page, the project's registrations tab or the "My Registrations" page; the form is prefilled with the previous responses and at least one field must change; a justification is required; every admin contributor is notified and has 48 hours to approve or cancel, after which the update is approved automatically unless the registration sits in a community-run registry (`apsf7` does not); while it waits the registration shows that an update is in progress. Files cannot be added or removed in an update, and documents cannot be uploaded or modified after submission; a document can be linked from a public project, a newly submitted public registration or an outside repository. OSF describes updates as for events outside the researcher's control or unexpected anomalies. A withdrawn registration cannot be reinstated. The registration's own page is a browser application and showed nothing to a plain fetch, so the API was read instead.
- **Consequence.** "Amended with this version" (the draft's line 11) cannot be done by attaching v2 to `apsf7`. The frozen v2 text needs its own registration, so that a copy sits on an independent server with a timestamp and a hash as v1's does; and `apsf7` needs an update whose text points to it, so that a reader of v1's record finds v2. Neither alone does both: an update alone leaves v2 with no registered copy; a new registration alone leaves `apsf7` silent about being superseded.

## The design

Two acts on OSF, both the author's, after the freeze commit: (1) a new Open-Ended Registration of the frozen v2 file from the same project, with the amendment statement as its summary; (2) an update to `apsf7` whose summary gains one paragraph pointing to (1), the same paragraph entered as the justification. The file registered is the git blob of the freeze commit (LF line endings), exported with `git show <freeze-commit>:docs/protocols/P5_protocol_v2_draft.md > P5_protocol_v2_draft.md`, so that the hash OSF records can be checked against the blob as v1's was. The file keeps its name; the Status line, not the name, says it is frozen. These are proposals; the author may prefer a single new registration with `apsf7` left untouched, in which case step 4 below is dropped and the decision log says why.

## The amendment statement

Placeholders: `<freeze-commit>` (the full hash), `<sha256>`, `<freeze-date>`, `<v2-id>`, `<v2-doi>`.

**Title of the new registration (proposed).** v1's OSF title is "Does a learned calibrator know what it cannot know? Prior dependence in deep calibration of the rough Heston model"; the draft's own title is "P5 Protocol v2 — Is a learned calibrator more prior-driven than Bayes' rule requires?". Proposed, for continuity with v1's record: Does a learned calibrator know what it cannot know? Prior dependence in deep calibration of the rough Heston model (protocol v2; supersedes OSF APSF7). The author chooses.

**Summary of the new registration.**

Pre-registration, version 2, of study P5 of the RoughVolLab research programme (Michael Lumor; https://github.com/Michaellumor/roughvollab). It supersedes version 1, registered on 2 October 2026 as OSF APSF7 (https://doi.org/10.17605/OSF.IO/APSF7), which remains the record of what was registered then. The authoritative document is the attached file, `P5_protocol_v2_draft.md`, byte-identical to the file at commit `<freeze-commit>` of the repository (SHA-256 `<sha256>`), frozen on `<freeze-date>`. One item in it is left open by design and declared so: the number of test parameter points, set from the exact reference's measured cost at gate G2 and written in as a dated amendment before any network is trained.

What changed from version 1, with the decision-log entries (ROADMAP.md) that fixed each change:

- The question (D55). Version 1's four predictions followed from Bayes' rule and the parameter box, so an exact posterior would have confirmed them as readily as a network. Version 2 asks whether a learned calibrator is more prior-driven than the exact posterior under its own training prior, and measures each calibrator against that posterior.
- The model (D55, D64). The forward model is a frozen Chebyshev emulator of the rough Heston implied-volatility surface, shared by the training data and the exact reference, in place of the pricer called directly; its accuracy is gated (G1) against the noise level.
- The observable, the grid, the box and the resolution (D58, D59, D61, D62). Each quote is the model implied volatility at the parameter point's own standardised moneyness, closing a leak version 1 inherited from the repository's D38 set-up (D58); the grid is D38's five maturities by seven standardised-moneyness points; the box is H [0.02, 0.48], the priors' support in both versions and by rule not narrowed by the pilot, with ν [0.05, 0.35], ρ [−0.90, −0.70] and ξ₀ [0.04, 0.25] narrowed from version 1's ranges by a pilot of the pricer (gate G0) run before freezing under a criterion written down before the run (D59, D61, D62); the pricer's resolution is fixed by the same pilot.
- Noise (D65). Three levels, 0.1, 0.3 and 0.5 volatility points, as a design factor, in place of one level of 0.1.
- The priors (D69). The two Beta priors on H, with means 0.12 and 0.32 on the same support as in version 1, now have their parameters written in, with equal variance.
- The exact reference and gate G2 (D66). The exact posterior of each surface is computed by adaptive Markov-chain Monte Carlo with stated acceptance criteria and checked against a grid posterior; version 1 took the Fisher information and the Cramér–Rao bound as the ground truth, which version 2 reports only descriptively.
- The calibrators and gate G3 (D67, D68). A five-member deep ensemble and a neural ratio estimator, with architecture, optimiser, stopping rule, seeds and failure rules written in; five training seeds per calibrator, prior and noise level; G3 tests the recovery of ξ₀ and the stability of the error in H under a doubling of the training set, and no longer reads "trained on prior A" or "in every quintile".
- Stratification (D70). Fixed cut-offs on the exact posterior's contraction of H under prior A, at 0.9 and 0.5, with a floor on bin occupancy, in place of quintiles of the Cramér–Rao bound computed on the test set.
- The hypotheses (D71). Four hypotheses with new measures and thresholds, scored per calibrator and pooled over noise levels; H-c expects a leak of prior dependence into the identified regime, the reverse of version 1's expectation there; the null requires all four refuted and the interval-fidelity measure within a stated band.
- The analysis (D70, D71, D75). A crossed bootstrap over training seeds and test pairs in place of a bootstrap over test surfaces (D71); a re-scoring at the cut-offs shifted by ±0.05 that decides nothing (D70); a descriptive dose-response (D71); the analysis script committed before any result exists (D75). The whole-interval verdict rule is kept from version 1.
- Dropped from version 1 without a counterpart. The control at home (version 1's coverage on test H drawn from the training prior); the optional tenfold-cleaner noise level, superseded by the three levels; and version 1's stated reason for the 0.20 gap between the priors' means, the draft now giving the gap in standard deviations of the priors. Recorded in the decision sheet's "Differences from v1".
- Prior work (D76). The section re-reads version 1's papers at source and records where their text does not bear out version 1's sentences, and covers prior-sensitivity analysis and the literature on unfaithful amortised posteriors; the novelty claim is limited to the identifiability map and the measurement against an exact posterior.
- Code before freezing (D59 to D62, D72 to D75). Version 1 was registered before any P5 code existed. Version 2 is frozen with the pilot of the pricer run and its output committed, and with every script committed; no emulator, reference posterior or trained network exists at freezing.

Why: so that the test falls on the part that is uncertain, the calibrator's fidelity to the exact posterior, and so that the grid, the noise and the affordability of the design, which version 1 left undefined against the repository, are fixed before the data are generated (D55).

What had been seen when version 2 was frozen: a timing and finiteness probe of the pricer on 3 October 2026, run outside the author's machine and not in the repository (D55); the scoping run of gate G0 on 9 October 2026 (UTC) and its main run with its confirmation on 10 October 2026, both on the author's machine from committed scripts with their output committed (D60 to D62), which fixed the box, the grid and the resolution; and nothing else. No emulator, reference posterior, trained network or test surface existed, which is to be verified at the freeze commit by a search of the repository's tracked files and its output directory. Every number in the hypotheses was fixed before any of those objects existed.

Drafted with AI assistance (Claude), disclosed in the document; the commitments are the author's on freezing.

**Paragraph for the update to APSF7 (added to the summary; entered as the justification).**

Superseded on `<freeze-date>` by version 2 of the protocol, registered as OSF `<v2-id>` (`<v2-doi>`) and frozen at commit `<freeze-commit>` of https://github.com/Michaellumor/roughvollab (file `docs/protocols/P5_protocol_v2_draft.md`, SHA-256 `<sha256>`). Version 1 remains the record of what was registered on 2 October 2026 and is not withdrawn. The reason for the amendment and the list of changes are in version 2's registration summary and in the repository's decision log (D55, D77).

## The order of acts at freeze

Each step is the author's ("author") or a repository commit made on the author's word ("repository"). Nothing before step 3 touches OSF.

1. **author** Read the eighteen sources listed under Relation to prior work; say so. **repository** Tick the box "Prior-work section written from the papers themselves" (line 214): a one-line docs commit, with a short decision-log entry recording that the author confirmed the reading.
2. **repository** The freeze commit, on the author's word, changing only: the Status line (line 7), "DRAFT." becoming "FROZEN on <freeze-date>." with the sentence that the number of test θ (line 94) is set after G2 as a dated amendment; the legend (line 11), reworded to name that one OPEN item; and the box "OSF registration amended", gaining the words "the act that follows this commit". Nothing else moves and the file keeps its name. Immediately after the commit, the repository records the output of `git show <freeze-commit>:docs/protocols/P5_protocol_v2_draft.md | sha256sum` and exports that blob to a file for upload.
3. **author** On OSF, signed in himself: a new Open-Ended Registration from the project `gdsv7`, with the title above, the summary above with its placeholders filled, and the exported file attached; submit; approve within 48 hours or let it approve; note the registration id and DOI.
4. **author** On OSF: an update to `apsf7` from its overview page, the paragraph above added to the summary and entered as the justification; approve within 48 hours or let it approve.
5. **repository** The post-freeze commit, a dated amendment: line 7 gains the v2 DOI; the box "OSF registration amended" is ticked; the decision-log entry states exactly those two changes and that no other line moved, fills the record below, and states the hash OSF reports for the registered file (API, `/v2/registrations/<v2-id>/` and its schema response, read-only) against the blob's hash.

## Record

| Item | Value |
|---|---|
| Freeze commit | `<freeze-commit>` |
| SHA-256 of the blob | `<sha256>` |
| Freeze date (UTC) | `<freeze-date>` |
| v2 registration id and DOI | `<v2-id>`, `<v2-doi>` |
| v2 date registered (UTC) | |
| Hash OSF reports for the registered file | |
| Update to APSF7 approved (UTC) | |

## What is not Claude's to do

Signing in to OSF, creating the registration, submitting the update and approving either are the author's acts. What a sitting can do: produce the text and the exported blob, compute the hashes, and read the public API afterwards to check them.
