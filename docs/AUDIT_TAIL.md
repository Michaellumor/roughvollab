# RoughVolLab — Audit Tail (durable tracker)

**Provenance.** The original itemised error audit (2026-07-02/05, baseline `3f59638`)
survives locally only as its reconciliation header; the per-item body was lost.
This list was reconstructed on 2026-07-25 from dated session records (2026-07-11
and 2026-07-19), which contain the audit's own later triage of the remaining
items. Five cosmetic items whose detailed blocks could not be recovered are
retired below rather than re-audited. Recorded as ROADMAP decision **D48**.

Baseline of the "still live" determination: `f77f3ac` reconciliation (2026-07-08),
minus everything closed by the 14 issue→PR→merge loops and later work.

## Open — 10 items

| ID | Nature (recovered) | Status / next step |
|---|---|---|
| RVL-009 | Five gate-check specs absent from `docs/gate_checks/` (`p2_antithetic_build_and_verify.md`, `p2_conditional_gate_check.md`, `p2_conditional_build_and_verify.md`, `gh1_kappa1_fine_path_spec.md`, `gh4_kappa1_adoption_spec.md`) — corroborated by `docs/gate_checks/README.md` | Fully specified. **Recommended next loop** — reconstruct from the driver scripts, one PR |
| RVL-003 | Monotonicity trend check missing ("real-ish") | Open; fuller spec in 2026-07-19 session record; own small loop with test |
| RVL-004 | Dropped `weights` (real fix, deliberately left optional) | Open; fuller spec in 2026-07-19 session record; own loop with test |
| RVL-005 | Characteristic-function overflow guard (real fix, deliberately left optional) | Open; fuller spec in 2026-07-19 session record; own loop with test |
| RVL-006 | Bias-constant doc/code mismatch (verification gap) | Open; fuller spec in 2026-07-19 session record |
| RVL-031 | Collapse-zone untested (verification gap) | Open; fuller spec in 2026-07-19 session record |
| RVL-042 | fig1 eta parameter untraceable; deferred because the fix changes output | Open; requires its own verified loop, not a doc batch |
| RVL-012 | Documentation/hygiene | Open at nature level |
| RVL-038 | Deribit exception contract (hygiene) | Open at nature level |
| RVL-041 | Checksum mislabel (hygiene) | Open at nature level |

## Retired — 5 items (2026-07-25, decision D48)

| ID | Nature (recovered) | Reason |
|---|---|---|
| RVL-015 | Documentation/hygiene | Detailed finding lost; nature cosmetic; retired as unrecoverable rather than re-audited |
| RVL-032 | Documentation/hygiene | Same |
| RVL-034 | Documentation/hygiene | Same |
| RVL-043 | Context-free error message (hygiene) | One-line nature insufficient to locate without re-audit; cosmetic; retired |
| RVL-044 | Silent OHLC row drops (hygiene) | Detailed block lost; retired — if independently rediscovered in an OHLC loader, file fresh with a test |

Retirement is a tracked decision, not a deletion: IDs are never renumbered or
reused, and a retired item found to matter can be reopened by a dated note here.

## Lifecycle

Open items close gradually, one reviewed PR at a time, referencing this file
and their RVL ID. Any status change gets a dated note in place. This file — not
the ROADMAP — is the tail's system of record.
