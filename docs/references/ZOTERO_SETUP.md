# Zotero setup for RoughVolLab

Zotero is your reference manager: it stores the papers this project builds on,
their PDFs, and your notes on them — and later it generates bibliographies for
your own papers automatically.

## 1. Install and import (5 minutes)

1. Install Zotero from <https://www.zotero.org/download/> (free). Create a free
   account when prompted — it syncs your library across devices.
2. Open Zotero → **File → Import…** → choose
   `docs/references/roughvollab.bib` from this repo → import into a new
   collection. Rename the collection **RoughVolLab**.
3. Get the PDFs: right-click any entry → **Find Full Text** (works for the
   arXiv ones immediately). For paywalled journal papers, search the title on
   Google Scholar — most have free author versions — and drag the PDF onto the
   Zotero entry. Your University of Salford login also unlocks most journals
   via the library portal.
4. Optional but recommended: install the **Zotero Connector** browser
   extension — one click saves any paper you're reading into the library.

## 2. Reading order (matches the study guide)

**Tier 1 — the story (read these first, in order):**
1. `gjr2018` — *Volatility is rough*. The founding claim.
2. `contdas2024` — *Rough volatility: fact or artefact?* The counterargument.
   These two papers ARE the debate RoughVolLab interrogates.
3. `bfg2016` — *Pricing under rough volatility*. The rough Bergomi model the
   code simulates.

**Tier 2 — the machinery (read alongside Level 3 of the study guide):**
4. `mvn1968` — fractional Brownian motion (the maths of H).
5. `blp2017` — the hybrid simulation scheme (`volterra_weights` comes from here).
6. `giles2008` then `giles2015` — MLMC and its complexity theorem.
7. `mccrickerd2018` — turbocharging (the estimator that WON in P2).
8. `eer2019` — the rough Heston characteristic function (Layer 4's ground truth).

**Tier 3 — per-layer depth (as needed):**
- Identifiability/estimation: `ftw2019`, `ftw2022`, `contdas2023`, `bolko2023`, `bns2004`.
- MLMC variants: `gilesszpruch2014` (antithetic — refuted in P2), `bfrs2016`, `bourgey2021`.
- Lifted rough Heston: `bbf2023`, `abijaberelEuch2019`, `dff2002`, `andersen2008`.
- Hedging/execution: `buehler2019`, `almgrenchriss2001`, `rockafellaruryasev2000`.
- The book: `roughvolbook2024` — the field's reference volume, good to own.

Don't try to read everything. Tier 1 + the study guide is enough to explain
the project; go deeper only when a layer pulls you in.

## 3. How to take notes (keeps Zotero useful)

For each paper you actually read, add a Zotero note with four lines:
- **Claim:** what the paper asserts, in one sentence.
- **Method:** how it shows it.
- **RoughVolLab link:** which layer/file uses or tests this.
- **Doubt:** the one thing you'd challenge.

(Substantial notes belong in your Obsidian vault at
`C:\Users\Micha\Documents\roughvollab-notes` — the Paper Note template there
mirrors these four lines. Zotero holds the papers; Obsidian holds the thinking.)
