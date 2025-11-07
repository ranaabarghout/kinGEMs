# kinGEMs Publication Timeline (Updated – Nov 2025)

Target journal: **Nature Biotechnology**  or **PNAS**
Target submission: **December 2025**

---

## Major Updates (since Aug 2025)
- **E. coli validation complete.** Currently converting outputs to **regression/correlation** style summaries (e.g., correlation vs. reference signals, precision deltas).
- **Emphasis shift:** prioritize **precision improvement** (e.g., flux variability/uncertainty reduction, consistency across media) over raw accuracy metrics.
- **Noor’s Biolog phenotype data:** remains **classification only** (growth vs. no growth). Exploring whether any **correlation-style** signals can be derived or aligned with our metrics (TBD).
- **Flux validation (Lya):** analyzing ~**26 fluxes**; results are **provisional** (not yet final).
  Add **enzyme constraint ablation** (hard → scaled/soft → off) to attribute precision gains specifically to enzyme constraints.
- Writing and validation continue in parallel.

---

## Timeline & To-Do (Nov–Dec 2025)

### Nov 7–14 (This Week)
- [x] Lock **E. coli validation** status and summary bullets.
- [ ] Implement **regression/correlation conversions** for E. coli (e.g., Pearson/Spearman vs. references; precision deltas vs. baseline).
- [ ] Sync with **Noor** on whether any correlation-adjacent signals can be surfaced from Biolog; if not, keep strictly **binary classification** in paper.
- [ ] Coordinate with **Lya** on the 26-flux set: data schema, units, comparators, and QC gates.
- [ ] **Design enzyme ablation protocol** (levels, metrics, datasets) and prepare run scripts.

### Nov 15–24
- Writing:
  - [ ] Draft **Results (Part 1)** focusing on **precision improvement**: FVA precision, variance/CI shrinkage, replicate stability, media consistency.
  - [ ] Update **Methods** with precise definitions: precision metrics, correlation computation, bootstrapping, and uncertainty reporting.
- Figures:
  - [ ] Add **precision-centric** plots (variance/CI bars, delta-precision vs. baseline), plus **correlation** panels for E. coli.
  - [ ] Keep Biolog to **ROC/PR** and **confusion matrices** (classification only).
- Validation:
  - [ ] Finalize E. coli regression/correlation tables; document caveats.
  - [ ] **Run enzyme ablation levels** (hard → scaled/soft → off) on E. coli across ≥3 media and record metrics.

### Nov 25 – Dec 5
- Validation:
  - [ ] Integrate **Lya’s provisional 26-flux** results; mark clearly as preliminary.
  - [ ] Sensitivity analyses (media, bounds, objective variants) focused on **precision changes** rather than point accuracy.
  - [ ] **Analyze ablation results**: compute paired deltas vs. GEM baseline, aggregate effect sizes and CIs.
- Writing:
  - [ ] Draft **Results (Part 2)** (flux validation, cross-species precision signals, **ablation study**).
  - [ ] Begin **Discussion** emphasizing why precision metrics are the right yardstick for kinGEMs.

### Dec 6–8
- [ ] Internal review with co-authors (precision framing, statistic choices, figure readability).
- [ ] Revise text, polish captions, ensure **precision-first narrative** is consistent.
- [ ] Prepare **Supplementary** (pipelines, tables, ablations, exact metric formulas).

### Dec 8-11
- [ ] Final checks: references, cover letter, journal formatting, data/code deposit.
- [ ] **Submit to Nature Biotechnology** (buffer: slipping into late Dec acceptable).

---

## Notes on Data and Scope
- **Biolog (Noor):** remains **binary growth**; correlation exploration is **ongoing/TBD**. If no robust correlation signal emerges, keep it strictly classification and avoid overclaiming.
- **Flux validation (Lya):** ~26 fluxes under active analysis; treat as **provisional** until QC passes and comparators are locked.
- **E. coli:** complete; headline is **precision improvement**, with supporting **regression/correlation** summaries now being added.

---

## Enzyme Constraint Ablation Plan (NEW)
**Goal:** Attribute precision gains to enzyme constraints and quantify their contribution.

**Ablation levels**
1. **Hard kinGEMs (full constraints):** standard enzyme-constrained model.
2. **Scaled/Soft constraints:** multiply enzyme caps by factors {1.5×, 2×, 4×}.
3. **Off (GEM):** remove enzyme constraints entirely.

**Datasets/conditions**
- Organisms: E. coli (primary), plus P. putida and S. elongatus if time permits.
- Media: ≥3 (rich, minimal, alternative carbon).
- Objectives: biomass v1, biomass v2 (and/or blended).
- Flux panel: sentinel set (central carbon + exchanges) and **Lya’s ~26 fluxes** when available.

**Primary metrics (precision-first)**
- **Normalized FVA Width (nFVAW)** and **Relative Variance Reduction (RVR)** vs. GEM.
- **Interval Score@95%** (if reference fluxes available).
- **Media Stability Index (MSI)** and **Objective Robustness** (mean |Δflux| normalized).
- **KO Locality Index** (concentration of |Δflux| in mechanistically relevant pathways).

**Statistical plan**
- Paired comparisons across matched fluxes/conditions; **bootstrap 95% CIs** and **Wilcoxon signed-rank** for medians.
- Report effect sizes: median ΔnFVAW, ΔInterval Score, ΔMSI, ΔKO Locality.

**Figures**
- Paired raincloud/violin of nFVAW (lines: hard → soft → off).
- Coverage–width frontier (Interval Score vs. width) by ablation level.
- MSI bar chart across media/objective variants.
- Sankey: flux-coupling class transitions (hard → soft → off).

**Reporting sentence (template)**
> Progressive relaxation of enzyme constraints increased median flux uncertainty by **X%** (nFVAW), worsened **Interval Score@95%** by **ΔS** (95% CI [a, b]), and reduced cross-media stability (ΔMSI = **c**), demonstrating that precision gains are driven by enzyme constraints rather than solver or modeling idiosyncrasies.

---

## Principles
- **Debug/validate daily** (AM).
- **Write daily** (PM).
- Keep figures and results **modular** so swaps don’t disrupt the manuscript flow.
- When in doubt, opt for **precision/uncertainty reporting** over single-number accuracy.
