# kinGEMs Publication Timeline (Updated – Nov 2025)

Target journal: **Nature Biotechnology** or **PNAS**  
Target submission: **December 10, 2025**

Draft manuscript ready by: **December 1, 2025**

---

## Major Updates (since Aug 2025)
- **E. coli validation complete.** Converting outputs into **regression/correlation** metrics (Pearson/Spearman to reference fluxes, precision deltas, CI shrinkage).
- **Shift to precision-first framing:** Focus on **uncertainty reduction**, **cross-media stability**, and **variance contraction** rather than point accuracy.
- **Noor’s Biolog phenotype data:** Still **binary classification** (growth/no growth). Will check if **correlation-style** signals exist; otherwise keep classification-only.
- **Flux validation (Lya):** ~**26 fluxes** in analysis; treat as **provisional** until QC finalized.
- **Add enzyme-constraint ablation** (hard → soft/scaled → off) to attribute precision gains to enzyme constraints.
- **Shlomi et al. (2012) validation experiment:** preparing model-tuning for **higher-growth prediction ranges** to enable direct replication.
- Writing, figure design, and validation are running in parallel.

---

## Timeline & To-Do (Nov–Dec 2025)

### Nov 7–14 (This Week)
- [x] Finalize **E. coli validation** bullet summaries.
- [x] Implement **regression/correlation conversions** for E. coli.
- [x] Sync with **Noor** re: Biolog correlation signals; default to binary classification if none.
- [x] Coordinate with **Lya** on 26-flux schema, units, QC.
- [x] Design **enzyme ablation protocol** (hard/soft/off).


### Nov 15–24
**Writing:**
- [ ] Draft **Results (Part 1)** — precision improvement:
  - FVA width/nFVAW
  - CI/variance shrinkage
  - replicate stability
  - media consistency
- [ ] Update **Methods** for: precision metrics, bootstrap CIs, correlation computations, uncertainty reporting.

**Figures:**
- [ ] Add precision-centric plots (variance bars, Δprecision vs. baseline).
- [ ] Add E. coli correlation panels.
- [ ] Keep Biolog to classification-only (ROC/PR, confusion matrices).
- [ ] Integrate Shlomi-style growth validation figures if ready.

**Validation:**
- [ ] **Process Noor’s dataset** fully for manuscript-ready outputs.
- [ ] **Process Shlomi et al. validation data** (2012 PLoS Comp Bio): extract fluxes, growth phenotypes, and validation schema.
- [ ] **Tune the kinGEMs model for higher growth regimes** (adjust bounds/objective if needed) to reproduce the Shlomi validation setup.
- [ ] **Consolidate figure styles** (fonts, color palette, CI styles, axis formats) and migrate templates to the **Overleaf file**.
- [ ] Finalize E. coli correlation tables w/ caveats.
- [x] Run first-pass **enzyme ablation** experiments.

### Nov 25 – Dec 5
**Validation:**
- [ ] Integrate **Lya’s provisional 26-flux** results.
- [ ] Run **sensitivity analyses** (media, bounds, objectives) focused on precision.
- [ ] Analyze and summarize **enzyme ablation** deltas vs. GEM baseline.
- [ ] Complete **Shlomi et al. (2012) validation replication**.

**Writing:**
- [ ] Draft **Results (Part 2)** (flux validation, cross-species signals, ablation study).
- [ ] Draft **Shlomi validation subsection** describing growth-tuning and replication steps.
- [ ] Begin **Discussion** focusing on precision as the main performance indicator.

### Dec 6–8
- [ ] Internal review with co-authors (precision framing, statistics choices, figure clarity).
- [ ] Revise text and figure captions.
- [ ] Prepare **Supplementary** (pipelines, ablations, metrics, extended tables).

### Dec 8–10
- [ ] Final checks: references, formatting, figures, data/code repository.
- [ ] **Submit manuscript by Dec 10.**

---

## Notes on Data and Scope
- **Biolog (Noor):** still strictly **binary** unless a robust correlation-adjacent signal is validated.
- **Flux validation (Lya):** ~26 fluxes, **provisional**.
- **Shlomi et al. 2012:** incorporate as additional **growth prediction + flux-pattern validation**; model tuning underway.
- **E. coli:** complete; strength is **precision improvement** supported by correlation summaries.

---

## Enzyme Constraint Ablation Plan
**Goal:** Attribute precision gains directly to the presence and magnitude of enzyme constraints.

**Ablation levels:**
1. **Hard constraints:** standard kinGEMs setting.
2. **Soft constraints:** scale caps by {1.5×, 2×, 4×}.
3. **Off:** revert to standard GEM.

**Datasets/conditions:**
- Organisms: primarily **E. coli**; optional P. putida, S. elongatus.
- Media: ≥3 conditions (rich/minimal/alt carbon).
- Objectives: biomass v1/v2; blended if useful.
- Flux panel: core carbon + exchanges + Lya's ~26 fluxes.

**Primary precision metrics:**
- **nFVAW** (normalized FVA width).
- **Relative Variance Reduction (RVR)**.
- **Interval Score@95%.**
- **Media Stability Index (MSI)**.
- **KO Locality Index.**

**Statistical plan:**
- Paired comparisons, **bootstrap 95% CIs**, **Wilcoxon signed-rank**.
- Report median ΔnFVAW, ΔInterval Score, ΔMSI, ΔKO Locality.

**Figures:**
- nFVAW progression (hard → soft → off).
- Coverage–width frontier.
- MSI across media/objectives.
- Sankey: flux-coupling transitions.

---
