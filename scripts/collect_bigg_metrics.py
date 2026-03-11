#!/usr/bin/env python3
"""
collect_bigg_metrics.py
=======================
Aggregate model_config_summary.json files from all BiGG model pipeline runs
into two output CSVs:

1. results/BiGG_models_scalability_metrics.csv  — full detail (all JSON fields)
2. results/BiGG_models_summary.csv              — legacy format expected by
   compute_scalability_analysis.py (Unique Reactions, Unique Genes, etc.)

Usage:
    python scripts/collect_bigg_metrics.py [--results-dir DIR] [--latest-only]
"""
import argparse
import json
import os
from pathlib import Path
import pandas as pd

PROJ_ROOT = Path(__file__).parent.parent
BIGG_TUNING_DIR = PROJ_ROOT / "results" / "tuning_results" / "BiGG_models"
RESULTS_DIR = PROJ_ROOT / "results"

# Organism lookup used when model_config_summary.json doesn't have one
# (older runs). Extend as needed.
ORGANISM_FALLBACK: dict[str, str] = {
    "RECON1": "Human",
    "Recon3D": "Human",
    "iMM1415": "Mouse",
    "iMM904": "S. cerevisiae",
    "iJN746": "P. putida",
    "iJN1463": "P. putida",
    "iML1515": "E. coli",
    "iJO1366": "E. coli",
    "iAF1260": "E. coli",
    "iAF1260b": "E. coli",
    "iJR904": "E. coli",
    "iNJ661": "M. tuberculosis",
    "iIT341": "H. pylori",
    "iSB619": "H. pylori",
    "iLB1027_lipid": "P. tricornutum",
    "iRC1080": "C. reinhardtii",
    "iHN637": "Geobacter",
    "iJB785": "S. elongatus",
    "iYS1720": "S. cerevisiae",
    "iYS854": "S. cerevisiae",
    "iND750": "S. cerevisiae",
    "iYL1228": "S. cerevisiae",
    "iYO844": "S. cerevisiae",
    "iAF692": "Methanosarcina",
    "iAF987": "C. acetobutylicum",
    "iAT_PLT_636": "Human platelet",
    "iPC815": "Plasmodium",
    "iLJ478": "T. brucei",
    "iNF517": "Leishmania",
    "e_coli_core": "E. coli",
    "iCN718": "M. bovis",
    "iCN900": "M. tuberculosis",
    "STM_v1_0": "Salmonella enterica",
    "iSynCJ816": "Synechocystis",
}


def find_run_dirs(base_dir: Path, latest_only: bool = True) -> dict[str, list[Path]]:
    """Return {model_name: [run_dir, ...]} for all completed runs."""
    runs_by_model: dict[str, list[Path]] = {}
    if not base_dir.exists():
        return runs_by_model
    for run_dir in sorted(base_dir.iterdir()):
        if not run_dir.is_dir():
            continue
        summary_path = run_dir / "model_config_summary.json"
        if not summary_path.exists():
            continue
        # Model name is everything before the last _<timestamp>_<runid> suffix
        # Directories are named  <ModelName>_<YYYYMMDD>_<runid>
        parts = run_dir.name.rsplit("_", 2)
        model_name = parts[0] if len(parts) >= 3 else run_dir.name
        runs_by_model.setdefault(model_name, []).append(run_dir)

    if latest_only:
        return {m: [sorted(dirs)[-1]] for m, dirs in runs_by_model.items()}
    return runs_by_model


def load_summary(run_dir: Path) -> dict:
    """Load model_config_summary.json, return {} on failure."""
    try:
        with open(run_dir / "model_config_summary.json") as f:
            return json.load(f)
    except Exception as e:
        print(f"  WARNING: Could not read {run_dir}: {e}")
        return {}


def determine_status(d: dict) -> str:
    """Infer run status from summary fields.

    Returns one of:
      Complete            – SA finished, final biomass recorded
      Failed: post-SA     – SA ran but final biomass missing
      Failed: SA          – optimization ran but SA timing absent/zero
      Failed: optimization – data prep ran but optimization timing absent/zero
      Failed: data prep   – pipeline started but failed before optimization
      Not started         – no timing information at all
    """
    def _pos(key: str) -> bool:
        v = d.get(key)
        try:
            return v is not None and float(v) > 0
        except (TypeError, ValueError):
            return False

    if d.get("final_biomass") and _pos("final_biomass"):
        return "Complete"
    if _pos("time_sa_s"):
        # SA ran but no final biomass → something went wrong after SA
        return "Failed: post-SA"
    if _pos("time_optimization_s"):
        return "Failed: SA"
    if _pos("time_data_prep_s"):
        return "Failed: optimization"
    if _pos("time_total_pipeline_s"):
        return "Failed: data prep"
    return "Not started"


def collect(latest_only: bool = True) -> pd.DataFrame:
    runs = find_run_dirs(BIGG_TUNING_DIR, latest_only=latest_only)
    if not runs:
        print(f"No run directories found under {BIGG_TUNING_DIR}")
        return pd.DataFrame()

    rows = []
    for model_name, dirs in sorted(runs.items()):
        for run_dir in dirs:
            d = load_summary(run_dir)
            if not d:
                continue
            organism = d.get("organism") or ORGANISM_FALLBACK.get(model_name, "Unknown")
            status = determine_status(d)
            rows.append({
                # ---- identification ----
                "Model":                     d.get("model_name", model_name),
                "Organism":                  organism,
                "Run ID":                    d.get("run_id", run_dir.name),
                "Status":                    status,
                # ---- model structure ----
                "Unique Reactions":          d.get("n_reactions"),
                "Unique Genes":              d.get("n_genes"),
                "Unique Metabolites":        d.get("n_metabolites"),
                # ---- CPI-Pred coverage ----
                "Unique Proteins (CPI-Pred)":    d.get("n_unique_proteins"),
                "Unique Substrates (CPI-Pred)":  d.get("n_unique_substrates"),
                "Rxn-Gene Pairs (CPI-Pred)":     d.get("n_rxn_gene_pairs"),
                "CPI-Pred Coverage (%)":         d.get("cpipred_coverage_pct"),
                # ---- biomass ----
                "COBRA Biomass":             d.get("cobra_biomass"),
                "Initial EC Biomass":        d.get("initial_ec_biomass"),
                "Initial Biomass":           d.get("initial_biomass"),
                "Final Biomass":             d.get("final_biomass"),
                "Biomass Improvement (%)":   d.get("improvement_percent"),
                # ---- kcat stats ----
                "Median Initial kcat (1/hr)":  d.get("median_kcat_initial_hr"),
                "Median Tuned kcat (1/hr)":    d.get("median_kcat_tuned_hr"),
                "Median Fold Change":           d.get("median_fold_change"),
                "N kcat Increased":            d.get("n_kcat_increased"),
                "N kcat Decreased":            d.get("n_kcat_decreased"),
                # ---- SA ----
                "SA Iterations":             d.get("iterations"),
                # ---- timing (seconds) ----
                "Time Data Prep (s)":        d.get("time_data_prep_s"),
                "Time Optimization (s)":     d.get("time_optimization_s"),
                "Time SA (s)":               d.get("time_sa_s"),
                "Time Total Pipeline (s)":   d.get("time_total_pipeline_s"),
                "Time Total Pipeline (hr)":  (
                    round(d["time_total_pipeline_s"] / 3600, 3)
                    if d.get("time_total_pipeline_s") else None
                ),
                # ---- SLURM ----
                "SLURM Job ID":              d.get("slurm_job_id"),
                "SLURM Node":                d.get("slurm_node"),
                "SLURM CPUs":                d.get("slurm_cpus"),
                "SLURM Mem Requested (GB)":  d.get("slurm_mem_requested_gb"),
                "SLURM Mem Used (MB)":       d.get("slurm_mem_used_mb"),
                "SLURM Walltime (s)":        d.get("slurm_walltime_s"),
                # ---- config ----
                "Enzyme Upper Bound":        d.get("enzyme_upper_bound"),
                "Biomass Reaction":          d.get("biomass_reaction"),
            })

    return pd.DataFrame(rows)


def save_outputs(df: pd.DataFrame) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # --- Full metrics CSV ---
    full_path = RESULTS_DIR / "BiGG_models_scalability_metrics.csv"
    df.to_csv(full_path, index=False)
    print(f"  Saved full metrics → {full_path.relative_to(PROJ_ROOT)}  ({len(df)} rows)")

    # --- Legacy BiGG_models_summary.csv (columns expected by compute_scalability_analysis.py) ---
    legacy_cols = [
        "Model", "Organism", "Status",
        "Unique Reactions", "Unique Genes",
        "Unique Proteins (CPI-Pred)", "Unique Substrates (CPI-Pred)",
        "Rxn-Gene Pairs (CPI-Pred)",
        "Median Initial kcat (1/hr)", "Median Tuned kcat (1/hr)",
        "Median Fold Change",
        "Initial Biomass", "Final Biomass",
        "Biomass Improvement (%)", "SA Iterations",
    ]
    # Also add N Runs column (always 1 when latest_only, but keep for compatibility)
    legacy = df[[c for c in legacy_cols if c in df.columns]].copy()
    legacy.insert(3, "N Runs", 1)
    legacy_path = RESULTS_DIR / "BiGG_models_summary.csv"
    legacy.to_csv(legacy_path, index=False)
    print(f"  Saved legacy summary → {legacy_path.relative_to(PROJ_ROOT)}  ({len(legacy)} rows)")


def print_summary(df: pd.DataFrame) -> None:
    n_total = len(df)
    n_complete = (df["Status"] == "Complete").sum()
    print(f"\n{'='*60}")
    print(f"BiGG Model Metrics Summary")
    print(f"{'='*60}")
    print(f"  Total models:     {n_total}")
    print(f"  Complete:         {n_complete}  ({100*n_complete/max(n_total,1):.1f}%)")

    # Breakdown by failure stage
    status_counts = df["Status"].value_counts()
    failed_statuses = [s for s in status_counts.index if s != "Complete"]
    if failed_statuses:
        print(f"  Failed/incomplete breakdown:")
        for s in sorted(failed_statuses):
            count = status_counts[s]
            models = df.loc[df["Status"] == s, "Model"].tolist()
            print(f"    {s:<28} {count:>3}  → {', '.join(models)}")
    else:
        print(f"  All runs complete — no failures.")
    n_incomplete = n_total - n_complete

    complete = df[df["Status"] == "Complete"]
    if not complete.empty:
        print(f"\n  --- Complete runs (timing) ---")
        if "Time Total Pipeline (hr)" in df.columns:
            t = complete["Time Total Pipeline (hr)"].dropna()
            if not t.empty:
                print(f"  Pipeline time:  mean={t.mean():.2f}h  min={t.min():.2f}h  max={t.max():.2f}h")
        if "Time SA (s)" in df.columns:
            s = complete["Time SA (s)"].dropna() / 3600
            if not s.empty:
                print(f"  SA time:        mean={s.mean():.2f}h  min={s.min():.2f}h  max={s.max():.2f}h")
        print(f"\n  --- Model sizes ---")
        for col, label in [("Unique Reactions", "Reactions"), ("Unique Genes", "Genes")]:
            v = complete[col].dropna()
            if not v.empty:
                print(f"  {label}: mean={v.mean():.0f}  min={v.min():.0f}  max={v.max():.0f}")
        if "Median Fold Change" in df.columns:
            fc = complete["Median Fold Change"].dropna()
            if not fc.empty:
                print(f"\n  Median fold change: mean={fc.mean():.3f}  min={fc.min():.3f}  max={fc.max():.3f}")
        if "SLURM Mem Used (MB)" in df.columns:
            mb = complete["SLURM Mem Used (MB)"].dropna()
            if not mb.empty:
                print(f"  SLURM peak mem:  mean={mb.mean():.0f} MB  max={mb.max():.0f} MB")
    print(f"{'='*60}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate BiGG model pipeline metrics")
    parser.add_argument("--results-dir", default=None,
                        help="Override BiGG tuning results directory")
    parser.add_argument("--latest-only", action="store_true", default=True,
                        help="Use only the most recent run per model (default: True)")
    parser.add_argument("--all-runs", action="store_true",
                        help="Include all runs per model (overrides --latest-only)")
    args = parser.parse_args()

    global BIGG_TUNING_DIR
    if args.results_dir:
        BIGG_TUNING_DIR = Path(args.results_dir)

    latest = not args.all_runs
    print(f"Scanning {BIGG_TUNING_DIR} ({'latest run only' if latest else 'all runs'}) ...")
    df = collect(latest_only=latest)

    if df.empty:
        print("No data collected. Check that pipeline runs exist under results/tuning_results/BiGG_models/")
        return

    save_outputs(df)
    print_summary(df)


if __name__ == "__main__":
    main()
