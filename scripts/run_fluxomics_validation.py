#!/usr/bin/env python3
"""
kinGEMs Fluxomics Validation Pipeline Script
============================================

This script provides a pipeline for validating a genome-scale metabolic model (GEM)
against experimental fluxomics data.

The experimental data comes from the study of Crown et al. (2015). It corresponds to
measured metabolic fluxes using 13C metabolic flux analysis (13C-MFA). The studied
organism was a W3110 strain of E. coli, grown on M9 media.

============================================

The script will:

1. Load the model and configuration to match the experimental data
2. Adjust the model to match the experimental conditions
3. Run FVA simulations
4. Match the simulated fluxes with the experimental fluxes
5. Calculate metrics of:
   - consistency score,
   - range precision ratio,
   - normalized Euclidean distance,
   - Jaccard index
6. Return the plots for:
   - FVA vs 13C-MFA flux range comparison
   - Jaccard index comparison

(Optionally) The script can also be used to provide existing FVA results
to skip steps 1-3.

============================================

Usage:
    python scripts/run_fluxomics_validation.py <config_file> <experimental_data_file>

    (example command)
    python scripts/run_fluxomics_validation.py \
        configs/fluxomics_iML1515_GEM.json \
        data/experimental/crown_fluxomics_final.csv

Usage providing existing FVA results:
    python scripts/run_fluxomics_validation.py <experimental_data_file> <fva_results_file(s)>

    (example command)
    python scripts/run_fluxomics_validation.py \
        data/experimental/crown_fluxomics_final.csv \
        results/tuning_results/iML1515_GEM_<run_name>/iML1515_GEM_fva_results.csv

Arguments:
    config_file: Path to JSON configuration file
    experimental_data_file: Path to the CSV file with experimental fluxomics data
    fva_results_file(s): Path to the CSV file(s) with FVA results
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import json
import logging
import os
import random
import sys
from typing import Dict

import cobra
from cobra.flux_analysis import flux_variability_analysis as cobra_fva
import pandas as pd

# Add parent directory to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from kinGEMs.dataset import convert_to_irreversible
from kinGEMs.modeling.fva import (
    flux_variability_analysis,
    flux_variability_analysis_parallel,
)
from kinGEMs.fluxomics_validation import (
    create_net_FVA_dataframe,
    create_fva_comparison_dataframe,
    calculate_consistency_score,
    calculate_range_precision_ratio,
    calculate_normalized_euclidean_dist,
    calculate_jaccard_index,
    plot_fva_mfa_comparison,
    plot_fva_mfa_comparison_normalized,
    plot_jaccard_index_comparison,
)


@dataclass
class FVAResultSpec:
    label: str
    path: str


def load_config(config_path: str) -> dict:
    """Load configuration from JSON file."""
    with open(config_path, "r", encoding="utf-8-sig") as f:
        return json.load(f)


def determine_biomass_reaction(model: cobra.Model) -> str:
    """Automatically determine the biomass reaction from model objective."""
    obj_rxns = {rxn.id: rxn.objective_coefficient for rxn in model.reactions
                if rxn.objective_coefficient != 0}
    if not obj_rxns:
        raise ValueError("No objective reaction found in model")
    return next(iter(obj_rxns.keys()))


def setup_logging(output_dir: str) -> logging.Logger:
    """Configure logging to file + stdout."""
    os.makedirs(output_dir, exist_ok=True)
    log_path = os.path.join(output_dir, "run.log")
    logger = logging.getLogger("fluxomics_validation")
    logger.setLevel(logging.INFO)
    logger.handlers = []

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    return logger


def sanitize_label(label: str) -> str:
    """Convert label to filesystem-safe string."""
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in label)


def apply_medium_constraints(model: cobra.Model, medium: dict | None, fix_fluxes: bool,
                             logger: logging.Logger) -> None:
    """Apply medium constraints to the COBRA model."""
    if not medium:
        return
    mode = "fixed fluxes" if fix_fluxes else "max uptake rates"
    logger.info("Applying medium conditions (%s): %d reactions", mode, len(medium))
    for rxn_id, flux_value in medium.items():
        try:
            rxn = model.reactions.get_by_id(rxn_id)
            rxn.lower_bound = flux_value
            if fix_fluxes:
                rxn.upper_bound = flux_value
            logger.info("  %s: LB=%s, UB=%s", rxn_id, rxn.lower_bound, rxn.upper_bound)
        except KeyError:
            logger.warning("  Reaction '%s' not found in model", rxn_id)


def run_cobra_fva(model: cobra.Model, opt_ratio: float, output_path: str,
                  logger: logging.Logger) -> pd.DataFrame:
    """Run COBRApy FVA and save results in kinGEMs-compatible format."""
    logger.info("Running COBRApy FVA (opt_ratio=%.3f)...", opt_ratio)
    cobra_fva_results = cobra_fva(model, fraction_of_optimum=opt_ratio)
    cobra_fva_df = pd.DataFrame({
        "Reactions": cobra_fva_results.index,
        "Min Solutions": cobra_fva_results["minimum"],
        "Max Solutions": cobra_fva_results["maximum"],
        "Solution Biomass": [model.slim_optimize()] * len(cobra_fva_results)
    })
    cobra_fva_df.to_csv(output_path, index=False)
    logger.info("Saved COBRApy FVA results: %s", output_path)
    return cobra_fva_df


def run_kinGEMs_fva(model: cobra.Model, processed_df: pd.DataFrame, biomass_reaction: str,
                    enzyme_upper_bound: float, fva_config: dict,
                    output_path: str, logger: logging.Logger) -> pd.DataFrame:
    """Run kinGEMs FVA (parallel or sequential)."""
    use_parallel = fva_config.get("parallel", False)
    n_workers = fva_config.get("workers", None)
    chunk_size = fva_config.get("chunk_size", None)
    method = fva_config.get("method", "dask")
    opt_ratio = fva_config.get("opt_ratio", 0.9)

    if use_parallel:
        logger.info("Running kinGEMs FVA in parallel (%s, workers=%s)...", method, n_workers or "auto")
        fva_results, _, _ = flux_variability_analysis_parallel(
            model=model,
            processed_df=processed_df,
            biomass_reaction=biomass_reaction,
            output_file=output_path,
            enzyme_upper_bound=enzyme_upper_bound,
            opt_ratio=opt_ratio,
            n_workers=n_workers,
            chunk_size=chunk_size,
            method=method
        )
    else:
        logger.info("Running kinGEMs FVA sequentially...")
        fva_results, _, _ = flux_variability_analysis(
            model=model,
            processed_df=processed_df,
            biomass_reaction=biomass_reaction,
            output_file=output_path,
            enzyme_upper_bound=enzyme_upper_bound,
            opt_ratio=opt_ratio
        )
    logger.info("kinGEMs FVA complete: %d reactions", len(fva_results))
    return fva_results


def build_fva_specs_from_config(config: dict, logger: logging.Logger) -> list[FVAResultSpec]:
    """Construct FVA result sources from config, if provided."""
    fva_specs = []
    fva_results = config.get("fluxomics_validation", {}).get("fva_results", {})
    for label, path in fva_results.items():
        if not path:
            continue
        fva_specs.append(FVAResultSpec(label=label, path=path))
    if fva_specs:
        logger.info("Using provided FVA results from config (%d sources).", len(fva_specs))
    return fva_specs


def save_metrics_summary(metrics: list[dict], output_path: str, logger: logging.Logger) -> None:
    metrics_df = pd.DataFrame(metrics)
    metrics_df.to_csv(output_path, index=False)
    logger.info("Saved metrics summary: %s", output_path)


def main() -> None:
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    config_path = sys.argv[1]
    use_config = config_path.lower().endswith(".json")
    config = load_config(config_path) if use_config else {}

    model_name = config.get("model_name", "model")
    enzyme_upper_bound = config.get("enzyme_upper_bound", 0.15)
    solver_name = config.get("solver", "glpk")
    medium = config.get("medium", None)
    medium_upper_bound = config.get("medium_upper_bound", True)
    fva_config = config.get("fva", {})

    fluxomics_cfg = config.get("fluxomics_validation", {})
    experimental_data_file = fluxomics_cfg.get("experimental_data_file")
    if not use_config:
        experimental_data_file = sys.argv[1]
    elif experimental_data_file is None:
        experimental_data_file = sys.argv[2] if len(sys.argv) > 2 else None
    if experimental_data_file is None:
        raise ValueError("experimental_data_file must be provided in config or as the second CLI argument")

    mfa_columns = fluxomics_cfg.get("mfa_columns", ["rxn_id", "exp_flux_lb", "exp_flux_ub"])
    top_jaccard_reactions = fluxomics_cfg.get("top_jaccard_reactions", 15)
    show_plots = fluxomics_cfg.get("show_plots", False)
    run_cobra = fluxomics_cfg.get("run_cobra_fva", True)
    processed_data_file = fluxomics_cfg.get("processed_data_file")
    convert_irreversible = fluxomics_cfg.get("convert_irreversible", True)

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    # Step 1: Load model (only if using config or running FVA)
    fva_specs = []
    if not use_config:
        fva_paths = sys.argv[2:]
        if not fva_paths:
            raise ValueError("At least one FVA results file must be provided in CSV-only mode.")
        for path in fva_paths:
            label = os.path.basename(path).replace(".csv", "")
            fva_specs.append(FVAResultSpec(label=label, path=path))
        if model_name == "model" and fva_specs:
            model_name = fva_specs[0].label

    run_id = f"{model_name}_{datetime.today().strftime('%Y%m%d')}_{random.randint(1000, 9999)}"
    results_dir = os.path.join(project_root, "results", "fluxomics_validation", run_id)
    logger = setup_logging(results_dir)

    logger.info("=== kinGEMs Fluxomics Validation ===")
    logger.info("Run ID: %s", run_id)
    if use_config:
        logger.info("Config file: %s", config_path)
    else:
        logger.info("Config file: (not used, CSV-only mode)")
    logger.info("Results directory: %s", results_dir)
    if not use_config:
        logger.info("CSV-only mode: using %d FVA result files.", len(fva_specs))

    if use_config:
        # Step 1: Load model
        model_path = os.path.join(project_root, "data", "raw", f"{model_name}.xml")
        logger.info("Loading model: %s", model_path)
        model = cobra.io.read_sbml_model(model_path)
        if solver_name:
            try:
                model.solver = solver_name
                logger.info("Solver set to: %s", solver_name)
            except Exception as exc:
                logger.warning("Failed to set solver '%s': %s", solver_name, exc)

        if convert_irreversible:
            logger.info("Converting model to irreversible reactions...")
            model = convert_to_irreversible(model)

        biomass_reaction = config.get("biomass_reaction") or determine_biomass_reaction(model)
        logger.info("Biomass reaction: %s", biomass_reaction)

        # Step 2: Apply experimental conditions (medium constraints)
        apply_medium_constraints(model, medium, medium_upper_bound, logger)

        # Step 3: Run or load FVA results
        fva_specs = build_fva_specs_from_config(config, logger)

    if not fva_specs:
        if run_cobra:
            cobra_path = os.path.join(results_dir, f"{model_name}_cobra_fva_results.csv")
            run_cobra_fva(model, fva_config.get("opt_ratio", 0.9), cobra_path, logger)
            fva_specs.append(FVAResultSpec(label="COBRA FVA", path=cobra_path))

        if processed_data_file:
            logger.info("Loading processed data: %s", processed_data_file)
            processed_df = pd.read_csv(processed_data_file)
            kin_path = os.path.join(results_dir, f"{model_name}_kinGEMs_fva_results.csv")
            run_kinGEMs_fva(model, processed_df, biomass_reaction,
                            enzyme_upper_bound, fva_config, kin_path, logger)
            fva_specs.append(FVAResultSpec(label="kinGEMs FVA", path=kin_path))

    if not fva_specs:
        raise ValueError("No FVA results available. Provide fluxomics_validation.fva_results or enable FVA runs.")

    # Step 4: Match simulated fluxes with experimental fluxes
    comparison_dfs: Dict[str, pd.DataFrame] = {}
    jaccard_info = []
    metrics_summary = []

    for spec in fva_specs:
        label_slug = sanitize_label(spec.label)
        net_fva_path = os.path.join(results_dir, f"net_fva_{label_slug}.csv")
        net_fva_df = create_net_FVA_dataframe(spec.path)
        net_fva_df.to_csv(net_fva_path, index=False)
        logger.info("Saved net FVA dataframe: %s", net_fva_path)

        comparison_df = create_fva_comparison_dataframe(
            net_fva_df,
            experimental_data_file,
            mfa_columns=mfa_columns
        )
        comparison_path = os.path.join(results_dir, f"fva_mfa_comparison_{label_slug}.csv")
        comparison_df.to_csv(comparison_path, index=False)
        logger.info("Saved FVA-MFA comparison: %s", comparison_path)
        comparison_dfs[spec.label] = comparison_df

        # Step 5: Calculate metrics
        consistency_score = calculate_consistency_score(comparison_df)
        range_precision_ratio = calculate_range_precision_ratio(comparison_df)
        normalized_euclidean_dist = calculate_normalized_euclidean_dist(comparison_df)
        jaccard_index, jaccard_df, zero_overlaps = calculate_jaccard_index(comparison_df)

        range_precision_path = os.path.join(results_dir, f"range_precision_ratio_{label_slug}.csv")
        range_precision_ratio.to_csv(range_precision_path, index=False, header=["range_precision_ratio"])
        logger.info("Saved range precision ratio: %s", range_precision_path)

        jaccard_path = os.path.join(results_dir, f"jaccard_{label_slug}.csv")
        jaccard_df.to_csv(jaccard_path, index=False)
        logger.info("Saved Jaccard details: %s", jaccard_path)

        metrics_summary.append({
            "model": spec.label,
            "consistency_score": consistency_score,
            "range_precision_ratio_median": range_precision_ratio.median(),
            "normalized_euclidean_dist": normalized_euclidean_dist,
            "jaccard_index": jaccard_index,
            "zero_overlaps": zero_overlaps
        })
        jaccard_info.append((spec.label, jaccard_index, zero_overlaps, jaccard_df))

    metrics_path = os.path.join(results_dir, "metrics_summary.csv")
    save_metrics_summary(metrics_summary, metrics_path, logger)

    # Step 6: Plots
    model_names = [info[0] for info in jaccard_info]
    jaccard_indices = [info[1] for info in jaccard_info]
    zero_overlaps = [info[2] for info in jaccard_info]
    jaccard_plot_path = os.path.join(results_dir, "jaccard_index_comparison.png")
    plot_jaccard_index_comparison(
        jaccard_indices=jaccard_indices,
        zero_overlaps=zero_overlaps,
        model_names=model_names,
        output_path=jaccard_plot_path,
        show=show_plots
    )
    logger.info("Saved Jaccard index comparison plot: %s", jaccard_plot_path)

    # Use first model's Jaccard ranking to select top reactions
    ref_label, _, _, ref_jaccard_df = jaccard_info[0]
    top_rxn_ids = ref_jaccard_df.sort_values("jaccard", ascending=False).head(top_jaccard_reactions)["rxn_id"].tolist()
    logger.info("Top %d reactions by Jaccard Index (%s): %s",
                top_jaccard_reactions, ref_label, ", ".join(top_rxn_ids))

    filtered_models = {
        name: df[df["rxn_id"].isin(top_rxn_ids)]
        for name, df in comparison_dfs.items()
    }

    fva_plot_path = os.path.join(results_dir, "fva_mfa_comparison.png")
    plot_fva_mfa_comparison(
        models_data=filtered_models,
        output_path=fva_plot_path,
        show=show_plots
    )
    logger.info("Saved FVA vs MFA comparison plot: %s", fva_plot_path)

    fva_plot_norm_path = os.path.join(results_dir, "fva_mfa_comparison_normalized.png")
    plot_fva_mfa_comparison_normalized(
        models_data=filtered_models,
        output_path=fva_plot_norm_path,
        show=show_plots
    )
    logger.info("Saved normalized FVA vs MFA comparison plot: %s", fva_plot_norm_path)

    logger.info("=== Fluxomics validation complete ===")


if __name__ == "__main__":
    main()
