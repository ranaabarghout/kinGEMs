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

Usage to regenerate analysis on existing results folder:
    python scripts/run_fluxomics_validation.py --regenerate_results <results_folder> [experimental_data_file]

    (example command - uses default experimental data from config)
    python scripts/run_fluxomics_validation.py --regenerate_results \
        results/fluxomics_validation/iML1515_GEM_20260126_6698

    (example command - specify experimental data file)
    python scripts/run_fluxomics_validation.py --regenerate_results \
        results/fluxomics_validation/iML1515_GEM_20260126_6698 \
        data/raw/crown_fluxomics_final.csv

Arguments:
    config_file: Path to JSON configuration file
    experimental_data_file: Path to the CSV file with experimental fluxomics data
    fva_results_file(s): Path to the CSV file(s) with FVA results
    --regenerate_results: Flag to regenerate analysis on existing results folder
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

import numpy as np
import pandas as pd

# ============================================================================
# REPRODUCIBILITY: Set fixed random seeds for deterministic results
# ============================================================================
# These seeds ensure that random operations produce the same results
# across different systems and runs. For non-reproducible behavior,
# comment out these lines or set PYTHONHASHSEED=0 at runtime.
random.seed(42)
np.random.seed(42)
# ============================================================================

# Add parent directory to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from kinGEMs.fluxomics_validation import (
    create_net_FVA_dataframe,
    create_fva_comparison_dataframe,
    calculate_consistency_score,
    calculate_range_precision_ratio,
    calculate_normalized_euclidean_dist,
    calculate_jaccard_index,
)
from kinGEMs.plots import (
    plot_fva_mfa_comparison,
    plot_fva_mfa_comparison_zoom_with_jaccard_table,
    plot_fva_mfa_comparison_normalized,
    plot_jaccard_index_comparison,
    plot_jaccard_index_comparison_overlapping
)

# Import pipeline core function (only used when running with config)
from run_pipeline import run_pipeline_core, PipelineResults


@dataclass
class FVAResultSpec:
    """Specification for an FVA result file."""
    label: str
    path: str


def load_config(config_path: str) -> dict:
    """Load configuration from JSON file (handles UTF-8 BOM)."""
    with open(config_path, "r", encoding="utf-8-sig") as f:
        return json.load(f)


def setup_logging(output_dir: str, log_name: str = "run.log") -> logging.Logger:
    """Configure logging to file + stdout."""
    os.makedirs(output_dir, exist_ok=True)
    log_path = os.path.join(output_dir, log_name)
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


def save_metrics_summary(metrics: list[dict], output_path: str, logger: logging.Logger) -> None:
    """Save metrics summary to CSV."""
    metrics_df = pd.DataFrame(metrics)
    metrics_df.to_csv(output_path, index=False)
    logger.info("Saved metrics summary: %s", output_path)


def run_fluxomics_analysis(
    fva_specs: list[FVAResultSpec],
    experimental_data_file: str,
    analysis_dir: str,
    logger: logging.Logger,
    mfa_columns: list[str] = None,
    top_jaccard_reactions: int = 15,
    show_plots: bool = False
) -> None:
    """
    Run fluxomics validation analysis (steps 4-6).

    Parameters
    ----------
    fva_specs : list[FVAResultSpec]
        List of FVA result specifications (label + path)
    experimental_data_file : str
        Path to the experimental MFA data CSV
    analysis_dir : str
        Directory to save analysis outputs
    logger : logging.Logger
        Logger instance
    mfa_columns : list[str], optional
        Column names for MFA file [rxn_id, lb, ub]
    top_jaccard_reactions : int
        Number of top reactions to include in comparison plots
    show_plots : bool
        Whether to display plots interactively
    """
    mfa_columns = mfa_columns or ["rxn_id", "exp_flux_lb", "exp_flux_ub"]
    os.makedirs(analysis_dir, exist_ok=True)

    logger.info("=== Starting Fluxomics Validation Analysis ===")
    logger.info("Analysis directory: %s", analysis_dir)
    logger.info("Experimental data: %s", experimental_data_file)
    logger.info("FVA sources: %d", len(fva_specs))

    comparison_dfs: Dict[str, pd.DataFrame] = {}
    jaccard_info = []
    metrics_summary = []

    # Step 4: Match simulated fluxes with experimental fluxes
    n_total_list = []
    for spec in fva_specs:
        logger.info("Processing FVA source: %s", spec.label)
        label_slug = sanitize_label(spec.label)

        # Create net FVA (collapse forward/reverse reactions)
        net_fva_path = os.path.join(analysis_dir, f"net_fva_{label_slug}.csv")
        net_fva_df = create_net_FVA_dataframe(spec.path)
        net_fva_df.to_csv(net_fva_path, index=False)
        logger.info("  Saved net FVA dataframe: %s", net_fva_path)

        # Create comparison dataframe
        comparison_df = create_fva_comparison_dataframe(
            net_fva_df,
            experimental_data_file,
            mfa_columns=mfa_columns
        )
        comparison_path = os.path.join(analysis_dir, f"fva_mfa_comparison_{label_slug}.csv")
        comparison_df.to_csv(comparison_path, index=False)
        logger.info("  Saved FVA-MFA comparison: %s", comparison_path)
        comparison_dfs[spec.label] = comparison_df

        # Step 5: Calculate metrics
        consistency_score = calculate_consistency_score(comparison_df)
        range_precision_ratio = calculate_range_precision_ratio(comparison_df)
        normalized_euclidean_dist = calculate_normalized_euclidean_dist(comparison_df)
        jaccard_index, jaccard_df, zero_overlaps = calculate_jaccard_index(comparison_df)
        n_total_list.append(len(jaccard_df))

        # Save detailed results
        range_precision_path = os.path.join(analysis_dir, f"range_precision_ratio_{label_slug}.csv")
        range_precision_ratio.to_csv(range_precision_path, index=False, header=["range_precision_ratio"])
        logger.info("  Saved range precision ratio: %s", range_precision_path)

        jaccard_path = os.path.join(analysis_dir, f"jaccard_{label_slug}.csv")
        jaccard_df.to_csv(jaccard_path, index=False)
        logger.info("  Saved Jaccard details: %s", jaccard_path)

        metrics_summary.append({
            "model": spec.label,
            "consistency_score": consistency_score,
            "range_precision_ratio_median": range_precision_ratio.median(),
            "normalized_euclidean_dist": normalized_euclidean_dist,
            "jaccard_index": jaccard_index,
            "zero_overlaps": zero_overlaps
        })
        jaccard_info.append((spec.label, jaccard_index, zero_overlaps, jaccard_df))

    # Save metrics summary
    metrics_path = os.path.join(analysis_dir, "metrics_summary.csv")
    save_metrics_summary(metrics_summary, metrics_path, logger)

    # Step 6: Generate plots
    logger.info("=== Generating Plots ===")

    # Jaccard index comparison plot
    model_names = [info[0] for info in jaccard_info]
    jaccard_indices = [info[1] for info in jaccard_info]
    zero_overlaps_list = [info[2] for info in jaccard_info]
    jaccard_plot_path = os.path.join(analysis_dir, "jaccard_index_comparison.png")


    plot_jaccard_index_comparison(
        jaccard_indices=jaccard_indices,
        zero_overlaps=zero_overlaps_list,
        model_names=model_names,
        output_path=jaccard_plot_path,
        show=show_plots,
        n_total=n_total_list
    )
    logger.info("Saved Jaccard index comparison plot: %s", jaccard_plot_path)

    # Overlapping-only Jaccard plot
    jaccard_dfs = [info[3] for info in jaccard_info]
    jaccard_overlap_plot_path = os.path.join(analysis_dir, "jaccard_index_comparison_overlapping.png")
    plot_jaccard_index_comparison_overlapping(
        jaccard_dfs=jaccard_dfs,
        model_names=model_names,
        output_path=jaccard_overlap_plot_path,
        show=show_plots
    )
    logger.info("Saved Jaccard index (overlapping only) plot: %s", jaccard_overlap_plot_path)

    # Use first model's Jaccard ranking to select top reactions
    ref_label, _, _, ref_jaccard_df = jaccard_info[0]
    top_rxn_ids = ref_jaccard_df.sort_values("jaccard", ascending=False).head(top_jaccard_reactions)["rxn_id"].tolist()
    logger.info("Top %d reactions by Jaccard Index (%s): %s",
                top_jaccard_reactions, ref_label, ", ".join(top_rxn_ids))

    # Filter comparison dataframes to top reactions
    filtered_models = {
        name: df[df["rxn_id"].isin(top_rxn_ids)]
        for name, df in comparison_dfs.items()
    }

    # FVA vs MFA comparison plots
    fva_plot_path = os.path.join(analysis_dir, "fva_mfa_comparison.png")
    plot_fva_mfa_comparison(
        models_data=filtered_models,
        output_path=fva_plot_path,
        show=show_plots
    )
    logger.info("Saved FVA vs MFA comparison plot: %s", fva_plot_path)

    rxn_ids=["EX_ac_e", "PPC", "MALS"]
    zoom_plot_path = os.path.join(analysis_dir, f"fva_mfa_comparison_zoom_{rxn_ids[0]}_{rxn_ids[1]}_{rxn_ids[2]}.png")
    plot_fva_mfa_comparison_zoom_with_jaccard_table(
        models_data=comparison_dfs,  # IMPORTANT: use full comparison_dfs (has MFA columns)
        rxn_ids=rxn_ids,
        output_path=zoom_plot_path,
        show=show_plots
    )
    logger.info("Saved zoomed MFA vs FVA plot with intersection/union table: %s", zoom_plot_path)


    fva_plot_norm_path = os.path.join(analysis_dir, "fva_mfa_comparison_normalized.png")
    plot_fva_mfa_comparison_normalized(
        models_data=filtered_models,
        output_path=fva_plot_norm_path,
        show=show_plots
    )
    logger.info("Saved normalized FVA vs MFA comparison plot: %s", fva_plot_norm_path)

    logger.info("=== Fluxomics Validation Analysis Complete ===")


def main() -> None:
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    # Setup paths
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

    # =========================================================================
    # REGENERATE RESULTS MODE: Regenerate analysis from existing results folder
    # =========================================================================
    if sys.argv[1] == "--regenerate_results":
        if len(sys.argv) < 3:
            print("Error: --regenerate_results requires a results folder path")
            print("Usage: python scripts/run_fluxomics_validation.py --regenerate_results <results_folder> [experimental_data_file]")
            sys.exit(1)

        results_dir = os.path.abspath(sys.argv[2])
        if not os.path.isdir(results_dir):
            raise ValueError(f"Results folder not found: {results_dir}")

        # Check for pipeline_results subfolder (indicates full pipeline run)
        pipeline_dir = os.path.join(results_dir, "pipeline_results")
        analysis_dir = os.path.join(results_dir, "fluxomics_analysis")

        # Default experimental data file (can be overridden)
        experimental_data_file = os.path.join(project_root, "data", "raw", "crown_fluxomics_final.csv")
        if len(sys.argv) > 3:
            experimental_data_file = sys.argv[3]

        if not os.path.exists(experimental_data_file):
            raise ValueError(f"Experimental data file not found: {experimental_data_file}")

        # Find FVA result files
        fva_specs = []
        if os.path.isdir(pipeline_dir):
            # Look for FVA files in pipeline_results
            fva_patterns = [
                ("COBRA FVA", "_cobra_fva_results.csv"),
                ("kinGEMs FVA (pre-tuning)", "_pre_tuning_fva_results.csv"),
                ("kinGEMs FVA", "_fva_results.csv"),  # Must come after pre_tuning to avoid matching it
            ]

            for filename in os.listdir(pipeline_dir):
                if filename.endswith(".csv"):
                    filepath = os.path.join(pipeline_dir, filename)
                    for label, pattern in fva_patterns:
                        if pattern in filename:
                            # Avoid duplicate matches (e.g., pre_tuning should not match regular fva)
                            if pattern == "_fva_results.csv" and "_pre_tuning_" in filename:
                                continue
                            if pattern == "_fva_results.csv" and "_cobra_" in filename:
                                continue
                            fva_specs.append(FVAResultSpec(label=label, path=filepath))
                            break
        else:
            # Direct mode: look for FVA files in results_dir itself
            analysis_dir = results_dir
            for filename in os.listdir(results_dir):
                if filename.endswith("_fva_results.csv") or filename.endswith("fva.csv"):
                    filepath = os.path.join(results_dir, filename)
                    label = os.path.basename(filename).replace(".csv", "")
                    fva_specs.append(FVAResultSpec(label=label, path=filepath))

        if not fva_specs:
            raise ValueError(f"No FVA result files found in {results_dir}")

        # Clear and recreate analysis directory
        os.makedirs(analysis_dir, exist_ok=True)

        # Setup logging
        run_id = os.path.basename(results_dir)
        logger = setup_logging(results_dir, log_name="regenerate_results.log")

        logger.info("=== kinGEMs Fluxomics Validation (REGENERATE RESULTS MODE) ===")
        logger.info("Results directory: %s", results_dir)
        logger.info("Experimental data: %s", experimental_data_file)
        logger.info("Found %d FVA result files:", len(fva_specs))
        for spec in fva_specs:
            logger.info("  - %s: %s", spec.label, spec.path)

        # Default settings for regenerate_results
        mfa_columns = ["rxn_id", "exp_flux_lb", "exp_flux_ub"]
        top_jaccard_reactions = 15
        show_plots = False

        # Run fluxomics analysis
        run_fluxomics_analysis(
            fva_specs=fva_specs,
            experimental_data_file=experimental_data_file,
            analysis_dir=analysis_dir,
            logger=logger,
            mfa_columns=mfa_columns,
            top_jaccard_reactions=top_jaccard_reactions,
            show_plots=show_plots
        )

        logger.info("=== Fluxomics Validation Regenerate Results Complete ===")
        logger.info("Results saved to: %s", analysis_dir)
        return

    # =========================================================================
    # STANDARD MODE: Config file or CSV-only
    # =========================================================================
    config_path = sys.argv[1]
    use_config = config_path.lower().endswith(".json")
    config = load_config(config_path) if use_config else {}

    # Extract config values
    model_name = config.get("model_name", "model")
    fluxomics_cfg = config.get("fluxomics_validation", {})

    # Determine experimental data file
    experimental_data_file = fluxomics_cfg.get("experimental_data_file")
    if not use_config:
        experimental_data_file = sys.argv[1]
    elif experimental_data_file is None:
        experimental_data_file = sys.argv[2] if len(sys.argv) > 2 else None
    if experimental_data_file is None:
        raise ValueError("experimental_data_file must be provided in config or as the second CLI argument")

    # Validation settings
    mfa_columns = fluxomics_cfg.get("mfa_columns", ["rxn_id", "exp_flux_lb", "exp_flux_ub"])
    top_jaccard_reactions = fluxomics_cfg.get("top_jaccard_reactions", 15)
    show_plots = fluxomics_cfg.get("show_plots", False)

    # Handle CSV-only mode (no config, just FVA result files)
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

    # Generate run ID and create directories
    run_id = f"{model_name}_{datetime.today().strftime('%Y%m%d')}_{random.randint(1000, 9999)}"
    results_dir = os.path.join(project_root, "results", "fluxomics_validation", run_id)

    if use_config:
        # Create subfolders for pipeline and analysis
        pipeline_dir = os.path.join(results_dir, "pipeline_results")
        analysis_dir = os.path.join(results_dir, "fluxomics_analysis")
        os.makedirs(pipeline_dir, exist_ok=True)
        os.makedirs(analysis_dir, exist_ok=True)
    else:
        # CSV-only mode: all outputs go to analysis folder
        analysis_dir = results_dir
        os.makedirs(analysis_dir, exist_ok=True)

    logger = setup_logging(results_dir)

    logger.info("=== kinGEMs Fluxomics Validation ===")
    logger.info("Run ID: %s", run_id)
    logger.info("Results directory: %s", results_dir)

    if use_config:
        logger.info("Config file: %s", config_path)
        logger.info("Pipeline output: %s", pipeline_dir)
        logger.info("Analysis output: %s", analysis_dir)

        # Run the pipeline (steps 1-3)
        logger.info("=== Running Pipeline (Steps 1-3) ===")
        pipeline_results = run_pipeline_core(
            config=config,
            output_dir=pipeline_dir,
            run_id=run_id,
            force_regenerate=False,
            logger=logger
        )

        # Build FVA specs from pipeline results
        if pipeline_results.cobra_fva_path and os.path.exists(pipeline_results.cobra_fva_path):
            fva_specs.append(FVAResultSpec(label="COBRA FVA", path=pipeline_results.cobra_fva_path))

        if pipeline_results.kingems_fva_pre_tuning_path and os.path.exists(pipeline_results.kingems_fva_pre_tuning_path):
            fva_specs.append(FVAResultSpec(label="kinGEMs FVA (pre-tuning)", path=pipeline_results.kingems_fva_pre_tuning_path))

        if pipeline_results.kingems_fva_post_tuning_path and os.path.exists(pipeline_results.kingems_fva_post_tuning_path):
            fva_specs.append(FVAResultSpec(label="kinGEMs FVA", path=pipeline_results.kingems_fva_post_tuning_path))

        # Also check for any additional FVA results specified in config
        config_fva_results = fluxomics_cfg.get("fva_results", {})
        for label, path in config_fva_results.items():
            if path and os.path.exists(path):
                fva_specs.append(FVAResultSpec(label=label, path=path))

    else:
        logger.info("CSV-only mode: using %d FVA result files", len(fva_specs))

    if not fva_specs:
        raise ValueError("No FVA results available. Enable FVA in config or provide FVA result files.")

    # Run fluxomics analysis (steps 4-6)
    run_fluxomics_analysis(
        fva_specs=fva_specs,
        experimental_data_file=experimental_data_file,
        analysis_dir=analysis_dir,
        logger=logger,
        mfa_columns=mfa_columns,
        top_jaccard_reactions=top_jaccard_reactions,
        show_plots=show_plots
    )

    logger.info("=== Fluxomics Validation Complete ===")
    logger.info("Results saved to: %s", results_dir)


if __name__ == "__main__":
    main()
