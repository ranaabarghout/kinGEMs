#!/usr/bin/env python3
"""
kinGEMs Pipeline with Alternative Solutions Analysis
====================================================

This script extends the standard pipeline to generate multiple alternative solutions
and compare the solution space between:
1. Non-enzyme constrained (COBRApy) models
2. kinGEMs pre-tuning (with initial kcat values)
3. kinGEMs post-tuning (after simulated annealing)

Features:
- Generate multiple near-optimal solutions using Gurobi solution pools
- Compare flux distributions across solutions for key reactions
- Analyze solution space characteristics (flexibility, diversity, entropy)
- Visualize differences between constraint levels
- Generate comprehensive comparison plots

Usage:
    python scripts/run_pipeline_with_alternatives.py <config_file> [options]

    --num-solutions N    Number of alternative solutions to generate (default: 10)
    --pool-gap FLOAT     Gap from optimal for solution pool (default: 0.02)
    --reactions LIST     Comma-separated list of reactions to analyze in detail
    --force, -f          Force regeneration of all intermediate files

Example:
    python scripts/run_pipeline_with_alternatives.py configs/iML1515_GEM.json \\
        --num-solutions 20 --pool-gap 0.01 \\
        --reactions "BIOMASS_Ecoli_core_w_GAM,PGI,PFK,FBA"

Requirements:
    - Gurobi solver (GLPK does not support solution pools)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from typing import List, Dict, Tuple
import warnings

import cobra
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

# Add parent directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from kinGEMs.modeling.optimize import run_optimization_with_dataframe
from kinGEMs.modeling.tuning import simulated_annealing, sweep_maintenance_parameters
from scripts.run_pipeline import (
    load_config,
    is_modelseed_model,
    determine_biomass_reaction,
    find_predictions_file
)

# Import data preparation functions
from kinGEMs.dataset import (
    annotate_model_with_kcat_and_gpr,
    load_model,
    merge_substrate_sequences,
    prepare_model_data,
    process_kcat_predictions,
    convert_to_irreversible
)
from kinGEMs.dataset_modelseed import prepare_modelseed_model_data

warnings.filterwarnings('ignore')


class SolutionSpaceAnalyzer:
    """Analyze and compare solution spaces from different constraint levels."""

    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def analyze_flux_distributions(self,
                                  solutions: List[Tuple[float, pd.DataFrame]],
                                  label: str,
                                  reactions: List[str] = None) -> pd.DataFrame:
        """
        Analyze flux distributions across multiple solutions.

        Parameters
        ----------
        solutions : list of (obj_val, df_fba) tuples
            List of solutions to analyze
        label : str
            Label for this solution set (e.g., "COBRApy", "kinGEMs pre-tuning")
        reactions : list of str, optional
            Specific reactions to analyze. If None, analyze all.

        Returns
        -------
        pd.DataFrame
            Statistics for each reaction (mean, std, min, max, entropy)
        """
        print(f"\n  Analyzing flux distributions for {label}...")

        # Extract flux data from all solutions
        flux_data = {}
        for idx, (obj_val, df_fba) in enumerate(solutions):
            fluxes = df_fba[df_fba['Variable'] == 'flux'].set_index('Index')['Value']
            flux_data[f'Sol_{idx+1}'] = fluxes

        flux_df = pd.DataFrame(flux_data)

        # Filter to specific reactions if requested
        if reactions:
            available_rxns = [r for r in reactions if r in flux_df.index]
            if len(available_rxns) < len(reactions):
                missing = set(reactions) - set(available_rxns)
                print(f"    Warning: Reactions not found: {missing}")
            flux_df = flux_df.loc[available_rxns]

        # Calculate statistics
        stats_df = pd.DataFrame({
            'Reaction': flux_df.index,
            'Mean': flux_df.mean(axis=1),
            'Std': flux_df.std(axis=1),
            'Min': flux_df.min(axis=1),
            'Max': flux_df.max(axis=1),
            'Range': flux_df.max(axis=1) - flux_df.min(axis=1),
            'CV': flux_df.std(axis=1) / (flux_df.mean(axis=1).abs() + 1e-10),  # Coefficient of variation
        })

        # Calculate Shannon entropy for each reaction
        def calculate_entropy(row):
            # Discretize flux values into bins
            values = row.dropna().values
            if len(values) == 0 or np.all(values == 0):
                return 0.0
            # Normalize to probabilities
            abs_vals = np.abs(values)
            if abs_vals.sum() == 0:
                return 0.0
            probs = abs_vals / abs_vals.sum()
            # Calculate entropy
            return stats.entropy(probs + 1e-10)

        stats_df['Entropy'] = flux_df.apply(calculate_entropy, axis=1)

        # Classify reactions by variability
        stats_df['Variability'] = 'Fixed'
        stats_df.loc[stats_df['Std'] > 0.01, 'Variability'] = 'Low'
        stats_df.loc[stats_df['Std'] > 0.1, 'Variability'] = 'Medium'
        stats_df.loc[stats_df['Std'] > 1.0, 'Variability'] = 'High'

        # Save detailed statistics
        stats_path = os.path.join(self.output_dir, f"{label.replace(' ', '_')}_flux_statistics.csv")
        stats_df.to_csv(stats_path, index=False)
        print(f"    Saved statistics to: {stats_path}")

        # Print summary
        print(f"    Reactions analyzed: {len(stats_df)}")
        print(f"    Fixed reactions (std < 0.01): {(stats_df['Variability'] == 'Fixed').sum()}")
        print(f"    Variable reactions (std > 0.01): {(stats_df['Variability'] != 'Fixed').sum()}")
        print(f"    Highly variable (std > 1.0): {(stats_df['Variability'] == 'High').sum()}")
        print(f"    Mean entropy: {stats_df['Entropy'].mean():.4f}")

        return stats_df

    def compare_solution_spaces(self,
                               cobrapy_stats: pd.DataFrame,
                               kingems_pre_stats: pd.DataFrame,
                               kingems_post_stats: pd.DataFrame) -> pd.DataFrame:
        """
        Compare solution space characteristics across constraint levels.

        Returns
        -------
        pd.DataFrame
            Comparison metrics for each reaction
        """
        print("\n  Comparing solution spaces...")

        # Merge statistics
        comparison = pd.DataFrame({'Reaction': cobrapy_stats['Reaction']})

        for label, stats in [('COBRApy', cobrapy_stats),
                             ('Pre_Tuning', kingems_pre_stats),
                             ('Post_Tuning', kingems_post_stats)]:
            comparison = comparison.merge(
                stats[['Reaction', 'Mean', 'Std', 'Range', 'Entropy']],
                on='Reaction',
                how='outer',
                suffixes=('', f'_{label}')
            )
            # Rename columns to include label
            for col in ['Mean', 'Std', 'Range', 'Entropy']:
                if col in comparison.columns:
                    comparison.rename(columns={col: f'{col}_{label}'}, inplace=True)

        # Calculate flexibility reduction metrics
        comparison['Std_Reduction_Pre'] = (
            (comparison['Std_COBRApy'] - comparison['Std_Pre_Tuning']) /
            (comparison['Std_COBRApy'] + 1e-10)
        )
        comparison['Std_Reduction_Post'] = (
            (comparison['Std_COBRApy'] - comparison['Std_Post_Tuning']) /
            (comparison['Std_COBRApy'] + 1e-10)
        )
        comparison['Entropy_Reduction_Pre'] = (
            (comparison['Entropy_COBRApy'] - comparison['Entropy_Pre_Tuning']) /
            (comparison['Entropy_COBRApy'] + 1e-10)
        )
        comparison['Entropy_Reduction_Post'] = (
            (comparison['Entropy_COBRApy'] - comparison['Entropy_Post_Tuning']) /
            (comparison['Entropy_COBRApy'] + 1e-10)
        )

        # Save comparison
        comparison_path = os.path.join(self.output_dir, "solution_space_comparison.csv")
        comparison.to_csv(comparison_path, index=False)
        print(f"    Saved comparison to: {comparison_path}")

        # Print summary metrics
        print("\n  Solution Space Summary:")
        print(f"    Average std reduction (pre-tuning): {comparison['Std_Reduction_Pre'].mean():.2%}")
        print(f"    Average std reduction (post-tuning): {comparison['Std_Reduction_Post'].mean():.2%}")
        print(f"    Average entropy reduction (pre-tuning): {comparison['Entropy_Reduction_Pre'].mean():.2%}")
        print(f"    Average entropy reduction (post-tuning): {comparison['Entropy_Reduction_Post'].mean():.2%}")

        return comparison

    def plot_flux_distributions(self,
                               cobrapy_solutions: List[Tuple[float, pd.DataFrame]],
                               kingems_pre_solutions: List[Tuple[float, pd.DataFrame]],
                               kingems_post_solutions: List[Tuple[float, pd.DataFrame]],
                               reactions: List[str],
                               model_name: str):
        """
        Create violin plots comparing flux distributions for specific reactions.
        """
        print(f"\n  Generating flux distribution plots for {len(reactions)} reactions...")

        # Extract flux data for specified reactions
        data_for_plot = []

        for rxn_id in reactions:
            # COBRApy
            for idx, (_, df_fba) in enumerate(cobrapy_solutions):
                flux_data = df_fba[df_fba['Variable'] == 'flux']
                flux_row = flux_data[flux_data['Index'] == rxn_id]
                if not flux_row.empty:
                    data_for_plot.append({
                        'Reaction': rxn_id,
                        'Constraint': 'No Constraints',
                        'Flux': flux_row['Value'].values[0],
                        'Solution': idx + 1
                    })

            # kinGEMs pre-tuning
            for idx, (_, df_fba) in enumerate(kingems_pre_solutions):
                flux_data = df_fba[df_fba['Variable'] == 'flux']
                flux_row = flux_data[flux_data['Index'] == rxn_id]
                if not flux_row.empty:
                    data_for_plot.append({
                        'Reaction': rxn_id,
                        'Constraint': 'Pre-Tuning',
                        'Flux': flux_row['Value'].values[0],
                        'Solution': idx + 1
                    })

            # kinGEMs post-tuning
            for idx, (_, df_fba) in enumerate(kingems_post_solutions):
                flux_data = df_fba[df_fba['Variable'] == 'flux']
                flux_row = flux_data[flux_data['Index'] == rxn_id]
                if not flux_row.empty:
                    data_for_plot.append({
                        'Reaction': rxn_id,
                        'Constraint': 'Post-Tuning',
                        'Flux': flux_row['Value'].values[0],
                        'Solution': idx + 1
                    })

        plot_df = pd.DataFrame(data_for_plot)

        if plot_df.empty:
            print("    Warning: No flux data found for specified reactions")
            return

        # Create figure with subplots
        n_reactions = len(reactions)
        n_cols = min(3, n_reactions)
        n_rows = (n_reactions + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 5*n_rows))
        if n_reactions == 1:
            axes = [axes]
        else:
            axes = axes.flatten()

        for idx, rxn_id in enumerate(reactions):
            ax = axes[idx]
            rxn_data = plot_df[plot_df['Reaction'] == rxn_id]

            if rxn_data.empty:
                ax.text(0.5, 0.5, f'{rxn_id}\n(No data)',
                       ha='center', va='center', transform=ax.transAxes)
                ax.set_xticks([])
                ax.set_yticks([])
                continue

            # Create violin plot
            sns.violinplot(data=rxn_data, x='Constraint', y='Flux', ax=ax,
                          order=['No Constraints', 'Pre-Tuning', 'Post-Tuning'],
                          palette='Set2')

            # Overlay individual points
            sns.stripplot(data=rxn_data, x='Constraint', y='Flux', ax=ax,
                         order=['No Constraints', 'Pre-Tuning', 'Post-Tuning'],
                         color='black', alpha=0.3, size=3)

            ax.set_title(f'{rxn_id}', fontweight='bold', fontsize=10)
            ax.set_xlabel('')
            ax.set_ylabel('Flux (mmol/gDW/h)', fontsize=9)
            ax.tick_params(axis='x', rotation=45)
            ax.grid(True, alpha=0.3, axis='y')

        # Remove extra subplots
        for idx in range(n_reactions, len(axes)):
            fig.delaxes(axes[idx])

        plt.suptitle(f'{model_name}: Flux Distributions Across Solution Spaces',
                    fontsize=14, fontweight='bold', y=0.995)
        plt.tight_layout()

        plot_path = os.path.join(self.output_dir, 'flux_distributions_comparison.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"    Saved plot to: {plot_path}")

    def plot_solution_space_metrics(self,
                                    comparison_df: pd.DataFrame,
                                    model_name: str):
        """
        Create summary plots showing solution space characteristics.
        """
        print("\n  Generating solution space metric plots...")

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # 1. Standard deviation comparison
        ax = axes[0, 0]
        std_data = comparison_df[['Std_COBRApy', 'Std_Pre_Tuning', 'Std_Post_Tuning']].dropna()
        std_data_long = pd.DataFrame({
            'Std': pd.concat([std_data['Std_COBRApy'],
                            std_data['Std_Pre_Tuning'],
                            std_data['Std_Post_Tuning']]),
            'Constraint': ['No Constraints']*len(std_data) +
                         ['Pre-Tuning']*len(std_data) +
                         ['Post-Tuning']*len(std_data)
        })
        sns.boxplot(data=std_data_long, x='Constraint', y='Std', ax=ax,
                   order=['No Constraints', 'Pre-Tuning', 'Post-Tuning'])
        ax.set_yscale('log')
        ax.set_title('Flux Standard Deviation Distribution', fontweight='bold')
        ax.set_ylabel('Standard Deviation (log scale)')
        ax.grid(True, alpha=0.3, axis='y')

        # 2. Entropy comparison
        ax = axes[0, 1]
        entropy_data = comparison_df[['Entropy_COBRApy', 'Entropy_Pre_Tuning', 'Entropy_Post_Tuning']].dropna()
        entropy_data_long = pd.DataFrame({
            'Entropy': pd.concat([entropy_data['Entropy_COBRApy'],
                                entropy_data['Entropy_Pre_Tuning'],
                                entropy_data['Entropy_Post_Tuning']]),
            'Constraint': ['No Constraints']*len(entropy_data) +
                         ['Pre-Tuning']*len(entropy_data) +
                         ['Post-Tuning']*len(entropy_data)
        })
        sns.violinplot(data=entropy_data_long, x='Constraint', y='Entropy', ax=ax,
                      order=['No Constraints', 'Pre-Tuning', 'Post-Tuning'],
                      palette='muted')
        ax.set_title('Solution Space Entropy Distribution', fontweight='bold')
        ax.set_ylabel('Shannon Entropy')
        ax.grid(True, alpha=0.3, axis='y')

        # 3. Flexibility reduction scatter
        ax = axes[1, 0]
        valid_data = comparison_df[
            (comparison_df['Std_Reduction_Pre'].notna()) &
            (comparison_df['Std_Reduction_Post'].notna())
        ]
        ax.scatter(valid_data['Std_Reduction_Pre'],
                  valid_data['Std_Reduction_Post'],
                  alpha=0.5, s=30)
        ax.plot([0, 1], [0, 1], 'k--', alpha=0.3, label='Equal reduction')
        ax.set_xlabel('Std Reduction: Pre-Tuning')
        ax.set_ylabel('Std Reduction: Post-Tuning')
        ax.set_title('Flexibility Reduction Comparison', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 4. Top variable reactions
        ax = axes[1, 1]
        # Get top 15 most variable reactions in COBRApy
        top_variable = comparison_df.nlargest(15, 'Std_COBRApy')
        x_pos = np.arange(len(top_variable))
        width = 0.25

        ax.barh(x_pos - width, top_variable['Std_COBRApy'], width,
               label='No Constraints', alpha=0.8)
        ax.barh(x_pos, top_variable['Std_Pre_Tuning'], width,
               label='Pre-Tuning', alpha=0.8)
        ax.barh(x_pos + width, top_variable['Std_Post_Tuning'], width,
               label='Post-Tuning', alpha=0.8)

        ax.set_yticks(x_pos)
        ax.set_yticklabels(top_variable['Reaction'], fontsize=8)
        ax.set_xlabel('Standard Deviation')
        ax.set_title('Top 15 Most Variable Reactions', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='x')

        plt.suptitle(f'{model_name}: Solution Space Characteristics',
                    fontsize=16, fontweight='bold', y=0.995)
        plt.tight_layout()

        plot_path = os.path.join(self.output_dir, 'solution_space_metrics.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"    Saved plot to: {plot_path}")

    def plot_objective_distributions(self,
                                    cobrapy_solutions: List[Tuple[float, pd.DataFrame]],
                                    kingems_pre_solutions: List[Tuple[float, pd.DataFrame]],
                                    kingems_post_solutions: List[Tuple[float, pd.DataFrame]],
                                    model_name: str):
        """
        Plot distribution of objective values across solutions.
        """
        print("\n  Generating objective value distribution plot...")

        fig, ax = plt.subplots(figsize=(10, 6))

        # Extract objective values
        cobrapy_objs = [obj for obj, _ in cobrapy_solutions]
        pre_objs = [obj for obj, _ in kingems_pre_solutions]
        post_objs = [obj for obj, _ in kingems_post_solutions]

        data = pd.DataFrame({
            'Objective': cobrapy_objs + pre_objs + post_objs,
            'Constraint': ['No Constraints']*len(cobrapy_objs) +
                         ['Pre-Tuning']*len(pre_objs) +
                         ['Post-Tuning']*len(post_objs)
        })

        # Create violin plot with box plot overlay
        sns.violinplot(data=data, x='Constraint', y='Objective', ax=ax,
                      order=['No Constraints', 'Pre-Tuning', 'Post-Tuning'],
                      palette='Set2', inner='box')

        # Add individual points
        sns.stripplot(data=data, x='Constraint', y='Objective', ax=ax,
                     order=['No Constraints', 'Pre-Tuning', 'Post-Tuning'],
                     color='black', alpha=0.3, size=5)

        # Add statistics text
        for idx, (label, objs) in enumerate([('No Constraints', cobrapy_objs),
                                             ('Pre-Tuning', pre_objs),
                                             ('Post-Tuning', post_objs)]):
            mean_val = np.mean(objs)
            std_val = np.std(objs)
            ax.text(idx, ax.get_ylim()[1]*0.95,
                   f'μ={mean_val:.4f}\nσ={std_val:.4f}',
                   ha='center', va='top', fontsize=9,
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        ax.set_ylabel('Objective Value (Biomass, 1/h)', fontsize=12)
        ax.set_xlabel('')
        ax.set_title(f'{model_name}: Objective Value Distributions',
                    fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        plot_path = os.path.join(self.output_dir, 'objective_distributions.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"    Saved plot to: {plot_path}")


def run_multi_solution_optimization(model,
                                   processed_df: pd.DataFrame,
                                   biomass_reaction: str,
                                   enzyme_upper_bound: float,
                                   num_solutions: int,
                                   solution_pool_gap: float,
                                   solver_name: str,
                                   medium: dict = None,
                                   medium_upper_bound: bool = True,
                                   label: str = "optimization") -> List[Tuple[float, pd.DataFrame]]:
    """
    Run optimization and return multiple alternative solutions.

    Returns
    -------
    list of (obj_val, df_fba) tuples
        List of alternative solutions
    """
    print(f"\n  Running {label} with solution pool...")
    print(f"    Requesting {num_solutions} solutions (gap: {solution_pool_gap*100:.1f}%)")

    result = run_optimization_with_dataframe(
        model=model,
        processed_df=processed_df,
        objective_reaction=biomass_reaction,
        enzyme_upper_bound=enzyme_upper_bound,
        enzyme_ratio=True,
        maximization=True,
        multi_enzyme_off=False,
        isoenzymes_off=False,
        promiscuous_off=False,
        complexes_off=False,
        output_dir=None,
        save_results=False,
        verbose=True,  # Enable verbose to see solution pool diagnostics
        solver_name=solver_name,
        medium=medium,
        medium_upper_bound=medium_upper_bound,
        num_solutions=num_solutions,
        solution_pool_gap=solution_pool_gap
    )

    # Check if multiple solutions were returned
    if isinstance(result[0], list):
        solutions, _, _ = result
        print(f"    ✓ Found {len(solutions)} solutions")
        return solutions
    else:
        # Single solution returned (solver doesn't support pools)
        print(f"    ⚠️  Only single solution returned (solver limitation)")
        obj_val, df_fba, _, _ = result
        return [(obj_val, df_fba)]


def main():
    parser = argparse.ArgumentParser(
        description='Run kinGEMs pipeline with alternative solutions analysis',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument('config_file', help='Path to JSON configuration file')
    parser.add_argument('--num-solutions', type=int, default=10,
                       help='Number of alternative solutions to generate (default: 10)')
    parser.add_argument('--pool-gap', type=float, default=0.02,
                       help='Solution pool gap from optimal (default: 0.02 = 2%%)')
    parser.add_argument('--reactions', type=str, default=None,
                       help='Comma-separated list of reactions to analyze in detail')
    parser.add_argument('--force', '-f', action='store_true',
                       help='Force regeneration of all intermediate files')

    args = parser.parse_args()

    # Load configuration
    print(f"Loading configuration from: {args.config_file}")
    config = load_config(args.config_file)

    # Check for Gurobi
    solver_name = config.get('solver', 'gurobi')
    if solver_name.lower() != 'gurobi':
        print("\n⚠️  WARNING: Solution pools require Gurobi solver")
        print("   Attempting to use Gurobi regardless of config setting...")
        solver_name = 'gurobi'

    try:
        import gurobipy
        print("  ✓ Gurobi is available")
    except ImportError:
        print("\n✗ ERROR: Gurobi is not available")
        print("  Solution pools require Gurobi. Please install Gurobi or use standard pipeline.")
        sys.exit(1)

    # Extract configuration
    model_name = config['model_name']
    organism = config.get('organism', 'Unknown')
    enzyme_upper_bound = config.get('enzyme_upper_bound', 0.15)
    sa_config = config.get('simulated_annealing', {})
    medium = config.get('medium', None)
    medium_upper_bound = config.get('medium_upper_bound', True)

    # Generate run ID
    run_id = f"{model_name}_alternatives_{datetime.today().strftime('%Y%m%d_%H%M%S')}"

    # Setup paths
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    data_dir = os.path.join(project_root, "data")
    raw_data_dir = os.path.join(data_dir, "raw")
    interim_data_dir = os.path.join(data_dir, "interim", model_name)
    processed_data_dir = os.path.join(data_dir, "processed", model_name)
    cpipred_predictions_dir = config.get(
        'cpipred_predictions_dir',
        os.path.join(data_dir, "interim", "CPI-Pred predictions")
    )
    if not os.path.isabs(cpipred_predictions_dir):
        cpipred_predictions_dir = os.path.abspath(os.path.join(project_root, cpipred_predictions_dir))
    results_dir = os.path.join(project_root, "results", "alternative_solutions", run_id)
    os.makedirs(results_dir, exist_ok=True)

    # File paths
    model_path = os.path.join(raw_data_dir, f"{model_name}.xml")
    substrates_output = os.path.join(interim_data_dir, f"{model_name}_substrates.csv")
    sequences_output = os.path.join(interim_data_dir, f"{model_name}_sequences.csv")
    merged_data_output = os.path.join(interim_data_dir, f"{model_name}_merged_data.csv")
    processed_data_output = os.path.join(processed_data_dir, f"{model_name}_processed_data.csv")

    print("\n" + "="*80)
    print(f"=== kinGEMs Pipeline with Alternative Solutions ===")
    print("="*80)
    print(f"Run ID: {run_id}")
    print(f"Model: {model_name}")
    print(f"Organism: {organism}")
    print(f"Results directory: {results_dir}")
    print(f"Number of solutions: {args.num_solutions}")
    print(f"Solution pool gap: {args.pool_gap*100:.1f}%")
    if args.reactions:
        print(f"Focus reactions: {args.reactions}")
    print("="*80)

    # Parse reactions to analyze
    focus_reactions = None
    if args.reactions:
        focus_reactions = [r.strip() for r in args.reactions.split(',')]
        print(f"\nWill generate detailed analysis for {len(focus_reactions)} reactions")

    # Determine biomass reaction
    temp_model = cobra.io.read_sbml_model(model_path)
    biomass_reaction = config.get('biomass_reaction') or determine_biomass_reaction(temp_model)
    print(f"\nBiomass reaction: {biomass_reaction}")

    # === Step 1-3: Data preparation (reuse cached if available) ===
    print("\n=== Steps 1-3: Data Preparation ===")

    is_modelseed = is_modelseed_model(model_name)

    if not args.force and os.path.exists(processed_data_output):
        print("  ✓ Loading cached processed data")
        processed_data = pd.read_csv(processed_data_output)
        model = load_model(model_path)
        model = convert_to_irreversible(model)
    else:
        print("  Preparing model data...")

        # Step 1: Prepare model
        if is_modelseed:
            metadata_dir = config.get('metadata_dir', os.path.join(data_dir, "Biolog experiments"))
            model, substrate_df, sequences_df = prepare_modelseed_model_data(
                model_path=model_path,
                substrates_output=substrates_output,
                sequences_output=sequences_output,
                organism=organism,
                metadata_dir=metadata_dir
            )
        else:
            model, substrate_df, sequences_df = prepare_model_data(
                model_path=model_path,
                substrates_output=substrates_output,
                sequences_output=sequences_output,
                organism=organism,
                convert_irreversible=True
            )

        # Step 2: Merge data
        if not os.path.exists(merged_data_output):
            merged_data = merge_substrate_sequences(
                substrate_df=substrate_df,
                sequences_df=sequences_df,
                model=model,
                output_path=merged_data_output
            )
        else:
            merged_data = pd.read_csv(merged_data_output)

        # Step 3: Process kcat predictions
        predictions_csv_path = config.get('cpipred_predictions_file')
        if predictions_csv_path:
            if not os.path.isabs(predictions_csv_path):
                predictions_csv_path = os.path.abspath(os.path.join(project_root, predictions_csv_path))
        else:
            predictions_csv_path = find_predictions_file(model_name, cpipred_predictions_dir)

        processed_data = process_kcat_predictions(
            merged_df=merged_data,
            predictions_csv_path=predictions_csv_path,
            output_path=processed_data_output
        )

    # Ensure kcat column
    if 'kcat_mean' in processed_data.columns and 'kcat' not in processed_data.columns:
        processed_data['kcat'] = processed_data['kcat_mean']

    # Annotate model
    model = annotate_model_with_kcat_and_gpr(model=model, df=processed_data)
    print(f"  Model: {len(model.genes)} genes, {len(model.reactions)} reactions")

    # Apply medium constraints if specified
    if medium is not None:
        print(f"  Applying medium constraints ({len(medium)} reactions)")
        for rxn_id, flux_value in medium.items():
            try:
                rxn = model.reactions.get_by_id(rxn_id)
                rxn.lower_bound = flux_value
                if medium_upper_bound:
                    rxn.upper_bound = flux_value
            except KeyError:
                print(f"    Warning: Reaction '{rxn_id}' not found")

    # === Step 4a: COBRApy Multiple Solutions (No Constraints) ===
    print("\n=== Step 4a: Generating COBRApy Solutions (No Enzyme Constraints) ===")

    # Load original reversible model for COBRApy (not the irreversible one used by kinGEMs)
    print("  Loading original (reversible) model for COBRApy analysis...")
    cobrapy_model = cobra.io.read_sbml_model(model_path)

    # Apply same medium constraints to COBRApy model
    if medium is not None:
        for rxn_id, flux_value in medium.items():
            try:
                rxn = cobrapy_model.reactions.get_by_id(rxn_id)
                rxn.lower_bound = flux_value
                if medium_upper_bound:
                    rxn.upper_bound = flux_value
            except KeyError:
                pass  # Reaction might not exist in original model

    # For COBRApy, we can't use the enzyme-constrained function
    # Instead, we'll use Gurobi directly with the original cobra model
    try:
        from cobra.core import get_solution
        cobrapy_model.solver = 'gurobi'

        # First, verify the model is feasible without solution pool
        solution = cobrapy_model.optimize()

        if solution.status != 'optimal':
            raise ValueError(f"Model not optimal: {solution.status}")

        print(f"  Model is feasible. Objective: {solution.objective_value:.4f}")

        # Configure solution pool BEFORE optimization
        print(f"  Configuring solution pool: PoolSearchMode=2, PoolSolutions={args.num_solutions}, PoolGap={args.pool_gap}")
        gurobi_model = cobrapy_model.solver.problem
        gurobi_model.setParam('PoolSearchMode', 2)  # 2 = systematic search
        gurobi_model.setParam('PoolSolutions', args.num_solutions)
        gurobi_model.setParam('PoolGap', args.pool_gap)

        # Re-optimize to populate solution pool
        print(f"  Re-optimizing with solution pool settings...")
        cobrapy_model.solver.optimize()

        # Check how many solutions were found
        num_solutions_found = gurobi_model.SolCount
        print(f"  ✓ Gurobi SolCount: {num_solutions_found}")

        # Also check if the model is LP or MIP (solution pools work better for MIP)
        is_mip = gurobi_model.IsMIP
        print(f"  Model type: {'MIP' if is_mip else 'LP'}")
        if not is_mip:
            print(f"  ⚠️  NOTE: Model is LP (continuous). Solution pool may only find 1 solution.")
            print(f"  For LP models with degeneracy, consider using Flux Variability Analysis (FVA)")
            print(f"  or sampling methods to explore the solution space.")

        # Extract solutions from pool using Gurobi's Xn attribute
        cobrapy_solutions = []
        gurobi_model = cobrapy_model.solver.problem
        num_solutions_found = gurobi_model.SolCount
        print(f"  ✓ Found {num_solutions_found} COBRApy solutions in pool")

        if num_solutions_found == 0:
            print(f"  ⚠️  No solutions in pool - using ACHR flux sampling as alternative")
            print(f"  Running ACHR (Artificial Centering Hit-and-Run) sampling...")

            # Use ACHR to sample the solution space
            # First constrain to near-optimal region
            with cobrapy_model:
                # Add constraint to stay within pool gap of optimal
                min_objective = solution.objective_value * (1.0 - args.pool_gap)
                cobrapy_model.reactions.get_by_id(cobrapy_model.objective.direction).lower_bound = min_objective

                print(f"  Sampling within {args.pool_gap*100}% of optimal ({min_objective:.6f} ≤ objective ≤ {solution.objective_value:.6f})")

                # Generate samples using ACHR
                # thinning=100 means take every 100th sample to reduce correlation
                sampler = cobra.sampling.ACHRSampler(
                    cobrapy_model,
                    thinning=100,
                    seed=42
                )

                # Generate samples (returns DataFrame with reactions as columns)
                samples_df = sampler.sample(args.num_solutions)

                print(f"  ✓ Generated {len(samples_df)} ACHR samples")

                # Convert samples to solution format
                for sample_idx in samples_df.index:
                    fluxes = samples_df.loc[sample_idx].to_dict()

                    # Calculate objective value for this sample
                    obj_val = sum(
                        coef * fluxes[rxn.id]
                        for rxn, coef in cobrapy_model.objective.items()
                    )

                    records = [('flux', rxn_id, flux, None) for rxn_id, flux in fluxes.items()]
                    df_fba = pd.DataFrame(records, columns=['Variable', 'Index', 'Value', 'Bounds'])
                    cobrapy_solutions.append((obj_val, df_fba))
        else:
            for i in range(min(num_solutions_found, args.num_solutions)):
                gurobi_model.params.SolutionNumber = i
                obj_val = gurobi_model.PoolObjVal

                # Get flux values from solution pool using Xn attribute
                fluxes = {}
                for rxn in cobrapy_model.reactions:
                    # Get the Gurobi variable for this reaction
                    cobra_var = cobrapy_model.solver.variables[rxn.id]
                    # Xn gets the value from the solution pool for SolutionNumber i
                    fluxes[rxn.id] = cobra_var.Xn

                # Create dataframe matching kinGEMs format
                records = [('flux', rxn_id, flux, None) for rxn_id, flux in fluxes.items()]
                df_fba = pd.DataFrame(records, columns=['Variable', 'Index', 'Value', 'Bounds'])

                cobrapy_solutions.append((obj_val, df_fba))
                print(f"    Solution {i+1}: objective = {obj_val:.4f}")

    except Exception as e:
        print(f"  ⚠️  Error generating COBRApy solutions: {e}")
        print(f"  Solution status: {solution.status if 'solution' in locals() else 'unknown'}")
        print("  This may be due to model infeasibility or solver issues.")
        print("  Skipping COBRApy solution generation - will only analyze kinGEMs models.")
        # Create empty solution list - we'll skip comparison with COBRApy
        cobrapy_solutions = []

    # === Step 4b: kinGEMs Pre-Tuning Multiple Solutions ===
    print("\n=== Step 4b: Generating kinGEMs Pre-Tuning Solutions ===")

    kingems_pre_solutions = run_multi_solution_optimization(
        model=model,
        processed_df=processed_data,
        biomass_reaction=biomass_reaction,
        enzyme_upper_bound=enzyme_upper_bound,
        num_solutions=args.num_solutions,
        solution_pool_gap=args.pool_gap,
        solver_name=solver_name,
        medium=medium,
        medium_upper_bound=medium_upper_bound,
        label="kinGEMs pre-tuning"
    )

    # === Step 5: Simulated Annealing ===
    print("\n=== Step 5: Running Simulated Annealing ===")

    # Get gene sequences for annealing
    gene_sequences_dict = {}
    if 'Single_gene' in processed_data.columns and 'SEQ' in processed_data.columns:
        for _, row in processed_data.iterrows():
            if pd.notna(row['SEQ']) and pd.notna(row['Single_gene']):
                gene_sequences_dict[row['Single_gene']] = row['SEQ']

    kcat_dict, top_targets, df_new, iterations, biomasses, df_FBA = simulated_annealing(
        model=model,
        processed_data=processed_data,
        biomass_reaction=biomass_reaction,
        objective_value=sa_config.get('biomass_goal', 0.5),
        gene_sequences_dict=gene_sequences_dict,
        output_dir=results_dir,
        enzyme_fraction=enzyme_upper_bound,
        n_top_enzymes=sa_config.get('n_top_enzymes', 65),
        temperature=sa_config.get('temperature', 1.0),
        cooling_rate=sa_config.get('cooling_rate', 0.95),
        min_temperature=sa_config.get('min_temperature', 0.01),
        max_iterations=sa_config.get('max_iterations', 100),
        max_unchanged_iterations=sa_config.get('max_unchanged_iterations', 5),
        change_threshold=sa_config.get('change_threshold', 0.009),
        verbose=sa_config.get('verbose', False),
        medium=medium,
        medium_upper_bound=medium_upper_bound
    )

    improvement = (biomasses[-1] - biomasses[0]) / biomasses[0] * 100 if biomasses[0] > 0 else 0
    print(f"\n  Annealing complete! Improvement: {improvement:.1f}%")
    print(f"    Initial biomass: {biomasses[0]:.4f}")
    print(f"    Post-annealing biomass: {biomasses[-1]:.4f}")

    # === Step 6: Maintenance Parameter Sweep ===
    print("\n=== Step 6: Maintenance Parameters Sweep ===")

    maintenance_config = config.get('maintenance_sweep', {})
    ngam_rxn_id = config.get('ngam_rxn_id', 'ATPM')
    ngam_range = maintenance_config.get('ngam_range', None)
    gam_range = maintenance_config.get('gam_range', None)
    biomass_goal = sa_config.get('biomass_goal', 0.5)

    print(f"  Configuration:")
    print(f"    - NGAM range: {ngam_range if ngam_range else 'default'}")
    print(f"    - GAM range: {gam_range if gam_range else 'constant'}")
    print(f"    - NGAM reaction: {ngam_rxn_id}")
    print(f"    - Biomass goal: {biomass_goal}")

    maintenance_results = sweep_maintenance_parameters(
        model=model,
        processed_data=df_new,  # Use tuned kcat values
        biomass_reaction=biomass_reaction,
        ngam_rxn_id=ngam_rxn_id,
        ngam_range=ngam_range,
        gam_range=gam_range,
        enzyme_upper_bound=enzyme_upper_bound,
        output_dir=results_dir,
        medium=medium,
        medium_upper_bound=medium_upper_bound,
        biomass_goal=biomass_goal,
        verbose=maintenance_config.get('verbose', False)
    )

    print(f"  Maintenance sweep completed: {len(maintenance_results)} parameter combinations tested")

    # Get optimal parameters from sweep
    optimal_ngam = None
    optimal_gam = None
    optimal_biomass = None

    if len(maintenance_results) > 0 and maintenance_results['biomass'].max() > 0:
        if biomass_goal is not None:
            # Find the combination closest to the biomass goal
            maintenance_results['distance_to_goal'] = abs(maintenance_results['biomass'] - biomass_goal)
            best_idx = maintenance_results['distance_to_goal'].idxmin()
            selection_method = f"closest to goal ({biomass_goal:.4f})"
        else:
            # Find the combination with maximum biomass
            best_idx = maintenance_results['biomass'].idxmax()
            selection_method = "maximum biomass"

        optimal_ngam = float(maintenance_results.loc[best_idx, 'ngam'])
        optimal_gam = float(maintenance_results.loc[best_idx, 'gam'])
        optimal_biomass = float(maintenance_results.loc[best_idx, 'biomass'])

        print(f"\n  Optimal maintenance parameters found ({selection_method}):")
        print(f"    - NGAM: {optimal_ngam:.2f} mmol/gDW/h")
        print(f"    - GAM: {optimal_gam:.2f} mmol ATP/gDW")
        print(f"    - Biomass: {optimal_biomass:.4f}")
        if biomass_goal is not None:
            deviation = abs(optimal_biomass - biomass_goal)
            print(f"    - Deviation from goal: {deviation:.4f}")
    else:
        print("\n  ⚠️  Warning: No feasible maintenance parameters found")
        print("     Using model with post-annealing parameters only")

    # === Step 7: Apply Optimal Maintenance Parameters ===
    print("\n=== Step 7: Applying Optimal Maintenance Parameters ===")

    # Create a copy of model with optimal maintenance parameters
    from copy import deepcopy
    model_with_maintenance = deepcopy(model)

    if optimal_ngam is not None and optimal_gam is not None:
        print(f"  Applying optimal NGAM and GAM to model...")

        # Apply optimal NGAM
        try:
            ngam_rxn = model_with_maintenance.reactions.get_by_id(ngam_rxn_id)
            ngam_rxn.lower_bound = optimal_ngam
            print(f"    ✓ Set {ngam_rxn_id} lower bound to {optimal_ngam:.2f}")
        except KeyError:
            print(f"    Warning: NGAM reaction '{ngam_rxn_id}' not found")

        # Apply optimal GAM
        try:
            biomass_rxn = model_with_maintenance.reactions.get_by_id(biomass_reaction)
            atp_met_ids = ['atp_c', 'ATP_c', 'cpd00002_c0']

            current_gam = None
            atp_met = None
            for met_id in atp_met_ids:
                try:
                    met = model_with_maintenance.metabolites.get_by_id(met_id)
                    if met in biomass_rxn.metabolites:
                        current_gam = abs(biomass_rxn.metabolites[met])
                        atp_met = met
                        break
                except KeyError:
                    continue

            if current_gam and atp_met and optimal_gam > 0:
                scale = optimal_gam / current_gam
                current_mets = biomass_rxn.metabolites.copy()

                # Scale ATP maintenance block (ATP, H2O, ADP, Pi, H+)
                met_mappings = {
                    'h2o': ['h2o_c', 'H2O_c', 'cpd00001_c0'],
                    'adp': ['adp_c', 'ADP_c', 'cpd00008_c0'],
                    'pi': ['pi_c', 'Pi_c', 'cpd00009_c0'],
                    'h': ['h_c', 'H_c', 'cpd00067_c0']
                }

                mets_to_scale = [atp_met]
                for met_type, possible_ids in met_mappings.items():
                    for met_id in possible_ids:
                        try:
                            met = model_with_maintenance.metabolites.get_by_id(met_id)
                            if met in current_mets:
                                mets_to_scale.append(met)
                                break
                        except KeyError:
                            continue

                for met in mets_to_scale:
                    old_coef = current_mets[met]
                    biomass_rxn.add_metabolites({met: old_coef * (scale - 1.0)}, combine=True)

                print(f"    ✓ Scaled GAM from {current_gam:.2f} to {optimal_gam:.2f}")
            elif optimal_gam == 0:
                print(f"    ⚠️  Skipping GAM scaling: GAM=0 would create invalid biomass reaction")
        except KeyError:
            print(f"    Warning: Biomass reaction '{biomass_reaction}' not found")
    else:
        print("  Using model with post-annealing parameters (no maintenance optimization)")
        model_with_maintenance = model

    # === Step 8: kinGEMs Post-Tuning Multiple Solutions (with optimal maintenance) ===
    print("\n=== Step 8: Generating kinGEMs Post-Tuning Solutions (with optimal maintenance) ===")

    kingems_post_solutions = run_multi_solution_optimization(
        model=model_with_maintenance,
        processed_df=df_new,
        biomass_reaction=biomass_reaction,
        enzyme_upper_bound=enzyme_upper_bound,
        num_solutions=args.num_solutions,
        solution_pool_gap=args.pool_gap,
        solver_name=solver_name,
        medium=medium,
        medium_upper_bound=medium_upper_bound,
        label="kinGEMs post-tuning (with optimal maintenance)"
    )

    # === Step 9: Analyze Solution Spaces ===
    print("\n=== Step 9: Analyzing Solution Spaces ===")

    analyzer = SolutionSpaceAnalyzer(results_dir)

    # Analyze flux distributions
    if len(cobrapy_solutions) > 0:
        cobrapy_stats = analyzer.analyze_flux_distributions(
            cobrapy_solutions, "COBRApy", reactions=focus_reactions
        )
    else:
        print("\n  ⚠️  Skipping COBRApy analysis (no solutions available)")
        cobrapy_stats = None

    kingems_pre_stats = analyzer.analyze_flux_distributions(
        kingems_pre_solutions, "kinGEMs Pre-Tuning", reactions=focus_reactions
    )

    kingems_post_stats = analyzer.analyze_flux_distributions(
        kingems_post_solutions, "kinGEMs Post-Tuning (with optimal maintenance)", reactions=focus_reactions
    )

    # Compare solution spaces
    if cobrapy_stats is not None:
        comparison_df = analyzer.compare_solution_spaces(
            cobrapy_stats, kingems_pre_stats, kingems_post_stats
        )
    else:
        print("\n  ⚠️  Skipping full comparison (no COBRApy baseline)")
        # Create partial comparison between pre and post tuning
        comparison_df = pd.DataFrame({'Reaction': kingems_pre_stats['Reaction']})
        for col in ['Mean', 'Std', 'Range', 'Entropy']:
            comparison_df[f'{col}_Pre_Tuning'] = kingems_pre_stats[col]
            comparison_df[f'{col}_Post_Tuning'] = kingems_post_stats[col]

        comparison_df['Std_Reduction_Post'] = (
            (comparison_df['Std_Pre_Tuning'] - comparison_df['Std_Post_Tuning']) /
            (comparison_df['Std_Pre_Tuning'] + 1e-10)
        )

        comparison_path = os.path.join(analyzer.output_dir, "solution_space_comparison_partial.csv")
        comparison_df.to_csv(comparison_path, index=False)

    # === Step 10: Generate Visualizations ===
    print("\n=== Step 10: Generating Visualizations ===")

    # Plot objective distributions
    if len(cobrapy_solutions) > 0:
        analyzer.plot_objective_distributions(
            cobrapy_solutions, kingems_pre_solutions, kingems_post_solutions, model_name
        )
    else:
        print("  ⚠️  Skipping objective distribution plot (no COBRApy solutions)")

    # Plot solution space metrics
    if cobrapy_stats is not None:
        analyzer.plot_solution_space_metrics(comparison_df, model_name)
    else:
        print("  ⚠️  Skipping solution space metrics plot (no COBRApy baseline)")

    # Plot flux distributions for specific reactions
    if focus_reactions:
        if len(cobrapy_solutions) > 0:
            analyzer.plot_flux_distributions(
                cobrapy_solutions, kingems_pre_solutions, kingems_post_solutions,
                focus_reactions, model_name
            )
        else:
            print("  ⚠️  Skipping flux distribution plots (no COBRApy solutions)")
    else:
        # Use top 10 most variable reactions
        if len(cobrapy_solutions) > 0:
            top_variable_rxns = cobrapy_stats.nlargest(10, 'Std')['Reaction'].tolist()
        else:
            top_variable_rxns = kingems_pre_stats.nlargest(10, 'Std')['Reaction'].tolist()

        print(f"\n  Generating plots for top 10 most variable reactions")

        if len(cobrapy_solutions) > 0:
            analyzer.plot_flux_distributions(
                cobrapy_solutions, kingems_pre_solutions, kingems_post_solutions,
                top_variable_rxns, model_name
            )
        else:
            print("  ⚠️  Skipping flux distribution plots (no COBRApy solutions)")

    # === Summary ===
    print("\n" + "="*80)
    print("=== Analysis Complete ===")
    print("="*80)
    print(f"Run ID: {run_id}")
    print(f"Model: {model_name}")
    print(f"\nPipeline Steps:")
    print(f"  1. Data preparation")
    print(f"  2. COBRApy solutions (no constraints)")
    print(f"  3. kinGEMs pre-tuning solutions")
    print(f"  4. Simulated annealing")
    print(f"  5. Maintenance parameter sweep")
    print(f"  6. kinGEMs post-tuning solutions (with optimal maintenance)")
    print(f"\nSolutions Generated:")
    print(f"  COBRApy (no constraints):              {len(cobrapy_solutions)}")
    print(f"  kinGEMs pre-tuning:                    {len(kingems_pre_solutions)}")
    print(f"  kinGEMs post-tuning (w/ maintenance):  {len(kingems_post_solutions)}")
    print(f"\nAverage Objective Values:")
    if len(cobrapy_solutions) > 0:
        print(f"  COBRApy:      {np.mean([obj for obj, _ in cobrapy_solutions]):.4f} ± {np.std([obj for obj, _ in cobrapy_solutions]):.4f}")
    else:
        print(f"  COBRApy:      N/A (no solutions generated)")
    print(f"  Pre-tuning:   {np.mean([obj for obj, _ in kingems_pre_solutions]):.4f} ± {np.std([obj for obj, _ in kingems_pre_solutions]):.4f}")
    print(f"  Post-tuning:  {np.mean([obj for obj, _ in kingems_post_solutions]):.4f} ± {np.std([obj for obj, _ in kingems_post_solutions]):.4f}")

    if optimal_ngam is not None and optimal_gam is not None:
        print(f"\nOptimal Maintenance Parameters:")
        print(f"  NGAM: {optimal_ngam:.2f} mmol/gDW/h")
        print(f"  GAM:  {optimal_gam:.2f} mmol ATP/gDW")
        print(f"  Biomass with optimal maintenance: {optimal_biomass:.4f}")

    print(f"\nAnnealing Performance:")
    print(f"  Initial biomass:        {biomasses[0]:.4f}")
    print(f"  Post-annealing biomass: {biomasses[-1]:.4f}")
    print(f"  Improvement:            {improvement:.1f}%")

    if optimal_biomass is not None:
        total_improvement = (optimal_biomass - biomasses[0]) / biomasses[0] * 100 if biomasses[0] > 0 else 0
        print(f"  Final biomass (w/ maintenance): {optimal_biomass:.4f}")
        print(f"  Total improvement:              {total_improvement:.1f}%")

    print(f"\nResults directory: {results_dir}")
    print("="*80)


if __name__ == '__main__':
    main()
