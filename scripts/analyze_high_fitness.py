#!/usr/bin/env python3
"""
Analyze High Fitness Values in Experimental Data
================================================

This script analyzes the experimental fitness data to find gene-carbon combinations
with fitness values greater than 2.0, providing summary statistics and detailed listings.

Usage:
    python analyze_high_fitness.py [--threshold 2.0]
"""

import argparse
import sys
import os
import numpy as np
import pandas as pd

# Add kinGEMs to path
sys.path.insert(0, os.path.abspath('..'))

from kinGEMs.config import ECOLI_VALIDATION_DIR
from kinGEMs.validation_utils import (
    load_data,
    load_environment,
    match_model_data,
    model_adjustments,
    prepare_model
)
import cobra


def analyze_high_fitness(threshold=2.0):
    """
    Analyze experimental fitness data for high fitness values.

    Parameters
    ----------
    threshold : float
        Fitness threshold above which to report gene-carbon combinations

    Returns
    -------
    dict
        Dictionary containing analysis results
    """
    print(f"Analyzing experimental fitness data for values > {threshold}")
    print("="*60)

    # Load model for gene matching
    print("Loading E. coli iML1515 model...")
    model_path = "models/iML1515_GEM_20251126_5181.xml"
    if not os.path.exists(model_path):
        print(f"Error: Model file not found: {model_path}")
        print("Available model files:")
        if os.path.exists("models"):
            for f in os.listdir("models"):
                if f.endswith(".xml"):
                    print(f"  - {f}")
        sys.exit(1)

    model = cobra.io.read_sbml_model(model_path)
    model = prepare_model(model)

    # Load experimental data and environment
    print("Loading experimental data...")
    name_medium_model, name_carbon_model, name_carbon_experiment = load_environment(ECOLI_VALIDATION_DIR)
    data_experiments, data_genes, data_fitness = load_data(ECOLI_VALIDATION_DIR)

    # Match model data
    print("Matching model and experimental data...")
    name_genes_matched, name_carbon_experiment_matched, name_carbon_model_matched, data_fitness_matched = match_model_data(
        model=model,
        name_carbon_model=name_carbon_model,
        name_carbon_experiment=name_carbon_experiment,
        data_experiments=data_experiments,
        data_genes=data_genes,
        data_fitness=data_fitness
    )

    # Apply model adjustments
    print("Applying model adjustments...")
    model_adj, name_genes_matched_adj, name_carbon_experiment_matched_adj, name_carbon_model_matched_adj, data_fitness_matched_adj = model_adjustments(
        adj_strain=True,
        adj_essential=True,
        adj_carbon=True,
        model=model,
        name_genes_matched=name_genes_matched,
        name_carbon_experiment_matched=name_carbon_experiment_matched,
        name_carbon_model_matched=name_carbon_model_matched,
        data_fitness_matched=data_fitness_matched
    )

    print("\nData dimensions after adjustments:")
    print(f"  Genes: {len(name_genes_matched_adj)}")
    print(f"  Carbon sources: {len(name_carbon_model_matched_adj)}")
    print(f"  Fitness matrix shape: {data_fitness_matched_adj.shape}")

    # Analyze fitness values
    print("\nFitness data statistics:")
    print(f"  Total values: {data_fitness_matched_adj.size}")
    print(f"  Valid values (not NaN): {np.sum(~np.isnan(data_fitness_matched_adj))}")
    print(f"  Mean fitness: {np.nanmean(data_fitness_matched_adj):.4f}")
    print(f"  Std fitness: {np.nanstd(data_fitness_matched_adj):.4f}")
    print(f"  Min fitness: {np.nanmin(data_fitness_matched_adj):.4f}")
    print(f"  Max fitness: {np.nanmax(data_fitness_matched_adj):.4f}")

    # Find high fitness values
    high_fitness_mask = data_fitness_matched_adj > threshold
    high_fitness_count = np.sum(high_fitness_mask)

    print(f"\nHigh fitness analysis (threshold > {threshold}):")
    print(f"  Count: {high_fitness_count}")
    print(f"  Percentage: {100 * high_fitness_count / np.sum(~np.isnan(data_fitness_matched_adj)):.2f}% of valid values")

    if high_fitness_count == 0:
        print(f"  No fitness values found above {threshold}")
        return {
            'threshold': threshold,
            'count': 0,
            'combinations': [],
            'fitness_stats': {
                'mean': np.nanmean(data_fitness_matched_adj),
                'std': np.nanstd(data_fitness_matched_adj),
                'min': np.nanmin(data_fitness_matched_adj),
                'max': np.nanmax(data_fitness_matched_adj)
            }
        }

    # Extract high fitness combinations
    high_fitness_combinations = []
    gene_indices, carbon_indices = np.where(high_fitness_mask)

    for gene_idx, carbon_idx in zip(gene_indices, carbon_indices):
        gene_id = name_genes_matched_adj[gene_idx]
        carbon_id = name_carbon_model_matched_adj[carbon_idx]
        fitness_value = data_fitness_matched_adj[gene_idx, carbon_idx]

        high_fitness_combinations.append({
            'gene_id': gene_id,
            'carbon_source': carbon_id,
            'fitness': fitness_value,
            'gene_index': gene_idx,
            'carbon_index': carbon_idx
        })

    # Sort by fitness value (highest first)
    high_fitness_combinations.sort(key=lambda x: x['fitness'], reverse=True)

    print("\nTop 20 highest fitness combinations:")
    print("-" * 80)
    print(f"{'Gene ID':<15} {'Carbon Source':<25} {'Fitness':<10} {'Indices':<15}")
    print("-" * 80)

    for i, combo in enumerate(high_fitness_combinations[:20]):
        print(f"{combo['gene_id']:<15} {combo['carbon_source']:<25} {combo['fitness']:<10.4f} "
              f"({combo['gene_index']},{combo['carbon_index']})")

    if len(high_fitness_combinations) > 20:
        print(f"... and {len(high_fitness_combinations) - 20} more")

    # Gene frequency analysis
    gene_counts = {}
    carbon_counts = {}

    for combo in high_fitness_combinations:
        gene_id = combo['gene_id']
        carbon_id = combo['carbon_source']

        if gene_id not in gene_counts:
            gene_counts[gene_id] = []
        gene_counts[gene_id].append(combo['fitness'])

        if carbon_id not in carbon_counts:
            carbon_counts[carbon_id] = []
        carbon_counts[carbon_id].append(combo['fitness'])

    print("\nGenes with most high fitness combinations:")
    print("-" * 50)
    gene_summary = [(gene, len(values), np.mean(values)) for gene, values in gene_counts.items()]
    gene_summary.sort(key=lambda x: x[1], reverse=True)

    for i, (gene, count, mean_fitness) in enumerate(gene_summary[:10]):
        print(f"{i+1:2d}. {gene:<15} {count:3d} combinations, mean fitness: {mean_fitness:.4f}")

    print("\nCarbon sources with most high fitness combinations:")
    print("-" * 55)
    carbon_summary = [(carbon, len(values), np.mean(values)) for carbon, values in carbon_counts.items()]
    carbon_summary.sort(key=lambda x: x[1], reverse=True)

    for i, (carbon, count, mean_fitness) in enumerate(carbon_summary[:10]):
        print(f"{i+1:2d}. {carbon:<25} {count:3d} combinations, mean fitness: {mean_fitness:.4f}")

    # Save detailed results
    output_dir = "results/fitness_data_analysis"
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"high_fitness_analysis_threshold_{threshold}.csv")
    df_results = pd.DataFrame(high_fitness_combinations)
    df_results.to_csv(output_file, index=False)
    print(f"\nDetailed results saved to: {output_file}")

    # Distribution analysis
    fitness_ranges = [
        (2.0, 3.0),
        (3.0, 4.0),
        (4.0, 5.0),
        (5.0, float('inf'))
    ]

    print(f"\nFitness distribution for high values (> {threshold}):")
    print("-" * 40)

    for min_val, max_val in fitness_ranges:
        if min_val < threshold:
            continue

        if max_val == float('inf'):
            range_mask = [combo['fitness'] >= min_val for combo in high_fitness_combinations]
            range_label = f">= {min_val}"
        else:
            range_mask = [min_val <= combo['fitness'] < max_val for combo in high_fitness_combinations]
            range_label = f"{min_val}-{max_val}"

        range_count = sum(range_mask)
        if range_count > 0:
            range_fitness = [combo['fitness'] for i, combo in enumerate(high_fitness_combinations) if range_mask[i]]
            print(f"  {range_label:<10} {range_count:4d} combinations (mean: {np.mean(range_fitness):.4f})")

    return {
        'threshold': threshold,
        'count': high_fitness_count,
        'combinations': high_fitness_combinations,
        'gene_summary': gene_summary,
        'carbon_summary': carbon_summary,
        'fitness_stats': {
            'mean': np.nanmean(data_fitness_matched_adj),
            'std': np.nanstd(data_fitness_matched_adj),
            'min': np.nanmin(data_fitness_matched_adj),
            'max': np.nanmax(data_fitness_matched_adj)
        }
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--threshold', type=float, default=2.0,
                       help='Fitness threshold above which to report combinations (default: 2.0)')

    args = parser.parse_args()

    try:
        results = analyze_high_fitness(threshold=args.threshold)

        print("\n" + "="*60)
        print("SUMMARY")
        print("="*60)
        print(f"Threshold: {results['threshold']}")
        print(f"High fitness combinations found: {results['count']}")

        if results['count'] > 0:
            print(f"Highest fitness value: {max(combo['fitness'] for combo in results['combinations']):.4f}")
            print(f"Gene with most hits: {results['gene_summary'][0][0]} ({results['gene_summary'][0][1]} combinations)")
            print(f"Carbon source with most hits: {results['carbon_summary'][0][0]} ({results['carbon_summary'][0][1]} combinations)")

        print("="*60)

    except Exception as e:
        print(f"Error during analysis: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
