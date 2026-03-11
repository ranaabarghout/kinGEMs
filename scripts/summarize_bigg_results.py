#!/usr/bin/env python3
"""
summarize_bigg_results.py
=========================
Build a summary statistics table for all BiGG model runs processed by kinGEMs.

For each model (using the most recent completed run) the table includes:
  - Model name & organism
  - Model size: unique reactions, unique genes
  - CPI-Pred characterisation: unique proteins, unique substrates, rxn-gene pairs
  - Simulated annealing: initial & final biomass, improvement %, iterations
  - kcat statistics from df_new: median initial, median tuned, fold change

Output
------
  results/BiGG_models_summary.csv          (full machine-readable table)
  results/BiGG_models_summary_display.csv  (rounded, display-ready table)
"""

import json
import os
import glob
from collections import defaultdict

import pandas as pd
import numpy as np

# ── Paths ─────────────────────────────────────────────────────────────────────
PROJ_ROOT    = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
RESULTS_BASE = os.path.join(PROJ_ROOT, 'results', 'tuning_results', 'BiGG_models')
INTERIM_BASE = os.path.join(PROJ_ROOT, 'data', 'interim')
PROC_BASE    = os.path.join(PROJ_ROOT, 'data', 'processed')
CONFIGS_DIR  = os.path.join(PROJ_ROOT, 'configs')
OUT_DIR      = os.path.join(PROJ_ROOT, 'results')

# ── Known organism map (from config files + BiGG model naming conventions) ───
# iAM_P* = Plasmodium species (malaria parasites)
# iIS312* = Trypanosoma cruzi life-cycle stages
# iAB_RBC = Human erythrocyte
# RECON / Recon3D / iMM1415 = Human
# iMM904 = Mouse
# iND750 / iYO844 / iYS854 etc = Yeast (S. cerevisiae)
# iJB785 = Synechococcus elongatus
# iJN746 / iJN1463 = Pseudomonas putida
# iNJ661 = M. tuberculosis
# iIT341 = H. pylori
# iLB1027_lipid = Phaeodactylum tricornutum (diatom)
# iRC1080 = Chlamydomonas reinhardtii
# iCHOv1* = Chinese Hamster Ovary
# iPC815 = Plasmodium cynomolgi
# iSynCJ816 = Synechocystis
# iYL1228 = Yarrowia lipolytica
# iLJ478 = Leishmania major
# iHN637 = Leishmania major
# iSBO_1134 = Salmonella
# Most iEC* / iAF* / iJO* / iJR* / iML* = E. coli strains

ORGANISM_HINTS = {
    # Exact model name → organism
    'RECON1':        'Human', 'Recon3D': 'Human',
    'iAB_RBC_283':   'Human (RBC)', 'iMM1415': 'Human',
    'iMM904':        'Mouse',
    'iCHOv1':        'CHO', 'iCHOv1_DG44': 'CHO',
    'iND750':        'Yeast (S. cerevisiae)', 'iYO844': 'Yeast (S. cerevisiae)',
    'iYS854':        'Yeast (S. cerevisiae)', 'iYS1720': 'Yeast (S. cerevisiae)',
    'iJB785':        'Synechococcus elongatus',
    'iSynCJ816':     'Synechocystis sp.',
    'iJN746':        'Pseudomonas putida', 'iJN1463': 'Pseudomonas putida',
    'iNJ661':        'M. tuberculosis',
    'iIT341':        'H. pylori',
    'iLB1027_lipid': 'Phaeodactylum tricornutum',
    'iRC1080':       'Chlamydomonas reinhardtii',
    'iLJ478':        'Leishmania major', 'iHN637': 'Leishmania major',
    'iSB619':        'Staphylococcus aureus',
    'iYL1228':       'Yarrowia lipolytica',
    'iEK1008':       'Klebsiella pneumoniae',
    'iAF692':        'Methanosarcina barkeri',
    'iAF987':        'Geobacter metallireducens',
    'iNF517':        'Methanobacterium sp.',
    'iPC815':        'P. cynomolgi',
    'iIS312':             'T. cruzi', 'iIS312_Amastigote': 'T. cruzi',
    'iIS312_Epimastigote':'T. cruzi', 'iIS312_Trypomastigote':'T. cruzi',
    # Candida albicans
    'iCN718':        'C. albicans', 'iCN900': 'C. albicans',
    # E. coli core (small model)
    'e_coli_core':   'E. coli',
}

PLASMODIUM_MODELS = {
    'iAM_Pb448': 'P. berghei',
    'iAM_Pc455': 'P. cynomolgi',
    'iAM_Pf480': 'P. falciparum',
    'iAM_Pk459': 'P. knowlesi',
    'iAM_Pv461': 'P. vivax',
}

ECOLI_PREFIXES = (
    'iML', 'iJO', 'iJR', 'iAF1260', 'iAF1260b', 'e_coli_core',
    'iEC', 'iEK', 'iEco', 'iB21', 'iBWG', 'iG2583', 'iAPECO1',
    'iEcD', 'iEcE', 'iEcH', 'iEcS', 'iEcolC', 'STM_v1_0',
    'iSBO_1134', 'iSF_', 'iSFV', 'iSFx', 'iSSON', 'iS_', 'iSbBS',
    'iSDY', 'iNRG', 'iUMN', 'iUTI', 'iWFL', 'iLF82', 'iZ_',
    'ic_', 'iY75', 'iEKO',
)

SALMONELLA_MODELS = ('STM_v1_0', 'iSBO_1134', 'iSDY_1059')


# Normalise organism strings that may come from config files
_ORG_NORMALISE = {
    'E coli': 'E. coli', 'Ecoli': 'E. coli', 'E.coli': 'E. coli',
    'e coli': 'E. coli',
    'Human':  'Human',
}

def infer_organism(model_name):
    if model_name in ORGANISM_HINTS:
        return ORGANISM_HINTS[model_name]
    if model_name in PLASMODIUM_MODELS:
        return PLASMODIUM_MODELS[model_name]
    for pfx in SALMONELLA_MODELS:
        if model_name.startswith(pfx):
            return 'Salmonella enterica'
    for pfx in ECOLI_PREFIXES:
        if model_name.startswith(pfx):
            return 'E. coli'
    return 'Unknown'


def load_config_lookup():
    lookup = {}
    for cf in glob.glob(os.path.join(CONFIGS_DIR, '*.json')):
        try:
            with open(cf) as f:
                c = json.load(f)
            if 'model_name' in c:
                lookup[c['model_name']] = c
        except Exception:
            pass
    return lookup


def build_summary():
    run_dirs = sorted(glob.glob(os.path.join(RESULTS_BASE, '*/')))
    config_lookup = load_config_lookup()

    # Group runs by model name (strip _YYYYMMDD_XXXX suffix)
    model_runs = defaultdict(list)
    for d in run_dirs:
        name = os.path.basename(d.rstrip('/'))
        parts = name.rsplit('_', 2)
        model_runs[parts[0]].append(d)

    rows = []
    for model_name, dirs in sorted(model_runs.items()):
        run_with_data = [d for d in sorted(dirs) if 'df_new.csv' in os.listdir(d)]
        status = 'Complete' if run_with_data else 'Incomplete'

        # Organism: prefer config file, fall back to inference
        cfg = config_lookup.get(model_name, {})
        raw_org = cfg.get('organism') or ''
        organism = _ORG_NORMALISE.get(raw_org, raw_org) or infer_organism(model_name)
        if not organism:
            organism = infer_organism(model_name)

        # ── Defaults ──────────────────────────────────────────────────────────
        n_reactions = n_genes = n_proteins = n_compounds = n_pairs = None
        initial_bm = final_bm = improvement = n_iters = None
        median_init_kcat = median_tuned_kcat = median_fold = None
        n_runs = len(dirs)

        if run_with_data:
            latest = run_with_data[-1]
            df_new = pd.read_csv(os.path.join(latest, 'df_new.csv'),
                                 low_memory=False)

            # Reaction / gene counts from df_new
            if 'Reactions' in df_new.columns:
                n_reactions = df_new['Reactions'].nunique()
            if 'Single_gene' in df_new.columns:
                n_genes = df_new['Single_gene'].nunique()

            # kcat statistics
            if 'kcat_mean' in df_new.columns and 'kcat_updated' in df_new.columns:
                init_vals  = pd.to_numeric(df_new['kcat_mean'],    errors='coerce').dropna()
                tuned_vals = pd.to_numeric(df_new['kcat_updated'], errors='coerce').dropna()
                merged_kc  = pd.merge(
                    df_new[['Reactions', 'Single_gene', 'kcat_mean']].dropna(),
                    df_new[['Reactions', 'Single_gene', 'kcat_updated']].dropna(),
                    on=['Reactions', 'Single_gene'], how='inner'
                )
                if len(merged_kc):
                    merged_kc['kcat_mean']    = pd.to_numeric(merged_kc['kcat_mean'],    errors='coerce')
                    merged_kc['kcat_updated'] = pd.to_numeric(merged_kc['kcat_updated'], errors='coerce')
                    merged_kc = merged_kc.dropna()
                    median_init_kcat  = merged_kc['kcat_mean'].median()
                    median_tuned_kcat = merged_kc['kcat_updated'].median()
                    fold = merged_kc['kcat_updated'] / merged_kc['kcat_mean']
                    median_fold = fold.median()

            # SA results — stored in iterations.csv (Iteration, Biomass)
            sa_path = os.path.join(latest, 'iterations.csv')
            if os.path.exists(sa_path):
                sa = pd.read_csv(sa_path)
                bm_col = next((c for c in sa.columns
                               if c.lower() in ('biomass', 'biomass_flux')), None)
                if bm_col and len(sa) > 0:
                    bm_vals = pd.to_numeric(sa[bm_col], errors='coerce').dropna()
                    if len(bm_vals):
                        initial_bm = bm_vals.iloc[0]
                        final_bm   = bm_vals.max()   # best biomass achieved
                        improvement = (
                            (final_bm - initial_bm) / initial_bm * 100
                            if initial_bm and initial_bm > 0 else None
                        )
                        n_iters = len(sa)

        # CPI-Pred processed data (lives in data/processed/<model>/)
        proc_path = os.path.join(PROC_BASE, model_name,
                                 f'{model_name}_processed_data.csv')
        if os.path.exists(proc_path):
            proc = pd.read_csv(proc_path, low_memory=False)
            n_pairs = len(proc)
            if 'Single_gene' in proc.columns:
                n_proteins = proc['Single_gene'].nunique()
            for col in ('CMPD_SMILES', 'Cleaned Substrate', 'DB Name'):
                if col in proc.columns:
                    n_compounds = proc[col].dropna().nunique()
                    break

        rows.append({
            'Model':                    model_name,
            'Organism':                 organism,
            'Status':                   status,
            'N Runs':                   n_runs,
            'Unique Reactions':         n_reactions,
            'Unique Genes':             n_genes,
            'Unique Proteins (CPI-Pred)':   n_proteins,
            'Unique Substrates (CPI-Pred)': n_compounds,
            'Rxn-Gene Pairs (CPI-Pred)':    n_pairs,
            'Median Initial kcat (1/hr)':   round(median_init_kcat,  2) if median_init_kcat  else None,
            'Median Tuned kcat (1/hr)':     round(median_tuned_kcat, 2) if median_tuned_kcat else None,
            'Median Fold Change':           round(median_fold, 2)        if median_fold        else None,
            'Initial Biomass':          round(initial_bm,  4) if initial_bm  else None,
            'Final Biomass':            round(final_bm,    4) if final_bm    else None,
            'Biomass Improvement (%)':  round(improvement, 1) if improvement else None,
            'SA Iterations':            n_iters,
        })

    df = pd.DataFrame(rows)

    # ── Overall stats printout ─────────────────────────────────────────────────
    complete = df[df['Status'] == 'Complete']
    print("=" * 70)
    print("kinGEMs BiGG Models – Pipeline Summary")
    print("=" * 70)
    print(f"  Total unique models attempted : {len(df)}")
    print(f"  Complete (df_new present)     : {len(complete)}")
    print(f"  Incomplete / no data          : {len(df) - len(complete)}")
    print(f"  Total pipeline runs           : {df['N Runs'].sum()}")
    print()

    # Totals across all complete models
    total_proteins   = complete['Unique Proteins (CPI-Pred)'].sum()
    total_substrates = complete['Unique Substrates (CPI-Pred)'].sum()
    total_pairs      = complete['Rxn-Gene Pairs (CPI-Pred)'].sum()
    print(f"  Σ Proteins mined (CPI-Pred)   : {int(total_proteins):,}")
    print(f"  Σ Substrates characterised    : {int(total_substrates):,}")
    print(f"  Σ Rxn-gene pairs (CPI-Pred)   : {int(total_pairs):,}")
    print()

    orgs = complete['Organism'].value_counts()
    print(f"  Unique organisms (complete)   : {complete['Organism'].nunique()}")
    print("  Organisms (# models):")
    for org, cnt in orgs.items():
        print(f"    {org:40s}  {cnt}")
    print()

    print("  Per-model distributions (complete runs):")
    for col in ('Unique Reactions', 'Unique Genes',
                'Unique Proteins (CPI-Pred)', 'Unique Substrates (CPI-Pred)',
                'Rxn-Gene Pairs (CPI-Pred)'):
        vals = complete[col].dropna()
        if len(vals):
            print(f"    {col}:")
            print(f"      Median = {vals.median():,.0f}  |  Range = {vals.min():,.0f} – {vals.max():,.0f}")

    print()
    print("  kcat & biomass statistics (complete runs):")
    for col in ('Median Initial kcat (1/hr)', 'Median Tuned kcat (1/hr)',
                'Median Fold Change', 'Biomass Improvement (%)'):
        vals = complete[col].dropna()
        if len(vals):
            print(f"    {col} (n={len(vals)}):")
            print(f"      Median = {vals.median():.2f}  |  Range = {vals.min():.2f} – {vals.max():.2f}")

    print("=" * 70)

    # Save
    out_full = os.path.join(OUT_DIR, 'BiGG_models_summary.csv')
    df.to_csv(out_full, index=False)
    print(f"\nFull table saved to: {out_full}")

    return df


if __name__ == '__main__':
    df = build_summary()
    print(df[df['Status'] == 'Complete'][[
        'Model', 'Organism', 'Unique Reactions', 'Unique Genes',
        'Unique Proteins (CPI-Pred)', 'Unique Substrates (CPI-Pred)',
        'Median Initial kcat (1/hr)', 'Median Tuned kcat (1/hr)',
        'Median Fold Change', 'Biomass Improvement (%)',
    ]].to_string(index=False))
