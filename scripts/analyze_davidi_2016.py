#!/usr/bin/env python
"""
analyze_davidi_2016.py
======================
Analyse the Davidi et al. 2016 PNAS supplementary dataset:
  data/experimental/Davidi_2016_pnas.1514240113.sd01.xlsx

Sheets used
-----------
  'kcat vs. kmax'  — 132 enzyme-reaction pairs with literature kcat and
                     in-vivo kmax values (both in s-1).
  'kapp 1s'        — per-reaction, per-condition apparent catalytic rates
                     (kapp, in s-1) across 31 growth conditions.

Output
------
  results/davidi_2016_kcat_kmax_analysis.png
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------
# Allow running from project root or from scripts/ directory
# --------------------------------------------------------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from kinGEMs.plots import plot_davidi_kcat_kmax_analysis

# --------------------------------------------------------------------------
# Default paths
# --------------------------------------------------------------------------
DEFAULT_DATA_PATH = os.path.join(
    PROJECT_ROOT, "data", "experimental",
    "Davidi_2016_pnas.1514240113.sd01.xlsx"
)
DEFAULT_OUTPUT_PATH = os.path.join(
    PROJECT_ROOT, "results", "davidi_2016_kcat_kmax_analysis.png"
)

# Excel header is at row index 2 for the relevant sheets (0-indexed)
HEADER_ROW = 2


def load_kcat_kmax(xlsx_path: str) -> pd.DataFrame:
    """Load the 'kcat vs. kmax' sheet and return a clean DataFrame."""
    df = pd.read_excel(xlsx_path, sheet_name="kcat vs. kmax", header=HEADER_ROW)
    required = {"kcat [s-1]", "kmax [s-1]", "kcat / kmax"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Expected columns not found in 'kcat vs. kmax' sheet: {missing}\n"
            f"Actual columns: {list(df.columns)}"
        )
    df = df.dropna(subset=["kcat [s-1]", "kmax [s-1]"])
    print(f"  Loaded {len(df)} kcat/kmax pairs from 'kcat vs. kmax' sheet.")
    return df


def load_kapp(xlsx_path: str) -> pd.DataFrame:
    """Load the 'kapp 1s' sheet (condition columns as floats, NaN = zero flux)."""
    df = pd.read_excel(xlsx_path, sheet_name="kapp 1s", header=HEADER_ROW)
    id_cols = [c for c in df.columns if c in ("reaction (model name)", "bnumber")]
    cond_cols = [c for c in df.columns if c not in id_cols]
    df[cond_cols] = df[cond_cols].apply(pd.to_numeric, errors="coerce")
    print(
        f"  Loaded kapp sheet: {len(df)} reactions × {len(cond_cols)} conditions."
    )
    return df


def print_summary(df_kcat_kmax: pd.DataFrame, df_kapp: pd.DataFrame) -> None:
    """Print key statistics to stdout."""
    kcat = df_kcat_kmax["kcat [s-1]"].values
    kmax = df_kcat_kmax["kmax [s-1]"].values
    ratio = df_kcat_kmax["kcat / kmax"].values

    print("\n=== kcat [s-1] ===")
    print(pd.Series(kcat).describe().to_string())

    print("\n=== kmax [s-1] ===")
    print(pd.Series(kmax).describe().to_string())

    print("\n=== kcat / kmax ===")
    print(pd.Series(ratio).describe().to_string())
    n_above = (ratio > 1).sum()
    print(f"kcat > kmax: {n_above}/{len(ratio)} ({100*n_above/len(ratio):.1f}%)")

    if df_kapp is not None:
        id_cols = [c for c in df_kapp.columns if c in ("reaction (model name)", "bnumber")]
        cond_cols = [c for c in df_kapp.columns if c not in id_cols]
        kapp_flat = df_kapp[cond_cols].values.flatten().astype(float)
        kapp_flat = kapp_flat[np.isfinite(kapp_flat) & (kapp_flat > 0)]
        print(f"\n=== kapp [s-1] — all conditions (N = {len(kapp_flat)}) ===")
        print(pd.Series(kapp_flat).describe().to_string())


def main():
    parser = argparse.ArgumentParser(
        description="Analyse Davidi et al. 2016 kcat / kmax / kapp data."
    )
    parser.add_argument(
        "--data", default=DEFAULT_DATA_PATH,
        help="Path to the Davidi 2016 Excel file (default: %(default)s)"
    )
    parser.add_argument(
        "--output", default=DEFAULT_OUTPUT_PATH,
        help="Output PNG path (default: %(default)s)"
    )
    parser.add_argument(
        "--no_kapp", action="store_true",
        help="Skip loading and plotting kapp data (faster)"
    )
    parser.add_argument(
        "--show", action="store_true",
        help="Display the figure interactively (requires a display)"
    )
    args = parser.parse_args()

    if not os.path.isfile(args.data):
        sys.exit(f"ERROR: Data file not found: {args.data}")

    print(f"Loading data from:\n  {args.data}\n")

    df_kcat_kmax = load_kcat_kmax(args.data)
    df_kapp = None if args.no_kapp else load_kapp(args.data)

    print_summary(df_kcat_kmax, df_kapp)

    print(f"\nGenerating plot → {args.output}")
    fig = plot_davidi_kcat_kmax_analysis(
        df_kcat_kmax=df_kcat_kmax,
        df_kapp=df_kapp,
        output_path=args.output,
        show=args.show,
    )

    print("Done.")
    return fig


if __name__ == "__main__":
    main()
