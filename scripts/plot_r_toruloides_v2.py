"""Plot figures for the case study for R toruloides (v2)."""

import json
import os

import cobra

from kinGEMs.modeling.tuning import apply_model_edits
from kinGEMs.plots import plot_r_toruloides_case_study_v2


# Glucose experiments
# GEXP_RUN_ID = "rhto_20260814_1440"
# GLIM_RUN_ID = "rhto_20260814_5840"

# # Acetate experiments
# AEXP_RUN_ID = "rhto_20260815_6288"
# ALIM_RUN_ID = "rhto_20260814_9995"

# # Xylose experiments
# XEXP_RUN_ID = "rhto_20260814_4023"
# XLIM_RUN_ID = "rhto_20260815_1563"



# With fixed bounds + blocking other substrates
# Glucose experiments
GEXP_RUN_ID = "rhto_20260815_5191"
GLIM_RUN_ID = "rhto_20260815_4765"

# Acetate experiments
AEXP_RUN_ID = "rhto_20260815_7806"
ALIM_RUN_ID = "rhto_20260815_5440"

# Xylose experiments
XEXP_RUN_ID = "rhto_20260815_1121"
XLIM_RUN_ID = "rhto_20260815_5917" # with maintenance sweep
#XLIM_RUN_ID = "rhto_20260815_9635" # without maintenance sweep



# Project root
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Output path (v2; do not overwrite the v1 figure)
OUTPUT_PATH = os.path.join(
    PROJECT_ROOT, "results/figures/r_toruloides_case_study_v2.png"
)

RESULTS_ROOT = os.path.join(PROJECT_ROOT, "results/tuning_results")
CONFIGS_ROOT = os.path.join(PROJECT_ROOT, "configs")

# Stock rhto-GEM (reversible). Xylose DAD/RK reactions are added from the
# xylose config's model_edits. Do not convert to irreversible here.
MODEL_PATH = os.path.join(PROJECT_ROOT, "data/raw/rhto.xml")
XY_CONFIG_PATH = os.path.join(CONFIGS_ROOT, "r_toruloides_xy_exp.json")

CONDITION_CONFIGS = {
    "Gexp": os.path.join(CONFIGS_ROOT, "r_toruloides_gluc_exp.json"),
    "GNlim": os.path.join(CONFIGS_ROOT, "r_toruloides_gluc_lim.json"),
    "Xexp": os.path.join(CONFIGS_ROOT, "r_toruloides_xy_exp.json"),
    "XNlim": os.path.join(CONFIGS_ROOT, "r_toruloides_xy_lim.json"),
    "Aexp": os.path.join(CONFIGS_ROOT, "r_toruloides_ac_exp.json"),
    "ANlim": os.path.join(CONFIGS_ROOT, "r_toruloides_ac_lim.json"),
}

# Input arguments:
# csv with fluxes and enzyme concentrations
GEXP_PATH = os.path.join(RESULTS_ROOT, GEXP_RUN_ID, "df_FBA.csv")
GLIM_PATH = os.path.join(RESULTS_ROOT, GLIM_RUN_ID, "df_FBA.csv")
AEXP_PATH = os.path.join(RESULTS_ROOT, AEXP_RUN_ID, "df_FBA.csv")
ALIM_PATH = os.path.join(RESULTS_ROOT, ALIM_RUN_ID, "df_FBA.csv")
XEXP_PATH = os.path.join(RESULTS_ROOT, XEXP_RUN_ID, "df_FBA.csv")
XLIM_PATH = os.path.join(RESULTS_ROOT, XLIM_RUN_ID, "df_FBA.csv")

panel_a_reactions = {
    "GDH1": "r_0471",  # NADP-dependent glutamate dehydrogenase
    "FAS1_palmitate": "r_2140",  # fatty-acyl-CoA synthase (n-C16:0CoA)
    "FAS2_stearate": "r_2141",  # fatty-acyl-CoA synthase (n-C18:0CoA)
}

# Measured substrate uptake rates (mmol/gDW/h) from condition configs
# (glucose r_1714, xylose r_1718, acetate r_1634).
GEXP_SUBS_UPTAKE = 2.489
GLIM_SUBS_UPTAKE = 0.41
XEXP_SUBS_UPTAKE = 1.86
XLIM_SUBS_UPTAKE = 0.4345
AEXP_SUBS_UPTAKE = 6.1
ALIM_SUBS_UPTAKE = 1.9706


def load_edited_rhto_model(model_path, config_path):
    """Load stock rhto-GEM and apply xylose-pathway edits from the config."""
    with open(config_path) as fh:
        edits = json.load(fh).get("model_edits")
    model = cobra.io.read_sbml_model(model_path)
    if edits:
        apply_model_edits(model, edits, verbose=False)
    return model


def warn_config_anomalies():
    """Flag enzyme-pool / biomass-goal settings that distort condition identity."""
    pools = {}
    goals = {}
    for label, path in CONDITION_CONFIGS.items():
        with open(path) as fh:
            cfg = json.load(fh)
        pools[label] = cfg.get("enzyme_upper_bound")
        goals[label] = cfg.get("simulated_annealing", {}).get("biomass_goal")
    typical = [p for lab, p in pools.items() if lab != "Aexp" and p is not None]
    if typical and pools.get("Aexp") is not None:
        med = sorted(typical)[len(typical) // 2]
        if pools["Aexp"] > 3.0 * med:
            print(
                f"  WARNING: Aexp enzyme_upper_bound={pools['Aexp']} "
                f"looks like raw Ptot, not Ptot·f·σ "
                f"(other conditions ~{min(typical):.2f}–{max(typical):.2f})."
            )
    if goals.get("Xexp") is not None and goals.get("XNlim") == goals.get("Xexp"):
        print(
            f"  WARNING: XNlim biomass_goal={goals['XNlim']} is identical "
            f"to Xexp; nitrogen-limited xylose growth should be lower."
        )


if __name__ == "__main__":
    warn_config_anomalies()
    rhto_model = load_edited_rhto_model(MODEL_PATH, XY_CONFIG_PATH)
    plot_r_toruloides_case_study_v2(
        gexp_path=GEXP_PATH,
        glim_path=GLIM_PATH,
        aexp_path=AEXP_PATH,
        alim_path=ALIM_PATH,
        xexp_path=XEXP_PATH,
        xlim_path=XLIM_PATH,
        panel_a_reactions=panel_a_reactions,
        gexp_subs_uptake=GEXP_SUBS_UPTAKE,
        glim_subs_uptake=GLIM_SUBS_UPTAKE,
        aexp_subs_uptake=AEXP_SUBS_UPTAKE,
        alim_subs_uptake=ALIM_SUBS_UPTAKE,
        xexp_subs_uptake=XEXP_SUBS_UPTAKE,
        xlim_subs_uptake=XLIM_SUBS_UPTAKE,
        model=rhto_model,
        output_path=OUTPUT_PATH,
        show=False,
    )
