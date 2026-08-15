"""Plot figures for the case study for R toruloides."""

import json
import os

import cobra

from kinGEMs.modeling.tuning import apply_model_edits
from kinGEMs.plots import plot_r_toruloides_case_study


#-------Baseline experiments using high enzyme upper bound (no sigma and f factors)------

# Glucose experiments
GEXP_RUN_ID = "rhto_20260814_1440"
GLIM_RUN_ID = "rhto_20260814_5840"

# Acetate experiments
AEXP_RUN_ID = "rhto_20260814_4348"
ALIM_RUN_ID = "rhto_20260814_9995"

# Xylose experiments
XEXP_RUN_ID = "rhto_20260814_4023"
XLIM_RUN_ID = "rhto_20260814_3869"

#-------Experiments using sigma and f factors------


# Project root
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Output path
OUTPUT_PATH = os.path.join(PROJECT_ROOT, "results/figures/r_toruloides_case_study.png")

RESULTS_ROOT = os.path.join(PROJECT_ROOT, "results/tuning_results")

# Stock rhto-GEM (reversible). Xylose DAD/RK reactions are added from the
# xylose config's model_edits. Do not convert to irreversible here.
MODEL_PATH = os.path.join(PROJECT_ROOT, "data/raw/rhto.xml")
XY_CONFIG_PATH = os.path.join(PROJECT_ROOT, "configs/r_toruloides_xy_exp.json")

# Input arguments:
# csv with fluxes and enzyme concentrations
GEXP_PATH = os.path.join(RESULTS_ROOT, GEXP_RUN_ID, "df_FBA.csv")
GLIM_PATH = os.path.join(RESULTS_ROOT, GLIM_RUN_ID, "df_FBA.csv")
AEXP_PATH = os.path.join(RESULTS_ROOT, AEXP_RUN_ID, "df_FBA.csv")
ALIM_PATH = os.path.join(RESULTS_ROOT, ALIM_RUN_ID, "df_FBA.csv")
XEXP_PATH = os.path.join(RESULTS_ROOT, XEXP_RUN_ID, "df_FBA.csv")
XLIM_PATH = os.path.join(RESULTS_ROOT, XLIM_RUN_ID, "df_FBA.csv")



panel_a_reactions = {
    "GDH1": "r_0471", # GDH1 = NADP-dependent glutamate dehydrogenase; 
    "FAS1_palmitate": "r_2140", # FAS = fatty-acyl-CoA synthase (n-C16:0CoA);
    "FAS2_stearate": "r_2141", # FAS = fatty-acyl-CoA synthase (n-C18:0CoA);
    "ACC": "r_0109",  # ACC = acetyl-CoA carboxylase.
}

# Pathway markers for Panel B (flux / substrate uptake).
panel_b_reactions = {
    "Glucose_XPK": "t_0081",
    "Glucose_PTA": "t_0082",
    "Glucose_ACL": "y200003",
    "Acetate_ICL": "r_0662",
    "Acetate_MLS": "r_0716",
    "Acetate_ME": "t_0027",  # cytosolic NADP malic enzyme
    "Xylose_DAD4": "t_0883",  # D-arabinitol 4-DH (D-xylulose → D-arabinitol)
    "Xylose_DAD2": "t_0884",  # DAD-2 / D-ribulose reductase (net: arabinitol → ribulose)
    "Xylose_RK": "t_0885",    # D-ribulokinase (D-ribulose → Ru5P); XK r_1094 is knocked out
}


def load_edited_rhto_model(model_path, config_path):
    """Load stock rhto-GEM and apply xylose-pathway edits from the config."""
    with open(config_path) as fh:
        edits = json.load(fh).get("model_edits")
    model = cobra.io.read_sbml_model(model_path)
    if edits:
        apply_model_edits(model, edits, verbose=False)
    return model

GEXP_SUBS_UPTAKE = 2.489
GLIM_SUBS_UPTAKE = 0.41
XEXP_SUBS_UPTAKE = 6.1
XLIM_SUBS_UPTAKE = 1.97
AEXP_SUBS_UPTAKE = 1.86
ALIM_SUBS_UPTAKE = 0.43

if __name__ == "__main__":
    rhto_model = load_edited_rhto_model(MODEL_PATH, XY_CONFIG_PATH)
    plot_r_toruloides_case_study(
        gexp_path=GEXP_PATH,
        glim_path=GLIM_PATH,
        aexp_path=AEXP_PATH,
        alim_path=ALIM_PATH,
        xexp_path=XEXP_PATH,
        xlim_path=XLIM_PATH,
        panel_a_reactions=panel_a_reactions,
        panel_b_reactions=panel_b_reactions,
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
