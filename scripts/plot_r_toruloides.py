"""Plot figures for the case study for R toruloides."""

import json
import os

import cobra

from kinGEMs.modeling.tuning import apply_model_edits
from kinGEMs.plots import plot_r_toruloides_case_study


# Tuning runs with fixed bounds + other substrates blocked.
# Glucose experiments
GEXP_RUN_ID = "rhto_20260815_5191"
GLIM_RUN_ID = "rhto_20260815_4765"

# Acetate experiments
AEXP_RUN_ID = "rhto_20260815_7806"
ALIM_RUN_ID = "rhto_20260815_5440"

# Xylose experiments
XEXP_RUN_ID = "rhto_20260815_1121"
XLIM_RUN_ID = "rhto_20260815_5917"  # with maintenance sweep


# Project root
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

OUTPUT_PATH = os.path.join(
    PROJECT_ROOT, "results/figures/r_toruloides_case_study.png"
)

RESULTS_ROOT = os.path.join(PROJECT_ROOT, "results/tuning_results")
CONFIGS_ROOT = os.path.join(PROJECT_ROOT, "configs")

# Stock rhto-GEM (reversible). Xylose DAD/RK reactions are added from the
# xylose config's model_edits. Do not convert to irreversible here.
MODEL_PATH = os.path.join(PROJECT_ROOT, "data/raw/rhto.xml")
XY_CONFIG_PATH = os.path.join(CONFIGS_ROOT, "r_toruloides_xy_exp.json")

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


def load_edited_rhto_model(model_path, config_path):
    """Load stock rhto-GEM and apply xylose-pathway edits from the config."""
    with open(config_path) as fh:
        edits = json.load(fh).get("model_edits")
    model = cobra.io.read_sbml_model(model_path)
    if edits:
        apply_model_edits(model, edits, verbose=False)
    return model


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
        model=rhto_model,
        output_path=OUTPUT_PATH,
        show=False,
    )
