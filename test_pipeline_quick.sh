#!/bin/bash
# Quick test of pipeline with reduced iterations

# Backup original config
cp configs/iML1515_GEM.json configs/iML1515_GEM.json.backup

# Create test config with fewer iterations
cat > configs/iML1515_GEM_test.json << 'EOF'
{
  "model_name": "iML1515_GEM",
  "organism": "E coli",
  "biomass_reaction": "BIOMASS_Ec_iML1515_core_75p37M",
  "enzyme_upper_bound": 0.35,
  "solver": "glpk",
  "enable_fva": false,
  "enable_biolog_validation": false,
  "simulated_annealing": {
    "temperature": 1.0,
    "cooling_rate": 0.95,
    "min_temperature": 0.01,
    "max_iterations": 5,
    "max_unchanged_iterations": 10,
    "change_threshold": 0.001,
    "biomass_goal": 0.5,
    "verbose": false
  }
}
EOF

echo "Running pipeline with 5 iterations..."
source venv/bin/activate
timeout 300 python scripts/run_pipeline.py configs/iML1515_GEM_test.json 2>&1 | tail -100

# Cleanup
rm configs/iML1515_GEM_test.json
