#!/usr/bin/env python3
"""
Check available COBRA solvers and set the best one.
"""

import cobra

print("Checking available solvers...")
print("=" * 60)

# Get model to test
try:
    model = cobra.io.read_sbml_model("models/ecoli_iML1515_20250826_4941.xml")
except:
    model = cobra.Model("test")
    model.add_reactions([cobra.Reaction("test")])

# Check current solver
print(f"\nCurrent solver: {model.solver.interface.__name__}")

# List all available solvers
print("\nAvailable solvers:")
available = []
for solver_name in ['cplex', 'gurobi', 'glpk', 'scip']:
    try:
        test_model = model.copy()
        test_model.solver = solver_name
        available.append(solver_name)
        print(f"  ✓ {solver_name.upper()}")
    except Exception as e:
        print(f"  ✗ {solver_name.upper()} - not available")

print("\n" + "=" * 60)

if 'cplex' in available:
    print("\n✅ CPLEX is available (BEST CHOICE)")
    print("   Add this to your script:")
    print("   model.solver = 'cplex'")
elif 'gurobi' in available:
    print("\n✅ Gurobi is available (EXCELLENT CHOICE)")
    print("   Add this to your script:")
    print("   model.solver = 'gurobi'")
elif 'scip' in available:
    print("\n✅ SCIP is available (GOOD CHOICE)")
    print("   Add this to your script:")
    print("   model.solver = 'scip'")
else:
    print("\n⚠️  Only GLPK available (SLOW)")
    print("   Consider installing a better solver:")
    print("   - Load CPLEX module: module load cplex")
    print("   - Or install SCIP: pip install pyscipopt")

print("\n" + "=" * 60)
