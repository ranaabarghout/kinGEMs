# kinGEMs Documentation

This directory contains comprehensive documentation for the kinGEMs v2 pipeline.

## Quick Start Guides

- **[CONFIG_GUIDE.md](CONFIG_GUIDE.md)** - Complete guide to pipeline configuration files
  - How to use `run_pipeline.py` with JSON configs
  - Model type detection (ModelSEED vs standard)
  - Configuration options and parameters
  - Creating configs for new models
  - Troubleshooting common issues

- **[PIPELINE_SUMMARY.md](PIPELINE_SUMMARY.md)** - Overview of the complete pipeline workflow
  - Step-by-step process description
  - Input/output files
  - Key functions and their roles

- **[VENV_SETUP.md](VENV_SETUP.md)** - Python environment configuration
  - Virtual environment setup
  - Package installation
  - Dependency management

## Technical Documentation

### Optimization & Constraints

- **[MIXED_CONSTRAINTS_EXPLAINED.md](MIXED_CONSTRAINTS_EXPLAINED.md)** - Understanding enzyme constraint types
  - AND constraints (enzyme complexes)
  - OR/ISO constraints (isoenzymes)
  - Promiscuous enzyme constraints
  - How constraints are applied

- **[MIXED_CONSTRAINTS_IMPLEMENTATION.md](MIXED_CONSTRAINTS_IMPLEMENTATION.md)** - Implementation details
  - Code structure and functions
  - GPR parsing logic
  - Constraint generation process

- **[CONSTRAINT_SKIPPING_EXPLAINED.md](CONSTRAINT_SKIPPING_EXPLAINED.md)** - Why constraints get skipped
  - Normal skipping behavior for complex models
  - ModelSEED vs standard model differences
  - Enzyme upper bound considerations
  - Troubleshooting constraint issues

### Solvers & Performance

- **[SOLVER_GUIDE.md](SOLVER_GUIDE.md)** - Optimization solver information
  - Gurobi vs GLPK comparison
  - Installation and setup
  - Performance considerations
  - When to use which solver

- **[SCRIPT_CACHING_GUIDE.md](SCRIPT_CACHING_GUIDE.md)** - File caching system
  - How caching works
  - When to use `--force` flag
  - Cache file locations
  - Performance impact

### Analysis & Validation

- **[VALIDATION_PIPELINE_GUIDE.md](VALIDATION_PIPELINE_GUIDE.md)** - Config-driven validation system ⭐ NEW
  - Compare baseline, pre-tuning, and post-tuning models
  - Performance metrics (accuracy, precision, recall, F1, AUC)
  - Essential genes and kcat-specific analysis
  - Automated visualization generation
  - Configuration examples and use cases

- **[FVA_PARALLELIZATION_ANALYSIS.md](FVA_PARALLELIZATION_ANALYSIS.md)** - FVA performance optimization
  - Parallel FVA with Dask and multiprocessing
  - Performance benchmarks and recommendations
  - Memory usage estimates
  - Configuration options

- **[MODEL_COMPARISON.md](MODEL_COMPARISON.md)** - Comparing different models
  - Standard vs ModelSEED models
  - Model complexity metrics
  - Expected behavior differences

- **[SYSTEM_SUMMARY.md](SYSTEM_SUMMARY.md)** - Complete technical reference
  - Detailed implementation notes
  - System architecture
  - File organization
  - Development guidelines

## Quick Reference

### Running the Pipeline
```bash
# Activate environment
source venv/bin/activate

# Run a model
python scripts/run_pipeline.py configs/iML1515_GEM.json

# Force regenerate all files
python scripts/run_pipeline.py configs/iML1515_GEM.json --force
```

### Key Concepts

- **Enzyme Constraints**: Limit metabolic flux based on enzyme availability
- **Simulated Annealing**: Optimize kcat values to improve model predictions
- **File Caching**: Reuse intermediate files to speed up subsequent runs
- **Model Types**: Standard (BiGG) vs ModelSEED models

## Document Index by Topic

### For New Users
1. Start with [CONFIG_GUIDE.md](CONFIG_GUIDE.md) to learn how to run the pipeline
2. Read [PIPELINE_SUMMARY.md](PIPELINE_SUMMARY.md) to understand the workflow
3. Set up your environment with [VENV_SETUP.md](VENV_SETUP.md)

### For Understanding Results
1. [MIXED_CONSTRAINTS_EXPLAINED.md](MIXED_CONSTRAINTS_EXPLAINED.md) - What constraints are being applied
2. [CONSTRAINT_SKIPPING_EXPLAINED.md](CONSTRAINT_SKIPPING_EXPLAINED.md) - Why some constraints are skipped
3. [MODEL_COMPARISON.md](MODEL_COMPARISON.md) - How different models behave

### For Troubleshooting
1. [CONFIG_GUIDE.md](CONFIG_GUIDE.md) - Common configuration errors
2. [CONSTRAINT_SKIPPING_EXPLAINED.md](CONSTRAINT_SKIPPING_EXPLAINED.md) - Enzyme budget issues
3. [SOLVER_GUIDE.md](SOLVER_GUIDE.md) - Solver-related problems
4. [SCRIPT_CACHING_GUIDE.md](SCRIPT_CACHING_GUIDE.md) - When to regenerate cached files

### For Advanced Users
1. [MIXED_CONSTRAINTS_IMPLEMENTATION.md](MIXED_CONSTRAINTS_IMPLEMENTATION.md) - Code internals
2. [SYSTEM_SUMMARY.md](SYSTEM_SUMMARY.md) - Complete technical reference
3. [MODEL_COMPARISON.md](MODEL_COMPARISON.md) - Model-specific considerations

## Common Questions

**Q: How do I run the pipeline for my model?**
A: See [CONFIG_GUIDE.md](CONFIG_GUIDE.md) - Quick Start section

**Q: Why is my simulated annealing showing 0% improvement?**
A: See [CONSTRAINT_SKIPPING_EXPLAINED.md](CONSTRAINT_SKIPPING_EXPLAINED.md) - The Real Problem: Enzyme Upper Bound

**Q: What's the difference between ModelSEED and standard models?**
A: See [MODEL_COMPARISON.md](MODEL_COMPARISON.md) and [CONFIG_GUIDE.md](CONFIG_GUIDE.md) - Model Type Detection

**Q: Should I use Gurobi or GLPK?**
A: See [SOLVER_GUIDE.md](SOLVER_GUIDE.md)

**Q: When should I use the `--force` flag?**
A: See [SCRIPT_CACHING_GUIDE.md](SCRIPT_CACHING_GUIDE.md)

**Q: What do the constraint statistics mean?**
A: See [MIXED_CONSTRAINTS_EXPLAINED.md](MIXED_CONSTRAINTS_EXPLAINED.md)

## Documentation Index by Audience

| Document | Purpose | Audience |
|----------|---------|----------|
| CONFIG_GUIDE.md | Pipeline configuration | All users |
| VENV_SETUP.md | Environment setup | New users |
| PIPELINE_SUMMARY.md | Workflow overview | All users |
| SYSTEM_SUMMARY.md | Complete technical reference | All users |
| SCRIPT_CACHING_GUIDE.md | Caching feature | All users |
| SOLVER_GUIDE.md | Solver configuration | Users with solver issues |
| MODEL_COMPARISON.md | Model differences | Researchers |
| MIXED_CONSTRAINTS_EXPLAINED.md | Constraint theory | Developers/Researchers |
| MIXED_CONSTRAINTS_IMPLEMENTATION.md | Technical implementation | Developers |
| CONSTRAINT_SKIPPING_EXPLAINED.md | Troubleshooting constraints | All users |

## Additional Resources

- Main README: `../README.md`
- Results Documentation: `../results/README.md`

## Contributing

When adding new documentation:
1. Create the `.md` file in this directory
2. Update this README with a link and description
3. Follow the existing documentation style
4. Add to the appropriate topic section

## Documentation Maintenance

These documents are maintained alongside the kinGEMs v2 codebase. When updating the pipeline code, please update relevant documentation to keep it synchronized.
