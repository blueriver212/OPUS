# Tolerance Search for Multi-Species Solver Optimization

This directory contains scripts to optimize the tolerance parameters for the multi-species solver to find the optimal balance between accuracy and computational speed.

## Target Values
- S (Satellites): 7677
- Su (Unbonded Satellites): 2665  
- Sns (Non-satellites): 1228

## Files

- `tolerance_search.py` - Main script for running the tolerance search with parallel processing
- `test_tolerance_search.py` - Test script to verify the setup works correctly
- `requirements.txt` - Python dependencies
- `README.md` - This file

## Usage

### 1. Test the Setup
First, run the test script to make sure everything is working:

```bash
cd /Users/indigobrownhall/Code/OPUS/indigo-thesis/tolerance_search
python test_tolerance_search.py
```

### 2. Run the Full Tolerance Search
Once the test passes, run the full search:

```bash
python tolerance_search.py
```

This will:
- Test 9 different tolerance configurations
- Run 10 simulations per configuration (90 total runs)
- Use all available CPU cores for parallel processing
- Generate a Pareto front plot showing accuracy vs computation time
- Save results to `results.json` and `pareto_front.png`

## Tolerance Configurations Tested

The script tests the following tolerance configurations:

1. **loose**: ftol=1e-3, xtol=0.1, gtol=1e-3, max_nfev=100
2. **loose_med**: ftol=5e-4, xtol=0.05, gtol=5e-4, max_nfev=200
3. **medium**: ftol=1e-4, xtol=0.01, gtol=1e-4, max_nfev=500
4. **medium_tight**: ftol=5e-5, xtol=0.005, gtol=5e-5, max_nfev=750
5. **tight**: ftol=1e-5, xtol=0.001, gtol=1e-5, max_nfev=1000
6. **tight_very**: ftol=5e-6, xtol=0.0005, gtol=5e-6, max_nfev=1500
7. **very_tight**: ftol=1e-6, xtol=0.0001, gtol=1e-6, max_nfev=2000
8. **very_tight_2**: ftol=5e-7, xtol=0.00005, gtol=5e-7, max_nfev=3000
9. **current**: ftol=1e-8, xtol=0.005, gtol=1e-8, max_nfev=1000

## Output

The script generates:

1. **Console output**: Progress updates and summary statistics
2. **results.json**: Detailed results for all runs
3. **pareto_front.png**: Visualization showing the Pareto front of accuracy vs computation time

## Interpretation

- **Lower accuracy error (%)** = better accuracy
- **Lower computation time (seconds)** = faster execution
- **Pareto optimal solutions** are those that cannot be improved in one dimension without worsening the other

The script will identify Pareto optimal solutions that provide the best trade-off between accuracy and speed.

## Customization

To modify the search:

1. **Change tolerance configurations**: Edit the `tolerance_configs` list in `tolerance_search.py`
2. **Change number of runs**: Modify the `n_runs` parameter in `run_parallel_search()`
3. **Change target values**: Update the `target_values` array in the `main()` function
4. **Add new tolerance parameters**: Extend the `ToleranceConfig` dataclass

## Dependencies

Install required packages:

```bash
pip install -r requirements.txt
```

The script also requires the OPUS codebase to be available in the Python path.

