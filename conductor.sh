#!/bin/bash

####################
### 1. INITIAL DEFINITIONS
# Don't need to change, defined here for reducing typo errors later

# Solver script name (no longer used in this version, but keeping for reference)
MATLAB_SCRIPT="iam_solver.m" 

# Behavior types
launch_pattern_type_equilibrium="equilibrium"
launch_pattern_type_feedback="sat_feedback"

####################
### 2. SCENARIO DEFINITIONS
# Change to set scenarios and parallelization

# Choose propagator
model_type="MOCAT" # Either "MOCAT" or "GMPHD"

# Define the scenario to run. Store them in an array. Should be valid names of parameter set CSV files. 
## See examples in scenarios/parsets and compare to files named --parameters.csv for how to create new ones.
scenario_files=(
                "benchmark"
                "scenarios/parsets/tax_1.csv"
                "scenarios/parsets/tax_2.csv"
                )

# Define the length of the horizon over which to solve (years)
model_horizon=20

# Set number of workers for parallelization
n_workers=10

####################
### 3.  EXECUTION
# For each scenario and launch behavior: 1. Run the Python iam_solver script; 2. Loop over scenarios with the scenario_file inputs; 3. Run comparisons.

# Step 3.1: Run simulations

# Set PYTHONPATH to include the directory where OPUS is located
export PYTHONPATH="$PYTHONPATH:/c/Users/IT/Documents/UCL/OPUS"

# Initialize empty array to store unique names for step 3.2
unique_name_array=()

# Main simulation loop
for scenario_file in "${scenario_files[@]}"
do
    # Call the scenarioNamer function with the parameter file as an argument
    python3 -c "from OPUS.utils.main import scenario_namer; scenario_namer('$scenario_file', '$model_type')"

    # Read the unique name from the scenario_name.txt file
    unique_name=$(cat scenario_name.txt)
    unique_name_array+=("$unique_name")

    # Launch behavior 1: Equilibrium, MOCAT or GMPHD
    python3 -c "from OPUS.utils.main import iam_solver; iam_solver('$unique_name', '$model_type', '$launch_pattern_type_equilibrium', '$scenario_file', $model_horizon, $n_workers)"

    # Launch behavior 2: Feedback, MOCAT only
    python3 -c "from OPUS.utils.main import iam_solver; iam_solver('$unique_name', 'MOCAT', '$launch_pattern_type_feedback', '$scenario_file', $model_horizon, $n_workers)"

    # Tell them you're done buddy!
    echo "Scenario $scenario_file complete!"
done
