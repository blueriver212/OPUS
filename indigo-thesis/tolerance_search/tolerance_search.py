#!/usr/bin/env python3
"""
Tolerance Search Script for Multi-Species Solver Optimization

This script tests different tolerance levels for the solver to find the optimal
balance between accuracy and computational speed. It runs multiple simulations
in parallel and creates a Pareto front visualization.

Target values: S=7677, Su=2665, Sns=1228
"""

import sys
import os
import json
import numpy as np
import time
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
from pathlib import Path

# Add the OPUS directory to the path
sys.path.append('/Users/indigobrownhall/Code/OPUS/OPUS')

from utils.MultiSpecies import MultiSpecies
from utils.MocatParameters import configure_mocat
from utils.MultiSpeciesOpenAccessSolver import MultiSpeciesOpenAccessSolver
from utils.Helpers import insert_launches_into_lam
from utils.PostMissionDisposal import evaluate_pmd, evaluate_pmd_elliptical

@dataclass
class ToleranceConfig:
    """Configuration for solver tolerance parameters"""
    ftol: float
    xtol: float
    gtol: float
    max_nfev: int
    name: str

@dataclass
class SimulationResult:
    """Results from a single simulation run"""
    config: ToleranceConfig
    final_counts: Dict[str, float]
    accuracy: float
    computation_time: float
    success: bool
    error_message: Optional[str] = None

class ToleranceSearch:
    """Main class for tolerance search optimization"""
    
    def __init__(self, config_path: str, target_values: np.ndarray):
        """
        Initialize the tolerance search
        
        Args:
            config_path: Path to MOCAT configuration JSON
            target_values: Target final counts [S, Su, Sns]
        """
        self.config_path = config_path
        self.target_values = target_values
        self.species_names = ["S", "Su", "Sns"]
        
        # Load MOCAT configuration
        with open(config_path, 'r') as f:
            self.MOCAT_config = json.load(f)
        
        # Define tolerance configurations to test
        self.tolerance_configs = [
            # Very loose tolerances (fast but potentially inaccurate)
            ToleranceConfig(ftol=1e-3, xtol=0.1, gtol=1e-3, max_nfev=100, name="loose"),
            ToleranceConfig(ftol=5e-4, xtol=0.05, gtol=5e-4, max_nfev=200, name="loose_med"),
            
            # Medium tolerances
            ToleranceConfig(ftol=1e-4, xtol=0.01, gtol=1e-4, max_nfev=500, name="medium"),
            ToleranceConfig(ftol=5e-5, xtol=0.005, gtol=5e-5, max_nfev=750, name="medium_tight"),
            
            # Tight tolerances
            ToleranceConfig(ftol=1e-5, xtol=0.001, gtol=1e-5, max_nfev=1000, name="tight"),
            ToleranceConfig(ftol=5e-6, xtol=0.0005, gtol=5e-6, max_nfev=1500, name="tight_very"),
            
            # Very tight tolerances (slow but accurate)
            ToleranceConfig(ftol=1e-6, xtol=0.0001, gtol=1e-6, max_nfev=2000, name="very_tight"),
            ToleranceConfig(ftol=5e-7, xtol=0.00005, gtol=5e-7, max_nfev=3000, name="very_tight_2"),
            
            # Current settings (commented out in original code)
            ToleranceConfig(ftol=1e-8, xtol=0.005, gtol=1e-8, max_nfev=1000, name="current"),
        ]
        
        # Results storage
        self.results: List[SimulationResult] = []
    
    def run_single_simulation(self, config: ToleranceConfig) -> SimulationResult:
        """
        Run a single simulation with given tolerance configuration
        
        Args:
            config: Tolerance configuration to test
            
        Returns:
            SimulationResult with results and metrics
        """
        start_time = time.time()
        
        try:
            # Initialize multi-species
            multi_species = MultiSpecies(self.species_names)
            
            # Configure MOCAT
            MOCAT, multi_species = configure_mocat(
                self.MOCAT_config, 
                multi_species=multi_species, 
                grid_search=True
            )
            
            # Get species position indexes and parameters
            multi_species.get_species_position_indexes(MOCAT)
            multi_species.get_mocat_species_parameters(MOCAT)
            
            # Get initial state
            x0 = MOCAT.scenario_properties.x0
            elliptical = MOCAT.scenario_properties.elliptical
            
            # Flatten for circular orbits
            if not elliptical:
                x0 = x0.T.values.flatten()
            
            # Create solver guess (5% of current population)
            solver_guess = x0.copy()
            lam = np.full_like(x0, None, dtype=object)
            
            if elliptical:
                for species in multi_species.species:
                    initial_guess = 0.05 * x0[:, species.species_idx, 0]
                    initial_guess = np.maximum(initial_guess, 0.0)
                    if np.sum(initial_guess) == 0:
                        initial_guess[:] = 5
                    lam[:, species.species_idx, 0] = initial_guess
                    solver_guess[:, species.species_idx, 0] = initial_guess
            else:
                for species in multi_species.species:
                    initial_guess = 0.05 * np.array(x0[species.start_slice:species.end_slice])
                    if np.sum(initial_guess) == 0:
                        initial_guess[:] = 5
                    solver_guess[species.start_slice:species.end_slice] = initial_guess
                    lam[species.start_slice:species.end_slice] = initial_guess
            
            # Run simulation for one time step to get final counts
            years = [2017 + i for i in range(MOCAT.scenario_properties.simulation_duration)]
            
            # Create solver with custom tolerance options
            custom_solver_options = {
                'method': 'trf',
                'verbose': 0,
                'ftol': config.ftol,
                'xtol': config.xtol,
                'gtol': config.gtol,
                'max_nfev': config.max_nfev
            }
            
            open_access = MultiSpeciesOpenAccessSolver(
                MOCAT, solver_guess, x0, "linear", lam, multi_species, years, 0, custom_solver_options
            )
            
            # Run solver
            launch_rate, col_probability_all_species, umpy, excess_returns, last_non_compliance = open_access.solver()
            
            # Calculate final species counts
            final_counts = {}
            for i, species_name in enumerate(self.species_names):
                if elliptical:
                    # For elliptical orbits, sum across all shells and eccentricity bins
                    species_data = x0[:, i, :]
                    final_counts[species_name] = np.sum(species_data)
                else:
                    # For circular orbits, sum across shells for this species
                    species_slice = slice(i * MOCAT.scenario_properties.n_shells, 
                                        (i + 1) * MOCAT.scenario_properties.n_shells)
                    final_counts[species_name] = np.sum(x0[species_slice])
            
            # Calculate accuracy (mean absolute percentage error)
            predicted_values = np.array([final_counts[name] for name in self.species_names])
            accuracy = self.calculate_accuracy(predicted_values, self.target_values)
            
            computation_time = time.time() - start_time
            
            return SimulationResult(
                config=config,
                final_counts=final_counts,
                accuracy=accuracy,
                computation_time=computation_time,
                success=True
            )
            
        except Exception as e:
            computation_time = time.time() - start_time
            return SimulationResult(
                config=config,
                final_counts={},
                accuracy=float('inf'),
                computation_time=computation_time,
                success=False,
                error_message=str(e)
            )
    
    def calculate_accuracy(self, predicted: np.ndarray, target: np.ndarray) -> float:
        """
        Calculate accuracy as mean absolute percentage error (MAPE)
        
        Args:
            predicted: Predicted values
            target: Target values
            
        Returns:
            MAPE as a percentage (lower is better)
        """
        # Avoid division by zero
        target_safe = np.where(target == 0, 1e-10, target)
        mape = np.mean(np.abs((predicted - target) / target_safe)) * 100
        return mape
    
    def run_parallel_search(self, n_runs: int = 10) -> List[SimulationResult]:
        """
        Run tolerance search with parallel processing
        
        Args:
            n_runs: Number of runs per tolerance configuration
            
        Returns:
            List of all simulation results
        """
        print(f"Starting tolerance search with {n_runs} runs per configuration...")
        print(f"Testing {len(self.tolerance_configs)} tolerance configurations...")
        
        # Create all combinations of configs and runs
        all_tasks = []
        for config in self.tolerance_configs:
            for run_id in range(n_runs):
                all_tasks.append((config, run_id))
        
        print(f"Total tasks: {len(all_tasks)}")
        
        # Run in parallel
        results = []
        with ProcessPoolExecutor(max_workers=mp.cpu_count()) as executor:
            # Submit all tasks
            future_to_task = {
                executor.submit(self.run_single_simulation, config): (config, run_id)
                for config, run_id in all_tasks
            }
            
            # Collect results as they complete
            completed = 0
            for future in as_completed(future_to_task):
                config, run_id = future_to_task[future]
                try:
                    result = future.result()
                    results.append(result)
                    completed += 1
                    
                    if result.success:
                        print(f"✓ {config.name} (run {run_id}): {result.computation_time:.2f}s, "
                              f"accuracy: {result.accuracy:.2f}%")
                    else:
                        print(f"✗ {config.name} (run {run_id}): FAILED - {result.error_message}")
                    
                except Exception as e:
                    print(f"✗ {config.name} (run {run_id}): EXCEPTION - {str(e)}")
                    completed += 1
                
                if completed % 10 == 0:
                    print(f"Progress: {completed}/{len(all_tasks)} tasks completed")
        
        self.results = results
        return results
    
    def analyze_results(self) -> Dict:
        """
        Analyze results and create summary statistics
        
        Returns:
            Dictionary with analysis results
        """
        if not self.results:
            raise ValueError("No results to analyze. Run search first.")
        
        # Group results by configuration
        config_results = {}
        for result in self.results:
            if result.success:
                config_name = result.config.name
                if config_name not in config_results:
                    config_results[config_name] = []
                config_results[config_name].append(result)
        
        # Calculate statistics for each configuration
        analysis = {}
        for config_name, results in config_results.items():
            if not results:
                continue
                
            accuracies = [r.accuracy for r in results]
            times = [r.computation_time for r in results]
            
            analysis[config_name] = {
                'mean_accuracy': np.mean(accuracies),
                'std_accuracy': np.std(accuracies),
                'mean_time': np.mean(times),
                'std_time': np.std(times),
                'min_accuracy': np.min(accuracies),
                'max_accuracy': np.max(accuracies),
                'min_time': np.min(times),
                'max_time': np.max(times),
                'n_successful': len(results),
                'config': results[0].config
            }
        
        return analysis
    
    def create_pareto_plot(self, save_path: str = None):
        """
        Create Pareto front visualization
        
        Args:
            save_path: Path to save the plot (optional)
        """
        if not self.results:
            raise ValueError("No results to plot. Run search first.")
        
        # Filter successful results
        successful_results = [r for r in self.results if r.success]
        
        if not successful_results:
            print("No successful results to plot!")
            return
        
        # Group by configuration and calculate means
        config_means = {}
        for result in successful_results:
            config_name = result.config.name
            if config_name not in config_means:
                config_means[config_name] = {'accuracies': [], 'times': []}
            config_means[config_name]['accuracies'].append(result.accuracy)
            config_means[config_name]['times'].append(result.computation_time)
        
        # Calculate mean values for each configuration
        config_data = {}
        for config_name, data in config_means.items():
            config_data[config_name] = {
                'mean_accuracy': np.mean(data['accuracies']),
                'mean_time': np.mean(data['times']),
                'std_accuracy': np.std(data['accuracies']),
                'std_time': np.std(data['times'])
            }
        
        # Create the plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot 1: Accuracy vs Time (Pareto front)
        config_names = list(config_data.keys())
        mean_accuracies = [config_data[name]['mean_accuracy'] for name in config_names]
        mean_times = [config_data[name]['mean_time'] for name in config_names]
        std_accuracies = [config_data[name]['std_accuracy'] for name in config_names]
        std_times = [config_data[name]['std_time'] for name in config_names]
        
        # Scatter plot with error bars
        ax1.errorbar(mean_times, mean_accuracies, 
                    xerr=std_times, yerr=std_accuracies,
                    fmt='o', capsize=5, capthick=2, markersize=8)
        
        # Add labels for each point
        for i, name in enumerate(config_names):
            ax1.annotate(name, (mean_times[i], mean_accuracies[i]), 
                        xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        ax1.set_xlabel('Computation Time (seconds)')
        ax1.set_ylabel('Accuracy Error (%)')
        ax1.set_title('Pareto Front: Accuracy vs Computation Time')
        ax1.grid(True, alpha=0.3)
        ax1.invert_yaxis()  # Lower accuracy error is better
        
        # Plot 2: Tolerance parameters comparison
        tolerance_params = ['ftol', 'xtol', 'gtol', 'max_nfev']
        x_pos = np.arange(len(tolerance_params))
        
        # Create a subplot for each configuration
        for i, config_name in enumerate(config_names):
            config = config_data[config_name]
            config_obj = next(r.config for r in successful_results if r.config.name == config_name)
            
            params = [config_obj.ftol, config_obj.xtol, config_obj.gtol, config_obj.max_nfev]
            
            # Normalize parameters for comparison (log scale for tolerances)
            normalized_params = [
                np.log10(config_obj.ftol),
                np.log10(config_obj.xtol), 
                np.log10(config_obj.gtol),
                config_obj.max_nfev / 1000  # Scale max_nfev
            ]
            
            ax2.plot(x_pos, normalized_params, 'o-', label=config_name, linewidth=2, markersize=6)
        
        ax2.set_xlabel('Tolerance Parameters')
        ax2.set_ylabel('Normalized Parameter Values')
        ax2.set_title('Tolerance Parameter Comparison')
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels(['ftol (log10)', 'xtol (log10)', 'gtol (log10)', 'max_nfev/1000'])
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to: {save_path}")
        
        plt.show()
        
        # Print summary table
        print("\n" + "="*80)
        print("TOLERANCE SEARCH RESULTS SUMMARY")
        print("="*80)
        print(f"{'Config':<15} {'Mean Acc (%)':<12} {'Mean Time (s)':<12} {'Std Acc (%)':<12} {'Std Time (s)':<12}")
        print("-"*80)
        
        for config_name in sorted(config_names, key=lambda x: config_data[x]['mean_accuracy']):
            data = config_data[config_name]
            print(f"{config_name:<15} {data['mean_accuracy']:<12.2f} {data['mean_time']:<12.2f} "
                  f"{data['std_accuracy']:<12.2f} {data['std_time']:<12.2f}")
        
        # Find Pareto optimal solutions
        print("\n" + "="*50)
        print("PARETO OPTIMAL SOLUTIONS")
        print("="*50)
        
        # Sort by accuracy (ascending) and time (ascending)
        sorted_configs = sorted(config_names, key=lambda x: (config_data[x]['mean_accuracy'], config_data[x]['mean_time']))
        
        pareto_optimal = []
        for config_name in sorted_configs:
            is_pareto = True
            current_acc = config_data[config_name]['mean_accuracy']
            current_time = config_data[config_name]['mean_time']
            
            for other_config in sorted_configs:
                if other_config == config_name:
                    continue
                other_acc = config_data[other_config]['mean_accuracy']
                other_time = config_data[other_config]['mean_time']
                
                # Check if other config dominates this one
                if (other_acc <= current_acc and other_time <= current_time and 
                    (other_acc < current_acc or other_time < current_time)):
                    is_pareto = False
                    break
            
            if is_pareto:
                pareto_optimal.append(config_name)
        
        for config_name in pareto_optimal:
            data = config_data[config_name]
            print(f"✓ {config_name}: {data['mean_accuracy']:.2f}% error, {data['mean_time']:.2f}s")
        
        return config_data

def main():
    """Main function to run the tolerance search"""
    # Configuration
    config_path = "/Users/indigobrownhall/Code/OPUS/OPUS/configuration/multi_single_species.json"
    target_values = np.array([7677, 2665, 1228])  # S, Su, Sns
    
    # Create tolerance search instance
    search = ToleranceSearch(config_path, target_values)
    
    print("Starting Tolerance Search Optimization")
    print("="*50)
    print(f"Target values: S={target_values[0]}, Su={target_values[1]}, Sns={target_values[2]}")
    print(f"Testing {len(search.tolerance_configs)} tolerance configurations")
    print(f"Running {10} simulations per configuration")
    print(f"Using {mp.cpu_count()} CPU cores")
    print("="*50)
    
    # Run the search
    results = search.run_parallel_search(n_runs=10)
    
    # Analyze results
    analysis = search.analyze_results()
    
    # Create Pareto plot
    plot_path = "/Users/indigobrownhall/Code/OPUS/indigo-thesis/tolerance_search/pareto_front.png"
    search.create_pareto_plot(save_path=plot_path)
    
    # Save results to file
    results_path = "/Users/indigobrownhall/Code/OPUS/indigo-thesis/tolerance_search/results.json"
    
    # Convert results to serializable format
    serializable_results = []
    for result in results:
        serializable_results.append({
            'config_name': result.config.name,
            'config': {
                'ftol': result.config.ftol,
                'xtol': result.config.xtol,
                'gtol': result.config.gtol,
                'max_nfev': result.config.max_nfev
            },
            'final_counts': result.final_counts,
            'accuracy': result.accuracy,
            'computation_time': result.computation_time,
            'success': result.success,
            'error_message': result.error_message
        })
    
    with open(results_path, 'w') as f:
        json.dump(serializable_results, f, indent=2)
    
    print(f"\nResults saved to: {results_path}")
    print(f"Plot saved to: {plot_path}")
    print("\nTolerance search completed!")

if __name__ == "__main__":
    main()
