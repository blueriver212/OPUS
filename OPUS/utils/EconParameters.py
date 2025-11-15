import numpy as np
from pyssem.model import Model
from pyssem.utils.drag.drag import densityexp
import pandas as pd
import os

class EconParameters:
    """
    Class to build satellite cost using physcial and economic parameters
    Uses the disposal time regulation in the EconParams to calculate the highest compliant altitude. 
    Then uses the delta-v required for Hohmann transfer to contruct the cost function.

    It is initialized with default values for the parameters. 
    Then the calculate_cost_fn_parameters function is called to calculate the cost function parameters.
    """
    def __init__(self, econ_params_json, mocat: Model):

        # Save MOCAT
        self.mocat = mocat
        self.maneuverable = False

        # Updated logic from EconParameters1.py for safety
        if econ_params_json is None:
            params = {}
        else:
            params = econ_params_json.get("OPUS", econ_params_json)

        # Lifetime of a satellite [years]
        # Default value is 5 years for MOCAT
        self.sat_lifetime = params.get("sat_lifetime", 5)

        # Regulated disposal rule time [years], default is 5
        self.disposal_time = params.get("disposal_time", 5)

        # # Discount rate [pp/year]
        self.discount_rate = params.get("discount_rate", 0.05)

        # # A satellite facing no competition earns $750,000 per year in revenues
        # [$/year]
        self.intercept = params.get("intercept", 7.5e5)

        # [$/satellite/year]
        self.coef = params.get("coef", 1.0e2)

        # [%] tax rate on shell-specific pc
        self.tax = params.get("tax", 0.0)

        # Cost for a single satellite to use any shell [$]
        # # Cost of delta-v [$/km/s]
        self.base_delta_v_cost = params.get("delta_v_cost", 1000)

        #Toggle the congestion switch to turn on congestion pricing for delta_v
        self.congestion_switch = 0 
        self.delta_v_cost = np.full(self.mocat.scenario_properties.n_shells, self.base_delta_v_cost)
        
        #Competitors for bonded and unbonded species
        self.competitors = params.get("competitors", [])

        # # Price of lift [$/kg]
        # # Default is $5000/kg based on price index calculations
        self.lift_price = params.get("lift_price", 5000)
        
        # regulatory non-compliance
        self.prob_of_non_compliance = params.get("prob_of_non_compliance", 0)

        # Bond amount
        self.bond = params.get("bond", None)

        # Demand growhth, annual rate
        self.demand_growth = params.get("demand_growth", None)

        self.mass = params.get("mass", None)

        # PMD options
        self.controlled_pmd = params.get("controlled_pmd", 0)
        self.uncontrolled_pmd = params.get("uncontrolled_pmd", 0)
        self.no_attempt_pmd = params.get("no_attempt_pmd", 0)
        self.failed_attempt_pmd = params.get("failed_attempt_pmd", 0)

    def calculate_cost_fn_parameters(self, pmd_rate, scenario_name):

        # Save the pmd_rate as this will be passed from MOCAT
        self.pmd_rate = pmd_rate

        shell_marginal_decay_rates = np.zeros(self.mocat.scenario_properties.n_shells)
        shell_marginal_residence_times = np.zeros(self.mocat.scenario_properties.n_shells)
        self.shell_cumulative_residence_times = np.zeros(self.mocat.scenario_properties.n_shells)

        # Here using the ballastic coefficient of the species, we are trying to find the highest compliant altitude/shell
        for k in range(self.mocat.scenario_properties.n_shells):
            #rhok = density_jb2008(self.mocat.scenario_properties.R0_km[k], solar_activity='medium')
            rhok = densityexp(self.mocat.scenario_properties.R0_km[k])
            # satellite 
            beta = 0.0172 # ballastic coefficient, area * mass * drag coefficient. This should be done for each species!
            rvel_current_D = -rhok * beta * np.sqrt(self.mocat.scenario_properties.mu * self.mocat.scenario_properties.R0[k]) * (24 * 3600 * 365.25)
            shell_marginal_decay_rates[k] = -rvel_current_D/self.mocat.scenario_properties.Dhl
            shell_marginal_residence_times[k] = 1/shell_marginal_decay_rates[k]
        
        self.shell_cumulative_residence_times = np.cumsum(shell_marginal_residence_times)

        # Find the index of shell_cumulative_residence_times, k_star, which is the largest index that  shell_cumulative_residence_times(k_star) <= self.disposalTime
        indices = np.where(self.shell_cumulative_residence_times <= self.disposal_time)[0]
        k_star = max(indices) if len(indices) > 0 else 0

        # Physics based cost function using delta-v and deorbit requirements
        v_drag = np.zeros(self.mocat.scenario_properties.n_shells)
        delta_t = 24 * 3600 * 365.25 # time inteval in seconds

        # Calculate drag delta-v for stationkeeping maneuvers
        for k in range(self.mocat.scenario_properties.n_shells):
            rhok = densityexp(self.mocat.scenario_properties.R0_km[k])
            orbital_velocity = np.sqrt(self.mocat.scenario_properties.mu / self.mocat.scenario_properties.R0[k])
            F_drag = 2.2 * 0.5 * rhok * orbital_velocity**2 * 1.741 # this is cd and area and have been hard coded
            v_drag[k] = F_drag / self.mass * delta_t * 1e-3

        # Calculate the delta-v for deorbiting
        original_orbit_delta_v = np.zeros(self.mocat.scenario_properties.n_shells)
        target_orbit_delta_v = np.zeros(self.mocat.scenario_properties.n_shells)
        r2 = self.mocat.scenario_properties.R0[k_star]

        for k in range(self.mocat.scenario_properties.n_shells):
            original_orbit_delta_v[k] = np.sqrt(self.mocat.scenario_properties.mu / self.mocat.scenario_properties.R0[k]) * (1 - np.sqrt(2 * r2 / (self.mocat.scenario_properties.R0[k] + r2)))
            target_orbit_delta_v[k] = np.sqrt(self.mocat.scenario_properties.mu / r2) * (np.sqrt(2 * self.mocat.scenario_properties.R0[k] / (self.mocat.scenario_properties.R0[k] + r2)) - 1)

        original_orbit_delta_v = np.maximum(0, original_orbit_delta_v)
        target_orbit_delta_v = np.maximum(0, target_orbit_delta_v)
        self.total_deorbit_delta_v = original_orbit_delta_v + target_orbit_delta_v

        # delta-v budget for mission
        delta_v_budget = 1.5 * self.sat_lifetime * v_drag + 100 # adding a safety margin

        # Indicator for altitudes that are normally compliant
        self.naturally_compliant_vector = np.zeros(self.mocat.scenario_properties.n_shells)
        self.naturally_compliant_vector[:k_star + 1] = 1

        # Calculate delta-v leftover after deorbit
        self.delta_v_after_deorbit = np.maximum(0, delta_v_budget - self.total_deorbit_delta_v * (1 - self.naturally_compliant_vector))

        # Calculate remaining lifetime after deorbit
        self.lifetime_after_deorbit = np.where(self.naturally_compliant_vector == 1, self.sat_lifetime, 
                                        (self.delta_v_after_deorbit / delta_v_budget) * self.sat_lifetime)

        # Calculate lifetime loss due to deorbit
        lifetime_loss = (self.sat_lifetime - self.lifetime_after_deorbit) / self.sat_lifetime

        # Cost function compilation
        self.total_lift_price = np.full_like(self.naturally_compliant_vector, self.lift_price * self.mass)
        self.lifetime_loss_cost = lifetime_loss * self.intercept
        self.deorbit_maneuver_cost = self.total_deorbit_delta_v * self.delta_v_cost
        self.stationkeeping_cost = delta_v_budget * self.delta_v_cost

        # Cost calculation moved from here
        # self.cost = (self.total_lift_price + self.stationkeeping_cost + self.lifetime_loss_cost + self.deorbit_maneuver_cost * (1 - 0)).tolist()
        self.v_drag = v_drag
        self.k_star = k_star 


        #BOND CALCULATIONS - compliance rate is defined in MOCAT json
        self.comp_rate = np.ones_like(self.total_lift_price) #* self.mocat.scenario_properties.species['active'][1].Pm # 0.95
        
        if self.bond is None:
            self.comp_rate = np.where(self.naturally_compliant_vector != 1, pmd_rate, self.comp_rate)
            # Calculate expected cost for baseline
            expected_deorbit_costs = (self.lifetime_loss_cost + self.deorbit_maneuver_cost) * self.comp_rate
            self.cost = (self.total_lift_price + self.stationkeeping_cost + expected_deorbit_costs)
            return 

        # Scale the bond with the mass of the satellite. Full bond cost = bond/700 kg
        # bond_per_kg = self.bond / 700
        # self.bond = bond_per_kg * self.mass
        
        self.discount_factor = 1/(1+self.discount_rate)
        self.bstar = (
            self.intercept
            * ((1 - self.discount_factor ** (self.lifetime_loss_cost / self.intercept)) / (1 - self.discount_factor))
            * self.discount_factor ** self.sat_lifetime
        )

        # Calculate compliance rate. 
        mask = self.bstar != 0  # Identify where bstar is nonzero
        
        # Updated logic from EconParameters1.py
        # non_comp_rate = 1 - pmd_rate
        # self.comp_rate[mask] = np.minimum(pmd_rate + non_comp_rate * self.bond / self.bstar[mask], 1)

        A = 57
        k = np.log(12) / 75
        scaled_effort = (self.bond / self.bstar[mask]) * 100
        self.comp_rate[mask] = 0.01 * (97 - A * np.exp(-k * scaled_effort))

        # Calculate expected cost for bond scenario 
        expected_deorbit_costs = (self.lifetime_loss_cost + self.deorbit_maneuver_cost) * self.comp_rate
        self.cost = (self.total_lift_price + self.stationkeeping_cost + expected_deorbit_costs)

        return       

    def modify_params_for_simulation(self, configuration: str):
        """
            This will modify the paramers for VAR and econ_parameters based on an input csv file. 
        """

        if configuration.lower() == 'baseline':
            self.bond = None
            self.tax = 0
            return

        # read the csv file - must be in the configuration folder
        path = f"./OPUS/configuration/{configuration}.csv"

        # read the csv file
        parameters = pd.read_csv(path)
        
        for i, row in parameters.iterrows():
            # Added .strip() from EconParameters1.py
            parameter_type = row['parameter_type'].strip()
            parameter_name = row['parameter_name'].strip()
            parameter_value = row['parameter_value']

            # Modify the value based on parameter_type
            if parameter_type == 'econ':
                # If the field exists in the class, update its value
                if hasattr(self, parameter_name):
                    setattr(self, parameter_name, parameter_value)
            else:
                print(f'Warning: Unknown parameter_type: {parameter_type}')

    def update_congestion_costs(self, current_environment, initial_environment):
            """
            Updates the delta-v cost for each shell based on object congestion.
            Handles both circular (1D flat) and elliptical (3D) environment arrays.

            Args:
                current_environment (np.ndarray): The current state of the environment.
                initial_environment (np.ndarray): The initial state of the environment (at year 0).
            """
            n_shells = self.mocat.scenario_properties.n_shells
            n_species = len(self.mocat.scenario_properties.species_names)
            is_elliptical = self.mocat.scenario_properties.elliptical

            # Sum objects per shell based on environment shape
            if is_elliptical:
                # current_environment shape is (n_shells, n_species, n_ecc_bins)
                # We sum over the species (axis 1) and eccentricity (axis 2)
                current_objects_per_shell = np.sum(current_environment, axis=(1, 2))
                initial_objects_per_shell = np.sum(initial_environment, axis=(1, 2))
            else:
                # current_environment shape is (n_species * n_shells,)
                current_reshaped = current_environment.reshape((n_species, n_shells))
                initial_reshaped = initial_environment.reshape((n_species, n_shells))
                # We sum over the species (axis 0)
                current_objects_per_shell = np.sum(current_reshaped, axis=0)
                initial_objects_per_shell = np.sum(initial_reshaped, axis=0)

            # Calculate congestion surcharge
            percent_change = np.zeros(n_shells)
            # Avoid division by zero for shells that started empty
            non_zero_mask = initial_objects_per_shell > 0
            
            # Calculate percent change
            percent_change[non_zero_mask] = self.congestion_switch * (current_objects_per_shell[non_zero_mask] - initial_objects_per_shell[non_zero_mask]) / initial_objects_per_shell[non_zero_mask]

            # Surcharge is based on the positive percent change
            surcharge = np.maximum(0, percent_change) * self.base_delta_v_cost
            self.delta_v_cost = self.base_delta_v_cost + surcharge

            # Recalculate cost components 
            self.deorbit_maneuver_cost = self.total_deorbit_delta_v * self.delta_v_cost
            delta_v_budget = 1.5 * self.sat_lifetime * self.v_drag + 100
            self.stationkeeping_cost = delta_v_budget * self.delta_v_cost
            
            # Recalculate the final cost list
            self.cost = (self.total_lift_price + self.stationkeeping_cost + self.lifetime_loss_cost + self.deorbit_maneuver_cost * (1 - 0)).tolist()
            return self.cost

    def return_congestion_costs(self, current_environment, initial_environment):
        n_shells = self.mocat.scenario_properties.n_shells
        n_species = len(self.mocat.scenario_properties.species_names)
        is_elliptical = self.mocat.scenario_properties.elliptical

        # Sum objects per shell based on environment shape
        if is_elliptical:
            # current_environment shape is (n_shells, n_species, n_ecc_bins)
            # We sum over the species (axis 1) and eccentricity (axis 2)
            current_objects_per_shell = np.sum(current_environment, axis=(1, 2))
            initial_objects_per_shell = np.sum(initial_environment, axis=(1, 2))
        else:
            # current_environment shape is (n_species * n_shells,)
            current_reshaped = current_environment.reshape((n_species, n_shells))
            initial_reshaped = initial_environment.reshape((n_species, n_shells))
            # We sum over the species (axis 0)
            current_objects_per_shell = np.sum(current_reshaped, axis=0)
            initial_objects_per_shell = np.sum(initial_reshaped, axis=0)

        # Calculate congestion surcharge
        percent_change = np.zeros(n_shells)
        # Avoid division by zero for shells that started empty
        non_zero_mask = initial_objects_per_shell > 0
        
        # Calculate percent change
        percent_change[non_zero_mask] = self.congestion_switch * (current_objects_per_shell[non_zero_mask] - initial_objects_per_shell[non_zero_mask]) / initial_objects_per_shell[non_zero_mask]

        # Surcharge is based on the positive percent change
        surcharge = np.maximum(0, percent_change) * self.base_delta_v_cost
        delta_v_cost = self.base_delta_v_cost + surcharge

        # Recalculate cost components 
        deorbit_maneuver_cost = self.total_deorbit_delta_v * self.delta_v_cost
        delta_v_budget = 1.5 * self.sat_lifetime * self.v_drag + 100
        stationkeeping_cost = delta_v_budget * delta_v_cost
        
        # Recalculate the final cost list
        cost = (self.total_lift_price + stationkeeping_cost + self.lifetime_loss_cost + deorbit_maneuver_cost * (1 - 0)).tolist()
        return cost