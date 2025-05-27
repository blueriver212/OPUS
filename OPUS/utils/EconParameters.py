import numpy as np
from pyssem.model import Model
from pyssem.utils.drag.drag import densityexp
import pandas as pd
import matplotlib.pyplot as plt

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

        # self.lift_price = 5000
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
        self.delta_v_cost = params.get("delta_v_cost", 1000)

        # # Price of lift [$/kg]
        # # Default is $5000/kg based on price index calculations
        self.lift_price = params.get("lift_price", 5000)
        
        # regulatory non-compliance
        self.prob_of_non_compliance = params.get("prob_of_non_compliance", 0)

        # Bond amount
        self.bond = params.get("bond", None)

        # Post Mission Disposal Rate
        self.pmd_rate = 0.9

    def calculate_cost_fn_parameters(self):
        shell_marginal_decay_rates = np.zeros(self.mocat.scenario_properties.n_shells)
        shell_marginal_residence_times = np.zeros(self.mocat.scenario_properties.n_shells)
        self.shell_cumulative_residence_times = np.zeros(self.mocat.scenario_properties.n_shells)

        # Here using the ballastic coefficient of the species, we are trying to find the highest compliant altitude/shell
        for k in range(self.mocat.scenario_properties.n_shells):
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
            v_drag[k] = F_drag / 223 * delta_t * 1e-3

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
        self.total_lift_price = np.full_like(self.naturally_compliant_vector, self.lift_price * 223) # this is mass and hard coded, needs to be fixed
        self.lifetime_loss_cost = lifetime_loss * self.intercept
        self.deorbit_maneuver_cost = self.total_deorbit_delta_v * self.delta_v_cost
        self.stationkeeping_cost = delta_v_budget * self.delta_v_cost

        # self.cost = (self.total_lift_price + self.stationkeeping_cost + self.lifetime_loss_cost + self.deorbit_maneuver_cost * (1 - 0)).tolist() # should be self.mocat.scenario_properties.P which is the probability of regulatory non-compliance. 
        self.cost = (self.total_lift_price + self.stationkeeping_cost + self.deorbit_maneuver_cost * (1 - 0)).tolist()
        self.v_drag = v_drag
        self.k_star = k_star 

        #BOND CALCULATIONS - compliance rate is defined in MOCAT json
        self.comp_rate = np.ones_like(self.cost) #* self.mocat.scenario_properties.species['active'][1].Pm # 0.95
        
        if self.bond is None:
            self.comp_rate = np.where(self.naturally_compliant_vector != 1, 0.65, self.comp_rate)
            return 
        
        self.discount_factor = 1/(1+self.discount_rate)
        self.bstar = (
            self.intercept
            * ((1 - self.discount_factor ** (self.lifetime_loss_cost / self.intercept)) / (1 - self.discount_factor))
            * self.discount_factor ** self.sat_lifetime
        )

        # Calculate compliance rate. 
        mask = self.bstar != 0  # Identify where bstar is nonzero
        # self.comp_rate[mask] = np.minimum(0.65 + 0.35 * self.bond / self.bstar[mask], 1)

        # With Option 1 quation 
        A = 57
        k = np.log(12) / 75

        scaled_effort = (self.bond / self.bstar[mask]) * 100
        self.comp_rate[mask] = 0.01 * (97 - A * np.exp(-k * scaled_effort))

        return

    def modify_params_for_simulation(self, configuration, baseline=False):
        """
            This will modify the paramers for VAR and econ_parameters based on an input csv file. 
        """

        # read the csv file - must be in the configuration folder
        path = f"./OPUS/configuration/{configuration}.csv"

        # read the csv file
        parameters = pd.read_csv(path)
        
        for i, row in parameters.iterrows():
            parameter_type = row['parameter_type']
            parameter_name = row['parameter_name']
            parameter_value = row['parameter_value']

            # Modify the value based on parameter_type
            if parameter_type == 'econ':
                # If the field exists in the class, update its value
                if hasattr(self, parameter_name):
                    setattr(self, parameter_name, parameter_value)
            else:
                print(f'Warning: Unknown parameter_type: {parameter_type}')

        if baseline:
            self.bond = None
            self.tax = 0



