import numpy as np
from pyssem.model import Model
from pyssem.utils.drag.drag import densityexp
class EconParameters:
    """
    Class to build satellite cost using physcial and economic parameters
    Uses the disposal time regulation in the EconParams to calculate the highest compliant altitude. 
    Then uses the delta-v required for Hohmann transfer to contruct the cost function.

    It is initialized with default values for the parameters. 
    Then the calculate_cost_fn_parameters function is called to calculate the cost function parameters.
    """
    def __init__(self):
        # Lifetime of a satellite [years]
        # Default value is 5 years for MOCAT4S
        self.sat_lifetime = 5

        # Regulated disposal rule time [years], default is 5
        self.disposal_time = 5

        # Discount rate [pp/year]
        self.discount_rate = 0.05

        # Parameters specific to linear revenue model [$/year]
        # A satellite facing no competition earns $750,000 per year in revenues
        self.intercept = 7.5e5  # [$/year]
        self.coef = 1.0e2       # [$/satellite/year]
        self.tax = 0.0          # [%] tax rate on shell-specific pc

        # Cost for a single satellite to use any shell [$]
        
        # Cost of delta-v [$/km/s]
        self.delta_v_cost = 1000

        # Price of lift [$/kg]
        # Default is $5000/kg based on price index calculations
        self.lift_price = 5000

    def calculate_cost_fn_parameters(self, mocat: Model):
        shell_marginal_decay_rates = np.zeros(mocat.scenario_properties.n_shells)
        shell_marginal_residence_times = np.zeros(mocat.scenario_properties.n_shells)
        shell_cumulative_residence_times = np.zeros(mocat.scenario_properties.n_shells)

        # Here using the ballastic coefficient of the species, we are trying to find the highest compliant altitude/shell
        for k in range(mocat.scenario_properties.n_shells):
            rhok = densityexp(mocat.scenario_properties.R0_km[k])

            # satellite 
            beta = 0.0172 # ballastic coefficient, area * mass * drag coefficient. This should be done for each species!
            rvel_current_D = -rhok * beta * np.sqrt(mocat.scenario_properties.mu * mocat.scenario_properties.R0[k]) * (24 * 3600 * 365.25)
            shell_marginal_decay_rates[k] = -rvel_current_D/mocat.scenario_properties.Dhl
            shell_marginal_residence_times[k] = 1/shell_marginal_decay_rates[k]
        
        shell_cumulative_residence_times = np.cumsum(shell_marginal_residence_times)

        # Find the index of shell_cumulative_residence_times, k_star, which is the largest index that  shell_cumulative_residence_times(k_star) <= self.disposalTime
        indices = np.where(shell_cumulative_residence_times <= self.disposal_time)[0]
        k_star = max(indices) if len(indices) > 0 else 0

        # Physics based cost function using delta-v and deorbit requirements
        v_drag = np.zeros(mocat.scenario_properties.n_shells)
        delta_t = 24 * 3600 * 365.25 # time inteval in seconds

        # Calculate drag delta-v for stationkeeping maneuvers
        for k in range(mocat.scenario_properties.n_shells):
            rhok = densityexp(mocat.scenario_properties.R0_km[k])
            orbital_velocity = np.sqrt(mocat.scenario_properties.mu / mocat.scenario_properties.R0[k])
            F_drag = 2.2 * 0.5 * rhok * orbital_velocity**2 * 1.741 # this is cd and area and have been hard coded
            v_drag[k] = F_drag / 223 * delta_t * 1e-3

        # Calculate the delta-v for deorbiting
        original_orbit_delta_v = np.zeros(mocat.scenario_properties.n_shells)
        target_orbit_delta_v = np.zeros(mocat.scenario_properties.n_shells)
        r2 = mocat.scenario_properties.R0[k_star]

        for k in range(mocat.scenario_properties.n_shells):
            original_orbit_delta_v[k] = np.sqrt(mocat.scenario_properties.mu / mocat.scenario_properties.R0[k]) * (1 - np.sqrt(2 * r2 / (mocat.scenario_properties.R0[k] + r2)))
            target_orbit_delta_v[k] = np.sqrt(mocat.scenario_properties.mu / r2) * (np.sqrt(2 * mocat.scenario_properties.R0[k] / (mocat.scenario_properties.R0[k] + r2)) - 1)

        original_orbit_delta_v = np.maximum(0, original_orbit_delta_v)
        target_orbit_delta_v = np.maximum(0, target_orbit_delta_v)
        total_deorbit_delta_v = original_orbit_delta_v + target_orbit_delta_v

        # delta-v budget for mission
        delta_v_budget = 1.5 * self.sat_lifetime * v_drag + 100 # adding a safety margin

        # Indicator for altitudes that are normally compliant
        naturally_compliant_vector = np.zeros(mocat.scenario_properties.n_shells)
        naturally_compliant_vector[:k_star + 1] = 1

        # Calculate delta-v leftover after deorbit
        delta_v_after_deorbit = np.maximum(0, delta_v_budget - total_deorbit_delta_v * (1 - naturally_compliant_vector))

        # Calculate remaining lifetime after deorbit
        lifetime_after_deorbit = np.where(naturally_compliant_vector == 1, self.sat_lifetime, 
                                        (delta_v_after_deorbit / delta_v_budget) * self.sat_lifetime)

        # Calculate lifetime loss due to deorbit
        lifetime_loss = (self.sat_lifetime - lifetime_after_deorbit) / self.sat_lifetime

        # Cost function compilation
        total_lift_price = self.lift_price * 223 # this is mass and hard coded
        lifetime_loss_cost = lifetime_loss * self.intercept
        deorbit_maneuver_cost = total_deorbit_delta_v * self.delta_v_cost
        stationkeeping_cost = delta_v_budget * self.delta_v_cost
        cost = (total_lift_price + stationkeeping_cost + lifetime_loss_cost * (1 - 0)).tolist() # should be mocat.scenario_properties.P which is the probability of regulatory non-compliance. 
        
        self.cost = cost
        self.total_lift_price = total_lift_price
        self.deorbit_manuever_cost = deorbit_maneuver_cost
        self.stationkeeping_cost = stationkeeping_cost
        self.lifetime_loss_cost = lifetime_loss_cost
        self.v_drag = v_drag
        self.k_star = k_star 