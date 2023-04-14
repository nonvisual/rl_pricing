from math import isclose
import numpy as np

"""
Simple adaptive forecast. Starts with general assumption about article (base demand, base_elasticity) and then adapts it with obeservations 
"""


class AdaptiveForecast:
    def __init__(self):
        self.trained = False

    def train(self, env, base_demand=100.0, elasticity=-3.0):
        num_products = env.num_products
        self.general_base_demand = base_demand
        self.general_elasticity = elasticity
        self.obs = []
        self.base_demand = [self.general_base_demand] * num_products
        self.elasticity = [self.general_elasticity] * num_products
        self.trained = True
        pass

    def create_forecast_grid(self, obs):
        # if there is obs for 0, and if current base demand is default -> overwrite, otherwise take average
        # if !=0 observation, < base demand -> update base demand, update elasticity
        # if != observation > base demand -> guess elasticity

        for i, d in enumerate(obs["discounts"]):
            if np.isclose(d, 0.0, atol=2.5):
                if np.isclose(self.base_demand[i], self.general_base_demand):
                    self.base_demand[i] = obs["sales"][i]
                else:
                    self.base_demand[i] = (self.base_demand[i] + obs["sales"][i]) / 2.0
            else:
                if obs["sales"][i] < self.base_demand[i]:
                    # x = bd * (1-d)**e
                    implied_bd = obs["sales"][i] / (1 - d) ** self.elasticity[i]
                    self.base_demand[i] = (self.base_demand[i] + implied_bd) / 2.0
                    implied_el = np.emath.logn(1 - d, obs["sales"][i] / self.base_demand[i])
                    self.elasticity[i] = implied_el
                else:
                    implied_bd = obs["sales"][i] / (1 - d) ** self.elasticity[i]
                    self.base_demand[i] = implied_bd

        basic_demands = []
        for d in np.arange(0, 0.70, 0.05):
            basic_demands.append(self.base_demand * (1 - d) ** self.elasticity)

        week_demands = np.vstack(basic_demands).T

        return np.vstack(basic_demands).T
