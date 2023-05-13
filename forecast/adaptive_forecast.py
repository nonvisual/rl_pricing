from math import isclose
import numpy as np
from base import GeneralForecast

"""
Simple adaptive forecast. Starts with general assumption about article (base demand, base_elasticity) and then adapts it with obeservations 
"""


class AdaptiveForecast(GeneralForecast):
    def __init__(self):
        self.trained = False

    def train(self, env, base_demand=30.0, elasticity=-5.0):
        num_products = env.num_products
        self.general_base_demand = base_demand
        self.general_elasticity = elasticity
        self.obs = []
        self.base_demand = [self.general_base_demand] * num_products
        self.elasticity = [self.general_elasticity] * num_products
        self.trained = True
        pass

    def create_forecast_grid(self, obs, num_discounts=10):
        # if there is obs for 0, and if current base demand is default -> overwrite, otherwise take average
        # if !=0 observation, < base demand -> update base demand, update elasticity
        # if != observation > base demand -> guess elasticity

        for i, discount in enumerate(obs["discounts"]):
            if obs["online_status"][i]:
                d = discount / 100.0  ## observations have discounts as %, i.e. * 100

                if np.isclose(d, 0.0, atol=0.001):
                    self.base_demand[i] = max(0, obs["sales"][i])
                else:
                    if obs["sales"][i] < self.base_demand[i]:
                        self.base_demand[i] = obs["sales"][i]
                    else:
                        implied_bd = obs["sales"][i] / (1 - d) ** self.elasticity[i]
                        self.base_demand[i] = max(self.base_demand[i], implied_bd)

        basic_demands = []
        for d in np.linspace(0, 0.7, num_discounts):
            basic_demands.append(self.base_demand * (1 - d) ** self.elasticity)

        return np.clip(np.vstack(basic_demands).T, 0, None)
