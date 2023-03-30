import numpy as np
import pandas as pd
from typing import Tuple


DISCOUNT_PERCENT = 100.0


class DemandGenerator:
    def compute_sale(self, discounts: np.ndarray):
        raise NotImplementedError()

    def set_products(self, num_products: int, article_season_start: np.ndarray, article_season_end: np.ndarray):
        raise NotImplementedError()


class SimpleDemandGenerator(DemandGenerator):
    def __init__(
        self,
        max_unexplained_noise_sd: float = 0.1,
        mu_elasticity: float = -3.0,
        sd_elasticity: float = 1.5,
        base_demand_mean: float = 30,
        base_demand_sd: float = 15,
    ):
        self.max_unexplained_noise_sd = max_unexplained_noise_sd
        self.mu_elasticity = mu_elasticity
        self.sd_elasticity = sd_elasticity
        self.base_demand_mean = base_demand_mean
        self.base_demand_sd = base_demand_sd

    def set_products(
        self, num_products: int, article_season_start: np.ndarray, article_season_end: np.ndarray, seed: int
    ):
        np.random.seed(seed)

        self.article_season_start = article_season_start
        self.article_season_end = article_season_end

        self.unexplained_noise_sd = np.clip(np.random.uniform(0, self.max_unexplained_noise_sd, num_products), 0, None)
        self.base_demands = np.clip(np.random.normal(self.base_demand_mean, self.base_demand_sd, num_products), 0, None)
        self.elasticities = np.random.normal(self.mu_elasticity, self.sd_elasticity, num_products)
        self.seasonality_factors = np.random.uniform(0.0, 0.9, num_products)

    def compute_sale(self, discounts: np.ndarray, online_status: np.ndarray, current_cw: int, stocks: np.ndarray):
        basic_demand = (
            self.base_demands
            * (1 - discounts / DISCOUNT_PERCENT) ** self.elasticities
            * online_status[current_cw - 1].astype(int)
        )  # 0 if not online
        demand_with_noise = basic_demand + basic_demand * np.random.normal(loc=0, scale=self.unexplained_noise_sd)

        demand_seasonality = (
            self.seasonality_factors
            * demand_with_noise
            * np.where((self.article_season_start <= current_cw) & (self.article_season_end >= current_cw), 1, 0)
        )

        return np.clip(demand_with_noise + demand_seasonality, 0, stocks[current_cw - 1]).astype(int)
