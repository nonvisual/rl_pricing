import numpy as np
import pandas as pd
from typing import Tuple


seasons = [1, 2, 3, 4]
default_market_seasons = [1] * 13 + [2] * 13 + [3] * 13 + [4] * 13


class DemandGenerator:
    def compute_sale(self, discounts):
        raise NotImplementedError()

    def generate(self):
        raise NotImplementedError()
    
    def play_prices(self, discounts)

class SimpleDemandGenerator:
    def __init__(
        self,
        max_unexplained_noise_sd=0.3,
        mu_elasticity=-2.0,
        sd_elasticity=1.5,
        base_demand_mean=30,
        base_demand_sd=15,
        seed=12345,
    ):
        np.random.seed(seed)

        self.max_unexplained_noise_sd = max_unexplained_noise_sd
        self.mu_elasticity = mu_elasticity
        self.sd_elasticity = sd_elasticity
        self.base_demand_mean = base_demand_mean
        self.base_demand_sd = base_demand_sd

    def set_products(self, num_products, article_season_start, article_season_end):
        
        self.article_season_start = article_season_start
        self.article_season_end = article_season_end

        self.unexplained_noise_sd = np.clip(np.random.uniform(0, self.max_unexplained_noise_sd, num_products), 0, None)
        self.base_demands = np.clip(np.random.normal(self.base_demand_mean, self.base_demand_sd, num_products), 0, None)
        self.elasticities = np.random.normal(self.mu_elasticity, self.sd_elasticity, num_products)
        self.seasonality_factors = np.random.uniform(0.0, 0.9, num_products)

        


    def compute_sale(self, discounts):
        basic_demand = (
            self.base_demands
            * (1 - discounts / 100.0) ** self.elasticities
            * self.online_status[self.current_cw - 1].astype(int)
        )  # 0 if not online
        demand_with_noise = basic_demand + basic_demand * np.random.normal(loc=0, scale=self.unexplained_noise_sd)

        demand_seasonality = (
            self.seasonality_factors * demand_with_noise * np.where(self.article_season == self.current_season, 1, 0)
        )

        return np.clip(demand_with_noise + demand_seasonality, 0, self.stocks[self.current_cw - 1]).astype(int)


    # discounts - array of chosen discounts, returns sales, revenue, profit
    def play_prices(self, discounts) -> Tuple[np.ndarray, np.ndarray]:
        # 1 update week and online statuses
        self.current_cw += 1
        self._update_online_status()

        # 2 compute sales
        sales = self.compute_sale(discounts)
        stocks = self.stocks[self.current_cw - 1] - sales
        self.discounts.append(discounts)
        self.sales.append(sales)
        self.stocks.append(stocks)

        return sales, stocks, self.online_status[self.current_cw - 1]

    def get_open_information(self):
        return

    def compute_financial_kpis(self):
        return
