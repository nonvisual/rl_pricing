import numpy as np
import pandas as pd
from typing import Tuple


seasons = [1,2,3,4]
default_market_seasons = [1]*13 + [2]*13 + [3]*13 + [4]*13


class DemandGenerator:
    
    def __init__(self, num_products, market_seasons = default_market_seasons, max_unexplained_noise_sd = 0.3,
                mu_elasticity = -2.0, sd_elasticity = 1.5):
        self.num_products = num_products
        self.black_prices = np.random.uniform(10.0, 200.0, num_products)
        self.cogs = np.random.uniform(0.1, 0.8, num_products) * self.black_prices
        self.residual_value = np.random.uniform(0.2, 0.9, num_products) * self.black_prices
        self.article_season =  np.random.choice(seasons, size=num_products) 
        self.market_seasons = market_seasons
        self.unexplained_noise_sd = np.clip(np.random.uniform(0, max_unexplained_noise_sd, num_products), 0, None)
        self.product_names = [f'product{i+1}' for i in range(num_products)]
        self.initial_stocks = np.random.uniform(10.0, 3000.0, num_products).astype(int)
        self.base_demands = np.clip(np.random.normal(30,30,num_products), 0, None)
        self.elasticities = np.random.normal(mu_elasticity, sd_elasticity, num_products)
        self.sales_events = np.random.choice([True, False], size=num_products, p = [0.4,0.6]) # 40% probability of sales event
        self.seasonality_factors = np.random.uniform(0.0, 0.9, num_products)
        
        self.current_cw = 0
        self.current_season = self.market_seasons[int(self.current_cw/52)]
        self.discounts = []
        self.stocks = [self.initial_stocks]
        self.sales = []
        self.online_status = []
        self._update_online_status()
    
    def compute_sale(self, discounts):
        seasonality_factor = 0.3
        basic_demand = self.base_demands * (1-discounts/100.0)**self.elasticities * self.online_status[self.current_cw-1].astype(int) # 0 if not online
        demand_with_noise = basic_demand + basic_demand * np.random.normal(loc=0, scale=self.unexplained_noise_sd) 

        demand_seasonality = self.seasonality_factors * demand_with_noise * np.where(self.article_season == self.current_season, 1, 0)
        
        
        return np.clip(demand_with_noise + demand_seasonality,0, self.stocks[self.current_cw-1]).astype(int)
    
    

    def _update_online_status(self):
        new_status = []
        if self.current_cw > 1:
            new_status = np.array([(self.current_season == self.article_season[i])
                                       or (self.online_status[self.current_cw-1][i] ) for i in range(self.num_products)])
        else: 
            new_status = np.array([(self.current_season == self.article_season[i]) 
                                                             for i in range(self.num_products)])

        self.online_status.append(new_status)
        
        
        
    # discounts - array of chosen discounts, returns sales, revenue, profit 
    def play_prices(self, discounts) -> Tuple[np.ndarray, np.ndarray]:
        
        # 1 update week and online statuses
        self.current_cw += 1
        self._update_online_status()
        
        # 2 compute sales
        sales = self.compute_sale(discounts)
        stocks = self.stocks[self.current_cw -1] - sales
        self.discounts.append(discounts)
        self.sales.append(sales)
        self.stocks.append(stocks)
        
        
        return sales, stocks, self.online_status[self.current_cw-1]
    
    def get_open_information(self):
        return
    
    def compute_financial_kpis(self):
        return
        
   

