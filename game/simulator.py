import numpy as np
from typing import List, Tuple
from data.generator import DemandGenerator
from game.state import GameState

YEAR_WEEKS = 52


class PricingGame:
    def __init__(
        self,
        demand_generator: DemandGenerator,
        num_products: int,
        seed: int = 12345,
        num_weeks: int = YEAR_WEEKS,
        min_price: float = 10.0,
        max_price: float = 200.0,
        min_cogs: float = 0.4,
        max_cogs: float = 0.9,
        max_initial_stock: int = 2000,
        profit_lack_penalty: float = 10.0,
        target_profit_ratio: float = 0.05,
    ):
        self.demand_generator = demand_generator
        self.num_products = num_products
        self.num_weeks = num_weeks

        self.min_price = min_price
        self.max_price = max_price
        self.max_cogs = max_cogs
        self.min_cogs = min_cogs
        self.max_initial_stock = max_initial_stock
        self.profit_lack_penalty = profit_lack_penalty
        self.target_profit_ratio = target_profit_ratio
        self.reset_game(seed)

    def reset_game(self, seed: int):
        # set new seed
        self.seed = seed
        np.random.seed(seed)

        # re-initialize article information
        self.black_prices = np.random.uniform(self.min_price, self.max_price, self.num_products)
        self.cogs = np.random.uniform(self.min_cogs, self.max_cogs, self.num_products) * self.black_prices
        self.residual_value = np.random.uniform(0.2, 1.0, self.num_products) * self.black_prices
        self.article_season_start = np.random.choice(range(YEAR_WEEKS - 10), size=self.num_products)
        self.article_season_end = np.clip(
            self.article_season_start + np.random.choice(range(30), size=self.num_products), 0, YEAR_WEEKS
        )
        self.product_names = [f"product{i+1}" for i in range(self.num_products)]
        self.initial_stocks = np.random.uniform(0, self.max_initial_stock, self.num_products).astype(int)
        self.shipment_costs = self.min_price / 2

        # reset game history
        self.current_cw = 0
        self.discounts = []
        self.stocks = [self.initial_stocks]
        self.sales = []
        self.online_status = []
        self._update_online_status()
        self.revenues = []
        self.profits = []
        self.sdrs = []

        self.demand_generator.set_products(self.num_products, self.article_season_start, self.article_season_end)

    # discounts - array of chosen discounts, returns sales, revenue, profit
    def play_prices(self, discounts: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        # 1 update week and online statuses
        self.current_cw += 1
        self._update_online_status()

        # 2 compute sales
        sales = self.demand_generator.compute_sale(discounts, self.online_status, self.current_cw, self.stocks)
        stocks = self.stocks[self.current_cw - 1] - sales
        self.discounts.append(discounts)
        self.sales.append(sales)
        self.stocks.append(stocks)

        revenue = self.black_prices * (1 - discounts / 100.0) * sales
        profit = (self.black_prices * (1 - discounts / 100.0) - self.cogs - self.shipment_costs) * sales
        sdr = (self.black_prices * (discounts / 100.0) * sales).sum() / (self.black_prices * sales).sum()
        self.revenues.append(revenue)
        self.profits.append(profit)
        self.sdrs.append(sdr)
        return sales, stocks, self.online_status[self.current_cw - 1], revenue, profit

    def _update_online_status(self):
        new_status = np.array(
            [True for i in range(self.num_products)]
        )  # np.array([(self.current_cw >= self.article_season_start[i]) for i in range(self.num_products)])
        self.online_status.append(new_status)

    def get_state(self) -> GameState:
        return GameState(self.profits, self.revenues, self.sales, self.sdrs, self.discounts)

    def get_final_score(self):
        revenue = sum([r.sum() for r in self.revenues])
        profit = sum([r.sum() for r in self.profits])

        residual_revenue = (self.stocks[-1] * self.black_prices).sum()
        residual_profit = (self.stocks[-1] * (self.black_prices - self.cogs)).sum()

        total_revenue = revenue + residual_revenue
        total_profit = profit + residual_profit

        penalty = max(0, total_revenue * self.target_profit_ratio - total_profit) * self.profit_lack_penalty

        score = total_revenue - penalty

        return score, total_revenue, total_profit, penalty, residual_revenue, residual_profit
