import numpy as np


seasons = [1, 2, 3, 4]
default_market_seasons = [1] * 13 + [2] * 13 + [3] * 13 + [4] * 13
YEAR_WEEKS = 52


class PricingGame:
    def __init__(
        self, demand_generator, num_products, market_seasons=default_market_seasons, seed=12345, num_weeks=YEAR_WEEKS
    ):
        self.demand_generator = demand_generator
        self.num_products = num_products
        self.num_weeks = num_weeks

        self.reset_game(seed)

    def reset_game(self, seed):
        # set new seed
        self.demand_generator.set_seed(seed)
        self.seed = seed
        np.random.seed(seed)

        # re-initialize article information
        self.black_prices = np.random.uniform(10.0, 200.0, self.num_products)
        self.cogs = np.random.uniform(0.1, 0.8, self.num_products) * self.black_prices
        self.residual_value = np.random.uniform(0.2, 0.9, self.num_products) * self.black_prices
        self.article_season_start = np.random.choice(range(YEAR_WEEKS - 10), size=self.num_products)
        self.article_season_end = np.clip(
            self.article_season_start + np.random.choice(range(30), size=self.num_products), 0, YEAR_WEEKS
        )
        self.product_names = [f"product{i+1}" for i in range(self.num_products)]
        self.initial_stocks = np.random.uniform(10.0, 3000.0, self.num_products).astype(int)

        # reset game history
        self.current_cw = 0
        self.discounts = []
        self.stocks = [self.initial_stocks]
        self.sales = []
        self.online_status = []
        self._update_online_status()

    # discounts - array of chosen discounts, returns sales, revenue, profit
    def play_prices(self, discounts):
        # 1 update week and online statuses
        self.current_cw += 1
        self._update_online_status()

        # 2 compute sales
        sales = self.generator.compute_sale(discounts)
        stocks = self.stocks[self.current_cw - 1] - sales
        self.discounts.append(discounts)
        self.sales.append(sales)
        self.stocks.append(stocks)

        return sales, stocks, self.online_status[self.current_cw - 1]

    def _update_online_status(self):
        new_status = np.array(
            [
                (self.current_cw >= self.article_season_start[i])
                or (self.online_status[self.current_cw - 1][i] and (self.current_cw > 1))
                for i in range(self.num_products)
            ]
        )
        self.online_status.append(new_status)

    def get_state(self):
        pass

    def get_history(self):
        pass
