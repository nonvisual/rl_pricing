import gym
from gym import spaces
import numpy as np
from typing import List, Tuple

import torch
from data.generator import DemandGenerator
from game.state import GameState
import plotly.graph_objs as go
import plotly.io as pio


YEAR_WEEKS = 52


class PricingGameEnv(gym.Env):
    metadata = {"render.modes": ["human"]}

    def __init__(
        self,
        demand_generator: DemandGenerator,
        num_products: int,
        seed: int = 12345,
        num_weeks: int = YEAR_WEEKS,
        min_price: float = 10.0,
        max_price: float = 200.0,
        min_cogs: float = 0.6,
        max_cogs: float = 0.9,
        max_initial_stock: int = 2000,
        profit_lack_penalty: float = 10.0,
        target_profit_ratio: float = 0.05,
    ):
        super().__init__()

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

        # Define observation and action spaces

        self.observation_space = spaces.Dict(
            {
                "cw": spaces.Box(low=0, high=200, shape=(1,), dtype=np.int32),
                "sales": spaces.Box(low=0, high=np.inf, shape=(self.num_products,), dtype=np.float32),
                "black_prices": spaces.Box(low=0, high=np.inf, shape=(self.num_products,), dtype=np.float32),
                "cogs": spaces.Box(low=0, high=np.inf, shape=(self.num_products,), dtype=np.float32),
                "residual_value": spaces.Box(low=0, high=np.inf, shape=(self.num_products,), dtype=np.float32),
                "article_season_start": spaces.Box(
                    low=0, high=self.max_initial_stock, shape=(self.num_products,), dtype=np.int32
                ),
                "article_season_end": spaces.Box(
                    low=0, high=self.max_initial_stock, shape=(self.num_products,), dtype=np.int32
                ),
                "shipment_costs": spaces.Box(low=0, high=np.inf, shape=(self.num_products,), dtype=np.float32),
                "stocks": spaces.Box(low=0, high=self.max_initial_stock, shape=(self.num_products,), dtype=np.int32),
                "online_status": spaces.Box(low=0, high=1, shape=(self.num_products,), dtype=np.int32),
                "revenues": spaces.Box(low=0, high=np.inf, shape=(self.num_products,), dtype=np.float32),
                "profits": spaces.Box(low=-np.inf, high=np.inf, shape=(self.num_products,), dtype=np.float32),
                "sdrs": spaces.Box(low=0, high=np.inf, shape=(1,), dtype=np.float32),
                "discounts": spaces.Box(low=0, high=100, shape=(self.num_products,), dtype=np.float32),
            }
        )

        self.action_space = spaces.Box(low=0, high=70, shape=(self.num_products,), dtype=np.float32)

    def reset(self, seed=None):
        if seed:
            self.seed = seed
        self.reset_game(self.seed)
        obs = self._get_observation()
        return obs

        # discounts - array of chosen discounts, returns sales, revenue, profit

    def _update_online_status(self):
        new_status = np.array([(self.current_cw >= self.article_season_start[i]) for i in range(self.num_products)])
        self.online_status.append(new_status)

    def _get_observation(self):
        observations = {
            "cw": self.current_cw,
            "sales": self.sales[-1],
            "black_prices": self.black_prices,
            "cogs": self.cogs,
            "residual_value": self.residual_value,
            "article_season_start": self.article_season_start,
            "article_season_end": self.article_season_end,
            "shipment_costs": self.shipment_costs,
            "stocks": self.stocks[-1],
            "online_status": self.online_status[-1],
            "revenues": self.revenues[-1],
            "profits": self.profits[-1],
            "sdrs": self.sdrs[-1],
            "discounts": self.discounts[-1],
        }

        return observations

    def _play_prices(self, discounts: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        # 1 update week and online statuses
        eps = 0.0001
        self._update_online_status()

        # 2 compute sales
        sales = self.demand_generator.compute_sale(discounts, self.online_status, self.current_cw, self.stocks)
        stocks = self.stocks[self.current_cw - 1] - sales

        revenue = self.black_prices * (1 - discounts / 100.0) * sales
        profit = (self.black_prices * (1 - discounts / 100.0) - self.cogs - self.shipment_costs) * sales
        sdr = (self.black_prices * (discounts / 100.0) * sales).sum() / ((self.black_prices * sales).sum() + eps)
        return sales, stocks, self.online_status[self.current_cw - 1], revenue, profit, sdr

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
        self.discounts = [np.zeros(self.num_products)]
        self.stocks = [self.initial_stocks]
        self.sales = [np.zeros(self.num_products)]
        self.online_status = [np.zeros(self.num_products, dtype=bool)]
        self._update_online_status()
        self.revenues = [np.zeros(self.num_products)]
        self.profits = [np.zeros(self.num_products)]
        self.sdrs = [0]

        self.demand_generator.set_products(self.num_products, self.article_season_start, self.article_season_end, seed)

    def step(self, action: torch.TensorType) -> Tuple[np.ndarray, float, bool, dict]:
        assert self.action_space.contains(action), f"Invalid action: {action}"

        sales, stocks, online_status, revenue, profit, sdr = self._play_prices(action)
        done = self.current_cw >= self.num_weeks
        info = {"week": self.current_cw}

        # Update game history
        self.current_cw += 1
        self.discounts.append(action)
        self.sales.append(sales)
        self.stocks.append(stocks)
        self.online_status.append(online_status)
        self.revenues.append(revenue)
        self.profits.append(profit)
        self.sdrs.append(sdr)

        # Calculate reward

        if done:
            residual_revenue = (self.stocks[-1] * self.black_prices).sum()
            residual_profit = (self.stocks[-1] * (self.black_prices - self.cogs)).sum()

            step_revenue = revenue.sum()
            total_revenue = sum([r.sum() for r in self.revenues]) + residual_revenue
            total_profit = sum([p.sum() for p in self.profits]) + residual_profit

            penalty = max(0, total_revenue * self.target_profit_ratio - total_profit) * self.profit_lack_penalty
            reward = step_revenue + residual_revenue - penalty
        else:
            reward = revenue.sum()

        observations = self._get_observation()

        return observations, reward, done, info

    def render(self, mode="plotly"):
        # assert mode == "plotly", "Only plotly mode is supported"

        summed_revenue = [r.sum() for r in self.revenues]
        summed_profit = [p.sum() for p in self.profits]
        x = list(range(len(self.profits)))
        trace1 = go.Scatter(x=x, y=summed_revenue, mode="lines", name="Revenue")
        trace2 = go.Scatter(x=x, y=summed_profit, mode="lines", name="Profits")

        fig1 = go.Figure(data=[trace1, trace2])
        fig1.update_layout(title="Revenue and Profit", xaxis_title="Calendar Week", yaxis_title="Euros")
        fig1.update_traces(line=dict(width=2))

        fig2 = go.Figure(data=go.Scatter(x=x, y=self.sdrs, mode="lines"))
        fig2.update_layout(title="SDR", xaxis_title="Calendar Week", yaxis_title="SDR")
        fig2.update_traces(line=dict(width=2))

        pio.show(fig1)
        pio.show(fig2)

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
