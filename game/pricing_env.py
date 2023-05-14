from ast import Dict
import gym
from gym import spaces
import numpy as np
from typing import List, Optional, Tuple

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
        num_weeks: int = YEAR_WEEKS,
        min_price: float = 10.0,
        max_price: float = 200.0,
        min_cogs: float = 0.4,
        max_cogs: float = 0.7,
        max_initial_stock: int = 2000,
        profit_lack_penalty: float = 10.0,
        target_profit_ratio: float = 0.05,
        mu_residual_value=0.25,
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
        self.mu_residual_value = mu_residual_value
        # dself.reset_game(seed)

        # Define observation and action spaces

        self.observation_space = spaces.Dict(
            {
                "cw": spaces.Box(low=0, high=200, shape=(1,), dtype=np.int32),
                "sales": spaces.Box(low=0, high=np.inf, shape=(self.num_products,), dtype=np.float32),
                "black_prices": spaces.Box(low=0, high=np.inf, shape=(self.num_products,), dtype=np.float32),
                "cogs": spaces.Box(low=0, high=np.inf, shape=(self.num_products,), dtype=np.float32),
                "residual_value": spaces.Box(low=0, high=np.inf, shape=(self.num_products,), dtype=np.float32),
                "article_season_start": spaces.Box(low=0, high=YEAR_WEEKS, shape=(self.num_products,), dtype=np.int32),
                "article_season_end": spaces.Box(low=0, high=YEAR_WEEKS, shape=(self.num_products,), dtype=np.int32),
                "shipment_costs": spaces.Box(low=0, high=np.inf, shape=(1,), dtype=np.float32),
                "stocks": spaces.Box(low=0, high=self.max_initial_stock, shape=(self.num_products,), dtype=np.int32),
                "online_status": spaces.Box(low=0, high=1, shape=(self.num_products,), dtype=np.int32),
                "revenues": spaces.Box(low=0, high=np.inf, shape=(self.num_products,), dtype=np.float32),
                "profits": spaces.Box(low=-np.inf, high=np.inf, shape=(self.num_products,), dtype=np.float32),
                "sdrs": spaces.Box(low=0, high=np.inf, shape=(1,), dtype=np.float32),
                "discounts": spaces.Box(low=0, high=100, shape=(self.num_products,), dtype=np.float32),
            }
        )

        self.action_space = spaces.Box(low=0, high=70, shape=(self.num_products,), dtype=np.float32)

    def reset(self, seed: Optional[int] = None) -> Dict:
        self.reset_game(seed)
        obs = self._get_observation()
        return obs

    def _update_online_status(self):
        new_status = np.array(
            [
                (self.current_cw >= self.article_season_start[i]) and (self.current_cw <= self.article_season_end[i])
                for i in range(self.num_products)
            ]
        )
        self.online_status.append(new_status)

    def _get_observation(self) -> Dict:
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

    def _play_prices(
        self, discounts: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
        # 1 update week and online statuses
        eps = 0.0001
        self._update_online_status()

        # 2 compute sales
        sales = self.demand_generator.compute_sale(discounts, self.online_status, self.current_cw, self.stocks)
        # no sales if status is offline
        sales[~self.online_status[-1]] = 0.0
        stocks = self.stocks[-1] - sales

        revenue = self.black_prices * (1 - discounts / 100.0) * sales
        profit = (self.black_prices * (1 - discounts / 100.0) - self.cogs - self.shipment_costs) * sales
        sdr = (self.black_prices * (discounts / 100.0) * sales).sum() / ((self.black_prices * sales).sum() + eps)
        return sales, stocks, self.online_status[self.current_cw - 1], revenue, profit, sdr

    def seed(self, seed):
        if not seed:
            seed = 12345  # jnp.random.randint(0, 2**32 - 1)
        np.random.seed(seed)
        self.current_seed = seed
        return seed

    def reset_game(self, seed):
        # set new seed
        self.seed(seed)

        # re-initialize article information
        self.black_prices = np.random.uniform(self.min_price, self.max_price, self.num_products)
        self.cogs = np.random.uniform(self.min_cogs, self.max_cogs, self.num_products) * self.black_prices
        self.residual_value = (
            np.clip(np.random.normal(self.mu_residual_value, 0.2, self.num_products), 0.0, 1.0) * self.black_prices
        )

        # model 2 big seasons
        season_peak_1 = 3
        season_peak_2 = 20
        season_peak_3 = 40
        sd = 4
        season_start = np.concatenate(
            [
                np.random.normal(season_peak_1, sd, int(self.num_products / 3)),
                np.random.normal(season_peak_2, sd, int(self.num_products / 3)),
                np.random.normal(season_peak_3, sd, self.num_products - 2 * int(self.num_products / 3)),
            ]
        )
        season_start = np.round(season_start).astype(int)
        np.random.shuffle(season_start)
        self.article_season_start = np.clip(season_start, 0, 52)

        self.article_season_end = np.clip(
            self.article_season_start + np.random.choice(range(12, 30), size=self.num_products), 0, YEAR_WEEKS
        ).astype(int)
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

        self.demand_generator.set_products(self.num_products, self.article_season_start, self.article_season_end)

    def step(self, action: torch.TensorType) -> Tuple[np.ndarray, float, bool, dict]:
        assert self.action_space.contains(action), f"Invalid action: {action}"

        sales, stocks, online_status, revenue, profit, sdr = self._play_prices(action)
        info = {"week": self.current_cw}

        # Update game history
        self.current_cw += 1
        self.discounts.append(action)
        self.sales.append(sales)
        self.stocks.append(stocks)
        self.revenues.append(revenue)
        self.profits.append(profit)
        self.sdrs.append(sdr)

        # Calculate reward
        # reward is sum of realized
        done = self.current_cw >= self.num_weeks

        season_is_done = (self.current_cw >= self.article_season_end).astype(int)
        if done:
            season_is_done = np.ones(len(season_is_done))
        residual_revenue = (self.stocks[-1] * self.residual_value * season_is_done).sum()
        residual_profit = (self.stocks[-1] * (self.residual_value - self.cogs) * season_is_done).sum()

        total_revenue = sum([r.sum() for r in self.revenues]) + residual_revenue
        total_profit = sum([p.sum() for p in self.profits]) + residual_profit

        penalty = max(0, total_revenue * self.target_profit_ratio - total_profit) * self.profit_lack_penalty
        reward = total_revenue - penalty

        observations = self._get_observation()

        return observations, reward, done, info

    def render(self, mode="plotly", extended=False):
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

        if extended:
            stock_online = [sum(self.stocks[i] * self.online_status[i].astype(int)) for i in range(len(self.profits))]

            fig3 = go.Figure(data=go.Scatter(x=x, y=stock_online, mode="lines"))
            fig3.update_layout(title="Units of stock online", xaxis_title="Calendar Week", yaxis_title="Units")
            fig3.update_traces(line=dict(width=2))

            pio.show(fig3)

            articles_online = [sum(self.online_status[i].astype(int)) for i in range(len(self.profits))]
            fig4 = go.Figure(data=go.Scatter(x=x, y=articles_online, mode="lines"))
            fig4.update_layout(title="Articles online", xaxis_title="Calendar Week", yaxis_title="Count")
            fig4.update_traces(line=dict(width=2))

            pio.show(fig4)

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

    def visualize_setup(self):
        season_length = self.article_season_end - self.article_season_start

        fig = go.Figure(data=[go.Histogram(x=season_length, nbinsx=30)])

        # Update the layout of the histogram
        fig.update_layout(title="Histogram of Article Season Length", xaxis_title="Weeks", yaxis_title="Count")

        # Show the histogram
        fig.show()

        fig = go.Figure(data=[go.Histogram(x=self.article_season_start, nbinsx=30)])

        # Update the layout of the histogram
        fig.update_layout(title="Histogram of Article Season Starts", xaxis_title="CW", yaxis_title="Count")

        # Show the histogram
        fig.show()
