import numpy as np
import pandas as pd
from typing import Optional, Tuple
import sys
import plotly.graph_objs as go

DISCOUNT_PERCENT = 100.0


class DemandGenerator:
    def compute_sale(self, discounts: np.ndarray):
        raise NotImplementedError()

    def set_products(self, num_products: int, article_season_start: np.ndarray, article_season_end: np.ndarray):
        raise NotImplementedError()


class SimpleDemandGenerator(DemandGenerator):
    def __init__(
        self,
        max_unexplained_noise_sd: float = 0.3,
        mu_elasticity: float = -5.0,
        sd_elasticity: float = 1.5,
        base_demand_mean: float = 30,
        base_demand_sd: float = 15,
    ):
        self.max_unexplained_noise_sd = max_unexplained_noise_sd
        self.mu_elasticity = mu_elasticity
        self.sd_elasticity = sd_elasticity
        self.base_demand_mean = base_demand_mean
        self.base_demand_sd = base_demand_sd

    def set_products(self, num_products: int, article_season_start: np.ndarray, article_season_end: np.ndarray):
        self.article_season_start = article_season_start
        self.article_season_end = article_season_end

        self.unexplained_noise_sd = np.clip(np.random.uniform(0, self.max_unexplained_noise_sd, num_products), 0, None)
        self.base_demands = np.clip(np.random.normal(self.base_demand_mean, self.base_demand_sd, num_products), 0, None)
        self.elasticities = np.random.normal(self.mu_elasticity, self.sd_elasticity, num_products)
        self.seasonality_factors = np.random.uniform(0.0, 0.9, num_products)

    def compute_sale(self, discounts: np.ndarray, online_status: np.ndarray, current_cw: int, stocks: np.ndarray):
        basic_demand = (
            self.base_demands * (1 - discounts / DISCOUNT_PERCENT) ** self.elasticities * online_status[-1].astype(int)
        )  # 0 if not online
        demand_with_noise = basic_demand + basic_demand * np.random.normal(loc=0, scale=self.unexplained_noise_sd)

        seasonality_gap = 3  # defines boundaries of seasonal sales boost for article
        demand_seasonality = (
            self.seasonality_factors
            * demand_with_noise
            * np.where(
                (self.article_season_start + seasonality_gap <= current_cw)
                & (self.article_season_end - seasonality_gap >= current_cw),
                1,
                0,
            )
        )

        return np.clip(demand_with_noise + demand_seasonality, 0, stocks[current_cw - 1]).astype(int)

    def visualize_demands(self, points: int = 20, article_index: Optional[int] = None):
        num_articles = len(self.base_demands)
        stocks = np.array([sys.maxsize] * num_articles)
        online_status = np.array([True] * num_articles)
        demand_points = []
        discount_points = []
        max_discount = 70.0
        for i in range(points):
            discounts = np.array([max_discount / points * i] * num_articles)
            demands = self.compute_sale(discounts, online_status, 0, stocks)
            demand_points.append(demands)
            discount_points.append(discounts)

        x = [d[0] for d in discount_points]

        if article_index is not None:
            assert (article_index >= 0) and (article_index < num_articles), "Invalid article index"

            y = [d[article_index] for d in demand_points]
        else:
            y = [sum(d) for d in demand_points]

        trace = go.Scatter(
            x=x,
            y=y,
            mode="lines",
        )

        # Create a layout
        layout = go.Layout(
            title=f"Demand vs discount assuming online status"
            + (" for whole assortment" if article_index is None else f" for article {article_index}"),
            xaxis=dict(title="Discount %"),
            yaxis=dict(title="Demand units"),
        )

        # Create a figure object that includes trace and layout
        fig = go.Figure(data=[trace], layout=layout)

        # Display the plot
        fig.show()
