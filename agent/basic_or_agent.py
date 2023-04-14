from agent.general_agent import GeneralAgent
from forecast.adaptive_forecast import AdaptiveForecast
from game.simulator import PricingGame
from game.state import GameState
import numpy as np
import pulp

WEEKS_YEAR = 52


class BasicORAgent(GeneralAgent):
    def __init__(self, forecast=AdaptiveForecast()):
        self.forecast = forecast

    def create_optimization_model(self, game: PricingGame, obs, forecast_grid):
        num_products = game.num_products
        black_prices = obs["black_prices"]
        cw = obs["cw"]
        cogs = obs["cogs"]
        residual_value = obs["residual_value"]
        article_season_start = obs["article_season_start"]
        article_season_end = obs["article_season_end"]
        shipment_costs = obs["shipment_costs"]
        stocks = obs["shipment_costs"]

        num_discounts = forecast_grid.shape[1]  # number of discounts
        discount_range = np.linspace(0, 0.7, num_discounts)

        model = pulp.LpProblem("Revenue Optimization", pulp.LpMaximize)

        discounts = pulp.LpVariable.dicts(
            "discount",
            [(i, w, j) for i in range(num_products) for w in range(cw, WEEKS_YEAR) for j in range(num_discounts)],
            lowBound=0,
            cat="Binary",
        )
        sales_quantity = pulp.LpVariable.dicts(
            "sales_quantity",
            [(i, w) for i in range(num_products) for w in range(cw, WEEKS_YEAR)],
            lowBound=0,
            cat="Continous",
        )
        stock_quantity = pulp.LpVariable.dicts(
            "stock_quantity",
            [(i, w) for i in range(num_products) for w in ([cw - 1] + list(range(cw, WEEKS_YEAR + 1)))],
            lowBound=0,
            cat="Continous",
        )

        total_revenue = pulp.LpVariable("revenue", lowBound=0, cat="Continuous")

        model += total_revenue == pulp.lpSum(
            [
                sales_quantity[(i, w)] * black_prices[i] * (1 - discount_range[j])
                for i in range(num_products)
                for w in range(cw, 52)
                for j in range(num_discounts)
            ]
        )
        # add residual value

        model += total_revenue

        for i in range(num_products):
            model += stock_quantity[i, cw - 1] == stocks[i]
            for w in range(cw, 52):
                model += pulp.lpSum([discounts[(i, w, j)] for j in range(num_discounts)]) == 1

                model += (
                    pulp.lpSum([forecast_grid[i][j] * discounts[(i, w, j)] for j in range(num_discounts)])
                    >= sales_quantity[i, w]
                )
                model += stock_quantity[i, w] >= sales_quantity[i, w]
                model += stock_quantity[i, w] == stock_quantity[i, w] - sales_quantity[i, w]
        return model, discounts, sales_quantity, total_revenue

    def act(self, game: PricingGame, obs) -> np.ndarray:
        if not self.forecast.trained:
            self.forecast.train(game)
        forecast_grid = self.forecast.create_forecast_grid(obs["sales"])

        model, discounts, sales, t_revenue = self.create_optimization_model(game, obs, forecast_grid)
        model.solve()

        num_discounts = forecast_grid.shape[1]  # number of discounts
        discount_range = np.linspace(0, 0.7, num_discounts)
        discount_values = np.zeros((forecast_grid.shape[0]))
        for i in range(forecast_grid.shape[0]):
            discount_values[i] = sum(discount_values[j] * discounts[(i, 0, j)].varValue for j in range(num_discounts))

        return discount_values
