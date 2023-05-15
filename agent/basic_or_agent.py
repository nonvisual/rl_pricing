from agent.general_agent import GeneralAgent
from forecast.adaptive_forecast import AdaptiveForecast
from game.simulator import PricingGame
from game.state import GameState
import numpy as np
import pulp

WEEKS_YEAR = 52


class BasicORAgent(GeneralAgent):
    def __init__(self, forecast=AdaptiveForecast(), include_future_articles=True, verbose_solver=True, timelimit=10):
        self.forecast = forecast
        self.verbose_solver = verbose_solver
        self.timelimit = timelimit
        self.include_future_articles = include_future_articles

    def is_in_optimization(self, current_cw, cw, start_dates, end_dates):
        in_optimization_horizon = np.logical_and(start_dates <= cw, cw <= end_dates)
        currently_online = current_cw >= start_dates
        if not self.include_future_articles:
            in_optimization_horizon = np.logical_and(in_optimization_horizon, currently_online)
        return in_optimization_horizon

    def create_optimization_model(self, game: PricingGame, obs, forecast_grid):
        num_products = game.num_products
        black_prices = obs["black_prices"]
        cw = obs["cw"]
        cogs = obs["cogs"]
        residual_value = obs["residual_value"]
        article_season_start = obs["article_season_start"]
        article_season_end = obs["article_season_end"]
        target_profit_ratio = game.target_profit_ratio
        penalty = game.profit_lack_penalty
        shipment_costs = obs["shipment_costs"]
        stocks = obs["stocks"]
        num_discounts = forecast_grid.shape[1]  # number of discounts
        discount_range = np.linspace(0, 0.7, num_discounts)

        model = pulp.LpProblem("Revenue_Optimization", pulp.LpMaximize)

        discounts = pulp.LpVariable.dicts(
            "discount",
            [
                (i, w, j)
                for i in range(num_products)
                for w in range(cw, WEEKS_YEAR)
                for j in range(num_discounts)
                if self.is_in_optimization(cw, w, article_season_start, article_season_end)[i]
            ],
            lowBound=0,
            cat="Binary",
        )
        sales_quantity = pulp.LpVariable.dicts(
            "sales_quantity",
            [
                (i, w, j)
                for i in range(num_products)
                for w in range(cw, WEEKS_YEAR)
                for j in range(num_discounts)
                if self.is_in_optimization(cw, w, article_season_start, article_season_end)[i]
            ],
            lowBound=0,
            cat="Continous",
        )
        stock_quantity = pulp.LpVariable.dicts(
            "stock_quantity",
            [(i, w) for i in range(num_products) for w in ([cw - 1] + list(range(cw, WEEKS_YEAR)))],
            lowBound=0,
            cat="Continous",
        )

        total_revenue = pulp.LpVariable("revenue", cat="Continuous")

        model += total_revenue == pulp.lpSum(
            [
                sales_quantity[(i, w, j)] * (black_prices[i] * (1 - discount_range[j]))
                for i in range(num_products)
                for w in range(cw, 52)
                for j in range(num_discounts)
                if self.is_in_optimization(cw, w, article_season_start, article_season_end)[i]
            ]
        )

        # add residual value
        residual_revenue = pulp.LpVariable("residual_revenue", cat="Continuous")

        model += residual_revenue == pulp.lpSum(
            [
                stock_quantity[(i, 51)] * residual_value[i]
                for i in range(num_products)
                if (cw >= article_season_start)[i]
            ]
        )

        # profit and revenue so far
        revenue_so_far = sum([r.sum() for r in game.revenues])

        profit_so_far = sum([r.sum() for r in game.profits])

        future_profit = pulp.LpVariable("profit", cat="Continuous")
        model += future_profit == pulp.lpSum(
            [
                sales_quantity[(i, w, j)] * (black_prices[i] * (1 - discount_range[j]) - cogs[i] - shipment_costs)
                for i in range(num_products)
                for w in range(cw, 52)
                for j in range(num_discounts)
                if self.is_in_optimization(cw, w, article_season_start, article_season_end)[i]
            ]
        )
        residual_profit = pulp.LpVariable("residual_profit", cat="Continuous")
        model += residual_profit == pulp.lpSum(
            [
                stock_quantity[(i, 51)] * (residual_value[i] - cogs[i])
                for i in range(num_products)
                if (cw >= article_season_start)[i]
            ]
        )

        profit_lack = pulp.LpVariable("profit_lack", lowBound=0.0, cat="Continuous")
        model += profit_lack >= target_profit_ratio * (total_revenue + residual_revenue + revenue_so_far) - (
            future_profit + residual_profit + profit_so_far
        )

        # objective function

        model += total_revenue + residual_revenue + revenue_so_far - penalty * profit_lack

        for i in range(num_products):
            model += stock_quantity[i, cw - 1] == stocks[i]
            for w in range(cw, 52):
                if self.is_in_optimization(cw, w, article_season_start, article_season_end)[i]:
                    model += pulp.lpSum([discounts[(i, w, j)] for j in range(num_discounts)]) == 1

                    for j in range(num_discounts):
                        model += forecast_grid[i][j] * discounts[(i, w, j)] >= sales_quantity[i, w, j]
                    model += stock_quantity[i, w - 1] >= pulp.lpSum(
                        sales_quantity[i, w, j] for j in range(num_discounts)
                    )
                    model += stock_quantity[i, w] == stock_quantity[i, w - 1] - pulp.lpSum(
                        sales_quantity[i, w, j] for j in range(num_discounts)
                    )
                else:
                    model += stock_quantity[i, w] == stock_quantity[i, w - 1]

        return model, discounts, sales_quantity, total_revenue, stock_quantity

    def act(self, game: PricingGame, obs, num_discounts=10) -> np.ndarray:
        cw = obs["cw"]

        if not self.forecast.trained:
            self.forecast.train(game)

        forecast_grid = self.forecast.create_forecast_grid(obs, num_discounts=num_discounts)
        num_products = forecast_grid.shape[0]
        num_discounts = forecast_grid.shape[1]
        article_season_start = obs["article_season_start"]
        article_season_end = obs["article_season_end"]
        if cw == 52:
            return np.zeros(num_products).astype(np.float32)

        model, discounts, sales, t_revenue, stock_quantity = self.create_optimization_model(game, obs, forecast_grid)
        solver = pulp.getSolver(
            "PULP_CBC_CMD",
            msg=self.verbose_solver,
            maxSeconds=self.timelimit,
            options=["gapRel 0.05", "gapAbs 50", f"timeLimit {self.timelimit}", "maxIts 1000"],
        )
        num_vars = len(model.variables())
        model.solve(solver)
        model.writeLP("my_lp_problem.lp")

        assert pulp.LpStatus[model.status] == "Optimal"
        objective = pulp.value(model.objective)
        print("objective", objective)
        print("left_stock", [pulp.value(stock_quantity[(i, 51)]) for i in range(num_products)])
        discount_range = np.linspace(0, 70, num_discounts)
        discount_values = np.zeros((forecast_grid.shape[0]))

        for i in range(num_products):
            if self.is_in_optimization(cw, cw, article_season_start, article_season_end)[i]:
                discount_values[i] = sum(
                    discount_range[j] * discounts[(i, cw, j)].varValue for j in range(num_discounts)
                )
            else:
                discount_values[i] = 0
        print("discounts", discount_values)
        return discount_values.astype(np.float32)
