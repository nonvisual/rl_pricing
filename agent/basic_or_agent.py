from agent.general_agent import GeneralAgent
from agent.optimization_model import add_profit_penalty_objective, create_base_optimization_model
from forecast.adaptive_forecast import AdaptiveForecast
from game.simulator import PricingGame
from game.state import GameState
import numpy as np
import pulp


class BasicORAgent(GeneralAgent):
    def __init__(self, forecast=AdaptiveForecast(), include_future_articles=True, verbose_solver=True, timelimit=10):
        self.forecast = forecast
        self.verbose_solver = verbose_solver
        self.timelimit = timelimit
        self.include_future_articles = include_future_articles

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

        model_dict = create_base_optimization_model(game, obs, forecast_grid, self.include_future_articles)
        add_profit_penalty_objective(model_dict, game)

        model = model_dict["model"]
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

        stock_quantity = model_dict["stocks"]
        discounts = model_dict["discounts"]
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
