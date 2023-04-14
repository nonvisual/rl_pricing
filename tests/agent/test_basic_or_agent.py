from hypothesis import given, reproduce_failure, settings, assume
import hypothesis.strategies as st
import numpy as np
import pulp
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from agent.basic_or_agent import BasicORAgent
from unittest.mock import MagicMock

from game.simulator import PricingGame


class TestBasicORAgent:
    @st.composite
    def generate_test_parameters(draw):
        num_products = draw(st.integers(min_value=1, max_value=10))
        num_discounts = draw(st.integers(min_value=1, max_value=10))
        cw = draw(st.integers(min_value=1, max_value=52))

        forecast_grid = [
            [
                draw(
                    st.lists(
                        st.floats(allow_nan=False, allow_infinity=False, min_value=0, max_value=100000),
                        min_size=num_products,
                        max_size=num_products,
                    )
                )
                for _ in range(num_discounts)
            ]
            for _ in range(num_products)
        ]
        stocks = draw(st.lists(st.integers(min_value=0), min_size=num_products, max_size=num_products))

        return num_products, num_discounts, cw, stocks, forecast_grid

    @given(generate_test_parameters())
    @settings(deadline=None, print_blob=True)
    def test_create_optimization_model(self, data):
        num_products, num_discounts, cw, stocks, forecast_grid = [d for d in data]
        stocks = np.array(stocks)
        forecast_grid = np.array(forecast_grid)
        assume(len(stocks) == num_products)
        black_prices = np.ones(num_products)
        cogs = np.ones(num_products)
        residual_value = np.ones(num_products)
        article_season_start = np.zeros(num_products)
        article_season_end = np.ones(num_products)
        shipment_costs = np.zeros(num_products)

        agent = BasicORAgent()
        demand_generator = MagicMock()
        demand_generator.generate_demand.return_value = np.array(forecast_grid)

        game = PricingGame(demand_generator, num_products)
        obs = {
            "black_prices": black_prices,
            "cw": cw,
            "cogs": cogs,
            "residual_value": residual_value,
            "article_season_start": article_season_start,
            "article_season_end": article_season_end,
            "shipment_costs": shipment_costs,
            "stocks": stocks,
        }

        model, discounts, sales_quantity, total_revenue = agent.create_optimization_model(game, obs, forecast_grid)

        model.solve()

        assert model.status == 1  # Optimal

        # Check that the model is of the correct type
        assert isinstance(model, pulp.LpProblem)

        # Check that the discount variables are within the correct range
        for i in range(num_products):
            for w in range(cw, 52):
                for j in range(num_discounts):
                    assert 0 <= discounts[(i, w, j)].varValue <= 1

        # Check that the sales and stock variables are positive
        for i in range(num_products):
            for w in range(cw, 52):
                assert sales_quantity[(i, w)].varValue >= 0

        # Check that the total revenue is non-negative
        assert total_revenue.varValue >= 0
