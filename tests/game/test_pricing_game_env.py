import unittest
import gym
from hypothesis import given, reproduce_failure, settings
import numpy as np
from unittest.mock import MagicMock
from game.pricing_env import PricingGameEnv
import hypothesis.strategies as st


class PricingGameEnvTests(unittest.TestCase):
    @st.composite
    def generate_test_parameters(draw):
        num_products = draw(st.integers(min_value=1, max_value=10))
        num_discounts = draw(st.integers(min_value=1, max_value=10))
        # cw = draw(st.integers(min_value=1, max_value=52))

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
        stocks = draw(
            st.lists(st.integers(min_value=0, max_value=100000), min_size=num_products, max_size=num_products)
        )
        demand_generator = MagicMock()
        demand_generator.generate_demand.return_value = np.array(forecast_grid)
        sales = draw(st.lists(st.integers(min_value=0, max_value=100000), min_size=num_products, max_size=num_products))
        sales = np.clip(sales, 0, stocks).astype(int)
        demand_generator.compute_sale.return_value = sales
        seed = draw(st.integers(min_value=1, max_value=20000))
        env = PricingGameEnv(demand_generator=demand_generator, num_products=num_products)
        env.reset_game(seed)

        env.stocks = [np.array(stocks)]
        return env

    @given(generate_test_parameters())
    @settings(deadline=None, print_blob=True, max_examples=100)
    def test_article_online_status_false_no_sales_no_stock_change(self, env):
        num_products = env.num_products
        # Set the online status of all articles to False
        status = env.online_status[-1]
        stock = env.stocks[-1]

        # Perform the step with action as zeros
        obs, reward, done, info = env.step(np.zeros(num_products).astype(np.float32))
        np.testing.assert_array_equal(
            obs["stocks"][~status], stock[~status], "For offline articles stock should not change"
        )
        np.testing.assert_array_equal(
            obs["sales"][~status], np.zeros(num_products)[~status], "For offline articles sales should be 0"
        )
        assert (np.all(obs["stocks"]) >= 0, "Stock cannot be negative")
        assert (np.all(obs["revenues"] <= obs["profits"]), "Profit cannot be higher than revenue")

    @given(generate_test_parameters())
    @settings(deadline=None, print_blob=True, max_examples=100)
    def test_article_zero_price_not_positive_revenue(self, env):
        num_products = env.num_products
        # Set the online status of all articles to False
        status = env.online_status[-1]
        stock = env.stocks[-1]
        env.black_prices = np.zeros(num_products)

        # Perform the step with action as zeros
        obs, reward, done, info = env.step(np.zeros(num_products).astype(np.float32))
        assert (reward <= 0.0, "When Black Price is 0, revenue cannot be positive")

    @given(generate_test_parameters())
    @settings(deadline=None, print_blob=True, max_examples=100)
    def test_article_no_stock_no_revenue(self, env):
        num_products = env.num_products
        # Set the online status of all articles to False
        status = env.online_status[-1]
        is_no_stock = env.stocks[-1] == 0
        env.black_prices = np.zeros(num_products)

        # Perform the step with action as zeros
        obs, reward, done, info = env.step(np.zeros(num_products).astype(np.float32))
        assert (np.all(obs["revenues"][is_no_stock] == 0), "No revenue if no stock")


if __name__ == "__main__":
    unittest.main()
