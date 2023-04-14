import unittest
from math import isclose
import numpy as np
from unittest.mock import MagicMock
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from forecast.adaptive_forecast import AdaptiveForecast


class TestAdaptiveForecast(unittest.TestCase):
    def setUp(self):
        self.env = MagicMock(num_products=2)
        self.forecast = AdaptiveForecast()

    def test_create_forecast_grid_base_demand(self):
        obs = {"sales": [80.0, 110.0], "discounts": [0.0, 0.0]}
        self.forecast.train(self.env)
        self.forecast.create_forecast_grid(obs)
        expected_base_demand = [80.0, 110.0]
        expected_elasticity = [-3.0, -3.0]
        np.testing.assert_allclose(self.forecast.base_demand, expected_base_demand, rtol=1e-4)
        np.testing.assert_allclose(self.forecast.elasticity, expected_elasticity, rtol=1e-4)
        forecast_grid = self.forecast.create_forecast_grid(obs)
        assert forecast_grid.shape == (2, 14)
