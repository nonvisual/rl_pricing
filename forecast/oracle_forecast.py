from forecast.base import GeneralForecast


class OracleForecast(GeneralForecast):
    def __init__(self):
        self.trained = False

    def train(self, env):
        self.env = env

        self.obs = []
        self.trained = True
        pass

    def create_forecast_grid(self, obs, num_discounts=10):
        cw = obs["cw"]
        return self.env.demand_generator.noiseless_oracle_forecast(current_cw=cw, num_discounts=num_discounts)
