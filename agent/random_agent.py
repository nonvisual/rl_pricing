from agent.general_agent import GeneralAgent
from game.simulator import PricingGame
from game.state import GameState
import numpy as np


class RandomAgent(GeneralAgent):
    def __init__(self, discount_options=[0, 5, 10, 15]):
        self.discount_options = discount_options

    def act(self, game: PricingGame) -> np.ndarray:
        n = game.num_products
        return np.float32(np.random.choice(self.discount_options, n))
