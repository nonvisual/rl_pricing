from agent.general_agent import GeneralAgent
from game.simulator import PricingGame
from game.state import GameState
import numpy as np


class RandomAgent(GeneralAgent):
    def act(self, game: PricingGame) -> np.ndarray:
        n = game.num_products
        return np.float32(np.random.choice([0, 5, 10, 15], n))
