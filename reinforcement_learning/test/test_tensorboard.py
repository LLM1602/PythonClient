# from stable_baselines3 import A2C
#
# model = A2C('MlpPolicy', 'CartPole-v1', verbose=1, tensorboard_log="E:/tensorboard_store/RL/test_tensorboard/a2c_cartpole_tensorboard/")
# model.learn(total_timesteps=10000)
import numpy as np

from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback

model = SAC("MlpPolicy", "Pendulum-v0", tensorboard_log="E:/tensorboard_store/RL/test_tensorboard/sac/", verbose=1)

class TensorboardCallback(BaseCallback):
    """
    Custom callback for plotting additional values in tensorboard.
    """
    def __init__(self, verbose=0):
        super(TensorboardCallback, self).__init__(verbose)

    def _on_step(self) -> bool:
        # Log scalar value (here a random variable)
        value = np.random.random()
        self.logger.record('random_value', value)
        return True


model.learn(50000, callback=TensorboardCallback())