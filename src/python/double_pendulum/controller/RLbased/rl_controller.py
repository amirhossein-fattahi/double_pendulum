# src/python/double_pendulum/controller/RLbased/rl_controller.py
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from double_pendulum.controller.RLbased.acrobot_rl_env import AcrobotRLEnv
import numpy as np

class RLController:
    """
    Wraps a Stable-Baselines3 model so that
    simulate_and_animate can call get_control_output(x, t).
    """
    def __init__(self, model_path, vecnorm_path, max_torque=10.0):
        self.model = PPO.load(model_path)
        self.max_torque = max_torque
        # load VecNormalize stats and freeze them
        self.vecnorm = VecNormalize.load(vecnorm_path, DummyVecEnv([lambda: AcrobotRLEnv()]))
        self.vecnorm.training = False
        self.vecnorm.norm_reward = False

    @staticmethod
    def _obs_from_state(x):
        th1, th2, d1, d2 = x
        return np.array([np.sin(th1), np.cos(th1), np.sin(th2), np.cos(th2), d1, d2], dtype=np.float32)

    def get_control_output(self, x, t):
        obs = self._obs_from_state(x)
        obs = self.vecnorm.normalize_obs(obs[None, :])  # shape (1,6)
        act, _ = self.model.predict(obs, deterministic=True)
        tau = float(np.clip(act[0], -self.max_torque, self.max_torque))
        return [0.0, tau]
