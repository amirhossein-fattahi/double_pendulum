import gymnasium as gym
import os
import numpy as np
import torch

from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.callbacks import EvalCallback
from acrobot_rl_env import AcrobotRLEnv
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize


def main():
    def make_env():
        e = AcrobotRLEnv()
        e = gym.wrappers.TimeLimit(e, max_episode_steps=1000)
        return e

    train_env = VecNormalize(DummyVecEnv([make_env]*8), norm_obs=True, norm_reward=True, clip_obs=10.0)
    eval_env  = VecNormalize(DummyVecEnv([make_env]*8), norm_obs=True, norm_reward=False, clip_obs=10.0)

    # Optional: check if env follows Gym API
    #check_env(env, warn=True)

    tb_log = "./ppo_acrobot_tensorboard"
    #model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="./ppo_acrobot_tensorboard/", 
    #            learning_rate=3e-4, n_steps=2048, batch_size=64, gamma=0.99)

    policy_kwargs = dict(net_arch=dict(pi=[256,256], vf=[256,256]))

    model = PPO(
        "MlpPolicy",
        train_env,
        verbose=1,
        tensorboard_log=tb_log,
        # ← key hyperparams tuned for continuous control
        n_steps=4096,
        batch_size=512,
        n_epochs=20,
        gamma=0.995,
        gae_lambda=0.98,
        clip_range=0.1,
        learning_rate=3e-4, # constant (no linear decay)
        ent_coef=0.1,
        vf_coef=0.5,
        use_sde=True,
        sde_sample_freq=4,
        max_grad_norm=0.5,
        policy_kwargs=policy_kwargs
    )
    # evaluation env (no VecNormalize, just raw)
    #eval_env = AcrobotRLEnv()
    # now pass eval_env into EvalCallback
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path="./logs/best_model",
        log_path="./logs/eval",
        eval_freq=5_000,
        n_eval_episodes=5,
        deterministic=True,
    )
    checkpoint_callback = CheckpointCallback(save_freq=5000, save_path='./checkpoints/', name_prefix='ppo_acrobot')

    model.learn(
        total_timesteps=1000_000,
        callback=[checkpoint_callback, eval_callback]
    )

    train_env.save("./checkpoints/vecnormalize.pkl")
    model.save("./checkpoints/ppo_acrobot.zip")
    #env.save("ppo_acrobot_vecnormalize")


    print("Training complete and model saved!")


        # ─── after training & saving ───
    # 1) build the same mpar as in visualize_with_sim
    from double_pendulum.model.model_parameters import model_parameters
    from double_pendulum.model.symbolic_plant       import SymbolicDoublePendulum
    from double_pendulum.simulation.simulation      import Simulator
    from double_pendulum.controller.RLbased.rl_controller import RLController

    # ─────── rebuild exactly the same mpar as Energy-Xin does ───────
    design      = "design_A.0"
    model_name  = "model_1.0"
    robot       = "acrobot"
    torque_limit = [0.0, 10.0]

    # adjust this path if your working directory is different
    model_par_path = os.path.join(
        os.path.dirname(__file__),
        "..", "..", "..", "..", "..", # back out from .../controller/RLbased
        "data", "system_identification", "identified_parameters",
        design, model_name,
        "model_parameters.yml"
    )

    mpar = model_parameters()
    mpar.load_yaml(model_par_path)
    mpar.set_motor_inertia(0.0)
    mpar.set_damping([0.0, 0.0])
    mpar.set_cfric([0.0, 0.0])
    mpar.set_torque_limit(torque_limit)

    # 2) wrap your trained model
    rl_ctrl = RLController("./checkpoints/ppo_acrobot.zip", "./checkpoints/vecnormalize.pkl", max_torque=10.0)

    # 2) initial condition
    x0 = np.array([0.1, 0.0, 0.0, 0.0], dtype=float)

    # 3) instantiate plant+sim
    plant = SymbolicDoublePendulum(model_pars=mpar)
    sim   = Simulator(plant)

    # 4) run the 60 s animation
    sim.simulate_and_animate(
        t0=0.0,
        x0=x0,            # reuse the same x0 you used in training
        tf=30.0,          # 60-second cap
        dt=0.02,            # from your training script
        controller=rl_ctrl,
        integrator="runge_kutta",
        phase_plot=False,
        save_video=False
    )
    


if __name__ == "__main__":
    main()