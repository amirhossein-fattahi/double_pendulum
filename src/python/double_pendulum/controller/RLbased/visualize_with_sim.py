import os
import numpy as np
from stable_baselines3 import PPO

# ─────── bring in your model-parameter loader ───────
from double_pendulum.model.model_parameters import model_parameters

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

# ─────── imports for sim + controller ───────
from double_pendulum.model.symbolic_plant       import SymbolicDoublePendulum
from double_pendulum.simulation.simulation      import Simulator
from double_pendulum.controller.RLbased.rl_controller import RLController

def main():
    # 1) load your PPO policy
    model      = PPO.load("ppo_acrobot")  
    max_torque = torque_limit[1]
    rl_ctrl    = RLController("./checkpoints/ppo_acrobot.zip", "./checkpoints/vecnormalize.pkl", max_torque=10.0)

    # 2) initial condition
    x0 = np.array([0.1, 0.0, 0.0, 0.0], dtype=float)

    # 3) build plant & simulator
    plant = SymbolicDoublePendulum(model_pars=mpar)
    sim   = Simulator(plant)

    # 4) run high-fidelity swing-up animation
    sim.simulate_and_animate(
        t0=0.0,
        x0=x0,
        tf=60.0,        # total sim time
        dt=0.002,       # step size
        controller=rl_ctrl,
        integrator="runge_kutta",
        phase_plot=False,
        save_video=False
    )

if __name__ == "__main__":
    main()
