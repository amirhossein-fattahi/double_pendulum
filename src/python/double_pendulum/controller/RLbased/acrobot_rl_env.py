import gymnasium as gym
from gymnasium import spaces
import numpy as np
from double_pendulum.simulation.simulation import Simulator
from double_pendulum.model.model_parameters import model_parameters
from double_pendulum.model.plant import DoublePendulumPlant
from double_pendulum.utils.plotting import plot_timeseries




class AcrobotRLEnv(gym.Env):
    def __init__(self):
        super(AcrobotRLEnv, self).__init__()

        # --- Load model parameters
        self.model_par = model_parameters(filepath="../../data/system_identification/identified_parameters/design_A.0/model_1.0/model_parameters.yml")
        self.model_par.set_motor_inertia(0.0)
        self.model_par.set_cfric([0.0, 0.0])
        self.model_par.set_damping([0.0, 0.0])
        

        # --- Set up plant and simulator
        self.plant = DoublePendulumPlant(model_pars=self.model_par)
        self.simulator = Simulator(plant=self.plant)

        # --- Action space (torque on 2nd joint only)
        self.max_torque = 10.0
        self.action_space = spaces.Box(-self.max_torque, self.max_torque, shape=(1,), dtype=np.float32)

        self.max_steps = 1000

        self.dt = 0.02
        self.prev_E_err = None

            # --- Internal state
        self.state = None
        self.dt = 0.02
        self.max_steps = 1000
        self.step_count = 0

        # --- Observation space: [theta1, theta2, theta1_dot, theta2_dot]
        # update observation_space to match 6-dim features
        high = np.array([1,1,1,1,30.0,40.0], dtype=np.float32)
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)

    def _obs(self, x):
        th1, th2, d1, d2 = x
        return np.array([np.sin(th1), np.cos(th1),
                        np.sin(th2), np.cos(th2),
                        d1, d2], dtype=np.float32)

    

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            np.random.seed(seed)
        theta1 = np.random.uniform(-0.5, 0.5)
        theta2 = np.random.uniform(-1.0, 1.0)
        self.state = np.array([theta1, theta2, 0.0, 0.0], dtype=np.float32)
        self.prev_E_err = None
        self.step_count = 0
        self.upright_counter = 0
        self.mode = "swing"          # swing-up until we capture near upright
        self.capture_count = 0
        return self._obs(self.state), {}


    def step(self, action):
        def wrap_pi(a):
            return (a + np.pi) % (2*np.pi) - np.pi

        # --- torque + integration
        torque = float(np.clip(action[0], -self.max_torque, self.max_torque))
        u = [0.0, torque]
        x = self.state
        x_dot = self.plant.rhs(0.0, x.tolist(), u)
        x_dot = np.array(x_dot, dtype=np.float32)
        x_next = x + self.dt * x_dot
        self.state = x_next
        self.step_count += 1

        # --- symbols (clipped velocities for reward stability)
        th1, th2, d1, d2 = map(float, x_next)
        th1, th2 = wrap_pi(th1), wrap_pi(th2)
        total = th1 + th2
        err   = wrap_pi(total - np.pi)
        d1c, d2c = np.clip(d1, -20.0, 20.0), np.clip(d2, -20.0, 20.0)
        vtop = d1c + d2c                         # angular speed around the top
        H = 0.5 * (1.0 - np.cos(total))          # height proxy in [0,1]
        in_upper = (H > 0.5)
        upright_zone = (abs(err) < 0.15)

        # --- energy (used lightly; not to dominate reward)
        m1, m2 = self.plant.m[0], self.plant.m[1]
        l1, l2 = self.plant.l[0], self.plant.l[1]
        g = 9.81
        E_target = m1*g*l1 + m2*g*(l1 + l2)
        Ep = m1*g*l1*(1 - np.cos(th1)) + m2*g*(l1*(1 - np.cos(th1)) + l2*(1 - np.cos(th1+th2)))
        Ek = 0.5*m1*(l1*d1c)**2 + 0.5*m2*((l1*d1c)**2 + (l2*(d1c+d2c))**2 + 2*l1*l2*d1c*(d1c+d2c)*np.cos(th2))
        E = Ep + Ek
        E_prev = getattr(self, "_E_prev", E)
        dE = E - E_prev
        self._E_prev = E

        # --- init mode flags
        if not hasattr(self, "mode"):
            self.mode = "swing"
            self.capture_count = 0

        # --- capture logic: switch to BALANCE after brief calm near top
        capture_ok = (H > 0.5 and abs(err) < 0.5 and abs(vtop) < 2.5)
        self.capture_count = (self.capture_count + 1) if capture_ok else 0
        if self.mode == "swing" and self.capture_count >= 5:
            self.mode = "balance"

        ### REWARD
        if self.mode == "swing":
            # Height progress (big driver to escape bottom)
            H_prev = getattr(self, "_H_prev", H)
            dH = H - H_prev
            self._H_prev = H
            r_height_prog = 5.0 * H

            # Reward being in upper half (keeps going once it gets there)
            r_upper = 2.0 * np.clip(H - 0.5, 0.0, 0.5) * 2.0

            # Energy pumping term: reward speed when it helps move away from bottom.
            # |sin(total)|≈0 near top -> no pumping there; large on the sides.
            r_pump = 0.8 * abs(vtop) * abs(np.sin(total))

            # Very light use of dE toward E_target (normalized)
            r_dE = 0.3 * np.sign(E_target - E) * (dE / (E_target + 1e-6))

            # Small smoothness penalty (don’t over‑penalize here)
            r_sm = -0.0005*(d1c**2 + d2c**2) - 0.0002*(torque/self.max_torque)**2

            reward = float(r_height_prog + r_upper + r_pump + r_dE + r_sm)

        else:
            ### BALANCE MODE 
            Q_err = 50.0
            Q_d1  = 6.0
            Q_d2  = 6.0
            R_u   = 0.002

            cost = Q_err*(err**2) + Q_d1*(d1c**2) + Q_d2*(d2c**2) + R_u*((torque/self.max_torque)**2)
            r_hold = 10.0 - cost
            r_no_orbit = -0.15 * abs(vtop)

            reward = float(r_hold + r_no_orbit)

            # If it drifts away, go back to swing‑up
            if (H < 0.4) or (abs(err) > 0.7):
                self.mode = "swing"
                self.capture_count = 0

        # Success bonus for sustained quiet upright
        if upright_zone and abs(d1c) < 0.4 and abs(d2c) < 0.4:
            self.upright_counter += 1
            if self.upright_counter==10:
                reward += 100
            reward += 2.0
        else:
            self.upright_counter = 0

        terminated = (self.upright_counter >= 20)
        truncated  = (self.step_count >= self.max_steps)

        if self.step_count == 1:
            print("first step -> terminated:", terminated, "truncated:", truncated)

        return self._obs(self.state), float(reward), terminated, truncated, {}


    def _compute_reward(self, state, torque):
        # Reward is higher when pendulum is upright and velocities are low
        theta1, theta2, dtheta1, dtheta2 = np.array(state).flatten()
        # Upright position = (theta1 + theta2) = pi
        uprightness = np.cos(theta1 + theta2 - np.pi)

        reward = uprightness - 0.01 * (dtheta1 ** 2 + dtheta2 ** 2) - 0.001 * torque ** 2
        return reward

    def _is_done(self, state):
        # Done if steps exceed max
        if self.step_count >= self.max_steps:
            return True
        return False

    def render(self, mode='human'):
        pass
