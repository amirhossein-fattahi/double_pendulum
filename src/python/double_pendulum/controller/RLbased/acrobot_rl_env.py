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

        self.vel_limit1 = 30.0
        self.vel_limit2 = 40.0
        self.obs_clip = 5.0          # final safety clip for observation normalization
        self.nan_penalty = -50.0     # penalty when the step produces non-finite values


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

        # 1) Torque (squash + clip)
        torque = float(np.clip(action[0], -self.max_torque, self.max_torque))
        u = [0.0, torque]

        # 2) One-step integration in float64 (more stable), then convert back
        x = self.state.astype(np.float64, copy=False)
        try:
            x_dot = np.asarray(self.plant.rhs(0.0, x.tolist(), u), dtype=np.float64)
        except Exception:
            # If the plant throws, cut the episode
            obs = self._obs(self.state)
            return obs, self.nan_penalty, False, True, {"err": "rhs_failed"}

        x_next = x + self.dt * x_dot

        # 3) Wrap angles, clip velocities (avoid energy/velocity blow-ups)
        th1, th2, d1, d2 = x_next
        th1 = wrap_pi(th1)
        th2 = wrap_pi(th2)
        d1 = float(np.clip(d1, -self.vel_limit1, self.vel_limit1))
        d2 = float(np.clip(d2, -self.vel_limit2, self.vel_limit2))
        self.state = np.array([th1, th2, d1, d2], dtype=np.float32)
        self.step_count += 1

        # 4) Safety: if anything non-finite → truncate episode with penalty
        if not np.isfinite(self.state).all():
            obs = np.nan_to_num(self._obs(self.state), nan=0.0, posinf=0.0, neginf=0.0)
            return obs, self.nan_penalty, False, True, {"err": "non_finite_state"}

        # ---------- reward shaping ----------
        total = th1 + th2
        err = wrap_pi(total - np.pi)

        # Height progress term
        H = 0.5 * (1.0 - np.cos(total))       # [0, 1]
        H_prev = getattr(self, "_H_prev", H)
        dH = H - H_prev
        self._H_prev = H
        r_height_prog = 12.0 * dH

        # Stay in upper half
        r_upper_stay = 4.0 * np.clip(H - 0.5, 0.0, 0.5) * 2.0

        # Upright precision bump
        r_upright = 2.0 * np.exp(- (err / 0.25)**2)

        # Energy progress (compute with clipped velocities to avoid overflow)
        m1, m2 = self.plant.m[0], self.plant.m[1]
        l1, l2 = self.plant.l[0], self.plant.l[1]
        g = 9.81
        E_target = m1*g*l1 + m2*g*(l1 + l2)

        # Use clipped velocities d1, d2 (already clipped)
        Ep = m1*g*l1*(1 - np.cos(th1)) + m2*g*(l1*(1 - np.cos(th1)) + l2*(1 - np.cos(th1 + th2)))
        Ek = 0.5*m1*(l1*d1)**2 + 0.5*m2*((l1*d1)**2 + (l2*(d1 + d2))**2 + 2*l1*l2*d1*(d1 + d2)*np.cos(th2))
        E  = Ep + Ek
        E_prev = getattr(self, "_E_prev", E)
        dE = E - E_prev
        self._E_prev = E
        r_dE = 0.5 * np.sign(E_target - E) * (dE / (E_target + 1e-6))

        # Smoothness
        r_smooth = -0.0008*(d1**2 + d2**2) - 0.0005*(torque/self.max_torque)**2

        # NEW: very strong "stay still at upright" reward (dominates inside a tight cone)
        upright_zone = (abs(err) < 0.12)
        if upright_zone:
            # Quadratic penalty for residual velocities; weight high so PPO prefers to stop there
            r_stability = 4.0 - 0.08*(abs(d1) + abs(d2)) - 0.0005*(d1**2 + d2**2)
        else:
            r_stability = 0.0

        reward = float(r_height_prog + r_upper_stay + r_upright + r_dE + r_smooth + r_stability)

        # --------- success / truncation ----------
        # Make the capture condition tighter and longer to truly "park" there
        if upright_zone and abs(d1) < 0.35 and abs(d2) < 0.35:
            self.upright_counter += 1
            reward += 3.0   # per-step bonus while balanced
        else:
            self.upright_counter = 0

        terminated = (self.upright_counter >= 150)  # ~3 seconds at dt=0.02
        truncated  = (self.step_count >= self.max_steps)

        # 5) Final obs (safety clip so VecNormalize never sees huge numbers)
        obs = self._obs(self.state)
        obs = np.clip(obs, -self.obs_clip, self.obs_clip)
        if not np.isfinite(obs).all():
            obs = np.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)
            truncated = True
            reward += self.nan_penalty

        # temp
        if self.step_count == 1:
            print("first step -> terminated:", terminated, "truncated:", truncated)

        return obs, reward, terminated, truncated, {}



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
