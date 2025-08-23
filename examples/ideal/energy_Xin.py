import os
from datetime import datetime
import yaml
import matplotlib.pyplot as plt
import numpy as np

from double_pendulum.model.symbolic_plant import SymbolicDoublePendulum
from double_pendulum.model.model_parameters import model_parameters
from double_pendulum.simulation.simulation import Simulator
from double_pendulum.utils.plotting import plot_timeseries
from double_pendulum.utils.csv_trajectory import save_trajectory
from double_pendulum.controller.energy.energy_Xin import EnergyController
from double_pendulum.controller.energy.energy_Xin import EnergyLQRController


def wrap_angle(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi

# model parameters
design = "design_A.0"
model = "model_1.0"
robot = "acrobot"

torque_limit = [0.0, 15.0]
active_act = 1

model_par_path = "../../data/system_identification/identified_parameters/"+design+"/"+model+"/model_parameters.yml"
mpar = model_parameters()
mpar.load_yaml(model_par_path)
mpar.set_motor_inertia(0.)
mpar.set_damping([0., 0.])
mpar.set_cfric([0., 0.])
mpar.set_torque_limit(torque_limit)

# simulation parameters
integrator = "runge_kutta"
goal = [np.pi, 0., 0., 0.]
dt = 0.002
x0 = [0.1, 0.0, 0.0, 0.0]
t_final = 60.0

# controller parameters
kp = 2.35  # > 0.67
kd = 0.3  # > 0.022
kv = 0.3  # > 0.0

plant = SymbolicDoublePendulum(model_pars=mpar)
sim = Simulator(plant=plant)

controller = EnergyLQRController(model_pars=mpar)
controller.energy_controller.set_parameters(kp=kp, kd=kd, kv=kv)
controller.energy_controller.set_goal(goal)
#controller.energy_controller.check_parameters()
controller.energy_controller.init()

T, X, U = sim.simulate_and_animate(t0=0.0,
                                   x0=x0,
                                   tf=t_final,
                                   dt=dt,
                                   controller=controller,
                                   integrator=integrator,
                                   phase_plot=False,
                                   save_video=False)

# controller.save(path)
energy = controller.energy_controller.en
des_energy = controller.energy_controller.desired_energy

# saving and plotting
timestamp = datetime.today().strftime("%Y%m%d-%H%M%S")
save_dir = os.path.join("data", design, model, robot, "lqr", timestamp)
os.makedirs(save_dir)

save_trajectory(csv_path=os.path.join(save_dir, "trajectory.csv"),
                T=T,
                X=X,
                U=U)

mpar.save_dict(os.path.join(save_dir, "model_parameters.yml"))
controller.save(save_dir)

#U = np.array(U).reshape(-1, 1)


plot_timeseries(T=T, X=X, U=U, energy=energy,
                plot_energy=True,
                pos_y_lines=[-np.pi, np.pi],
                tau_y_lines=[-torque_limit[active_act], torque_limit[active_act]],
                energy_y_lines=[des_energy],
                save_to=os.path.join(save_dir, "time_series"))

final_state = X[-1]
# Wrap only angle components (theta1 and theta2)
theta1_err = wrap_angle(final_state[0]) - np.pi
theta2_err = wrap_angle(final_state[1])

# Velocity components (no wrapping needed)
theta1_dot_err = final_state[2] - goal[2]
theta2_dot_err = final_state[3] - goal[3]

# Error vector
error_vec = np.array([theta1_err, theta2_err, theta1_dot_err, theta2_dot_err])
error = np.linalg.norm(error_vec)

#print("Final error norm to goal:", error)

#plt.plot(controller.scores)
#plt.title("Sink Score Over Time")
##plt.axhline(0.04, color='red', linestyle='--', label='Sink Threshold')
#plt.legend()
#plt.show()



# par_dict = {
#             "dt": dt,
#             "t_final": t_final,
#             "integrator": integrator,
#             "start_pos1": x0[0],
#             "start_pos2": x0[1],
#             "start_vel1": x0[2],
#             "start_vel2": x0[3],
#             "goal_pos1": goal[0],
#             "goal_pos2": goal[1],
#             "goal_vel1": goal[2],
#             "goal_vel2": goal[3],
#             }
#
# with open(os.path.join(save_dir, "parameters.yml"), 'w') as f:
#     yaml.dump(par_dict, f)
