import time

import mujoco
import mujoco.viewer

from mj_utils import *

scene_path = "./unitree_a1/scene.xml"
m = mujoco.MjModel.from_xml_path(scene_path)
d = mujoco.MjData(m)

m.opt.timestep = .002
m.opt.impratio = 15

BODY_NAMES = (
    "trunk",
    "FR_hip", "FR_thigh", "FR_calf",
    "FL_hip", "FL_thigh", "FL_calf",
    "RR_hip", "RR_thigh", "RR_calf",
    "RL_hip", "RL_thigh", "RL_calf",
)

JOINT_NAMES = (
    "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
    "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
    "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
    "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
)

ACTUATOR_NAMES = (
    "FR_hip", "FR_thigh", "FR_calf",
    "FL_hip", "FL_thigh", "FL_calf",
    "RR_hip", "RR_thigh", "RR_calf",
    "RL_hip", "RL_thigh", "RL_calf",
)


joint_inds = get_qpos_indices(m, JOINT_NAMES)
ctrl_inds = get_ctrl_indices(m, ACTUATOR_NAMES)

t_start = time.time()
with mujoco.viewer.launch_passive(m, d) as viewer:
  start = time.time()
  while viewer.is_running():

    step_start = time.time()

    theta = np.sin((time.time() - t_start)*5)
    d.qpos[joint_inds] = theta

    mujoco.mj_step1(m, d)
    viewer.sync()
    time_until_next_step = m.opt.timestep - (time.time() - step_start)
    if time_until_next_step > 0:
      time.sleep(time_until_next_step)
