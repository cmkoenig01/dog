import time

import mujoco
import mujoco.viewer

scene_path = "./unitree_a1/scene.xml"
m = mujoco.MjModel.from_xml_path(scene_path)
d = mujoco.MjData(m)

m.opt.timestep = .002
m.opt.impratio = 15

with mujoco.viewer.launch_passive(m, d) as viewer:
  start = time.time()
  while viewer.is_running():
    step_start = time.time()
    mujoco.mj_step(m, d)
    viewer.sync()
    time_until_next_step = m.opt.timestep - (time.time() - step_start)
    if time_until_next_step > 0:
      time.sleep(time_until_next_step)
