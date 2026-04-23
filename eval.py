import os
import time
import mujoco
import mujoco.viewer
import numpy as np
from mj_utils import *
from PPO import PPO

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

ACTION_SCALE   = 0.25
RESET_EVERY_S  = 20.0
CIRCLE_RADIUS  = 3.0
GOAL_RADIUS    = 0.4

NAV_STATE_DIM  = 15
WALK_STATE_DIM = 34

HOME_CTRL = np.array([0, 0.9, -1.8, 0, 0.9, -1.8, 0, 0.9, -1.8, 0, 0.9, -1.8])

scene_path = "unitree_a1/scene.xml"
m = mujoco.MjModel.from_xml_path(scene_path)
d = mujoco.MjData(m)
mujoco.mj_resetDataKeyframe(m, d, 0)

ctr_inds = get_ctrl_indices(m, ACTUATOR_NAMES)

goal_body_id  = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "goal_marker")
goal_mocap_id = m.body_mocapid[goal_body_id]

def random_goal():
    angle = np.random.uniform(0, 2 * np.pi)
    return np.array([CIRCLE_RADIUS * np.cos(angle), CIRCLE_RADIUS * np.sin(angle)])

def set_goal(goal_pos):
    d.mocap_pos[goal_mocap_id, 0] = goal_pos[0]
    d.mocap_pos[goal_mocap_id, 1] = goal_pos[1]

nav_ppo  = PPO(state_dim=NAV_STATE_DIM,  action_dim=2,  hidden_dim=128, critic_hidden_dim=256)
walk_ppo = PPO(state_dim=WALK_STATE_DIM, action_dim=12, hidden_dim=256, critic_hidden_dim=512)

nav_ckpt  = "nav_checkpoint_best.pt"  if os.path.exists("nav_checkpoint_best.pt")  else "nav_checkpoint.pt"
walk_ckpt = "walk_checkpoint_best.pt" if os.path.exists("walk_checkpoint_best.pt") else "walk_checkpoint.pt"
nav_ppo.load(nav_ckpt)
walk_ppo.load(walk_ckpt)
nav_ppo.actor.eval()
nav_ppo.critic.eval()
walk_ppo.actor.eval()
walk_ppo.critic.eval()

def get_nav_state(d, goal_pos):
    robot_xy     = d.xpos[1][:2]
    to_goal      = goal_pos - robot_xy
    dist_to_goal = np.linalg.norm(to_goal)
    dx_norm      = np.clip(to_goal[0] / (CIRCLE_RADIUS * 2), -1.0, 1.0)
    dy_norm      = np.clip(to_goal[1] / (CIRCLE_RADIUS * 2), -1.0, 1.0)
    dist_norm    = np.clip(dist_to_goal / (CIRCLE_RADIUS * 2), 0.0, 1.0)

    qw, qx, qy, qz = d.qpos[3], d.qpos[4], d.qpos[5], d.qpos[6]
    robot_yaw  = np.arctan2(2*(qw*qz + qx*qy), 1 - 2*(qy*qy + qz*qz))
    goal_angle = np.arctan2(to_goal[1], to_goal[0] + 1e-6)
    angle_err  = goal_angle - robot_yaw

    return np.concatenate([
        d.qpos[3:7],
        d.qvel[3:6],
        [d.xpos[1][2]],
        [dx_norm],
        [dy_norm],
        [dist_norm],
        [np.cos(angle_err)],
        [np.sin(angle_err)],
        [np.cos(robot_yaw)],
        [np.sin(robot_yaw)],
    ])

def get_walk_state(d, nav_cmd):
    return np.concatenate([
        d.qpos[3:7],
        d.qvel[3:6],
        d.qpos[7:19],
        d.qvel[6:18],
        [d.xpos[1][2]],
        nav_cmd,
    ])

def _is_fallen(d):
    return d.qpos[3] < 0.65 or d.xpos[1][2] < 0.15

def reset(goal_pos):
    mujoco.mj_resetDataKeyframe(m, d, 0)
    set_goal(goal_pos)

goal_pos   = random_goal()
set_goal(goal_pos)
ep         = 0
goals_hit  = 0
reset_time = time.time()

print(f"=== EVAL MODE — auto-reset every {RESET_EVERY_S:.0f}s ===")

with mujoco.viewer.launch_passive(m, d) as viewer:
    while viewer.is_running():
        step_start = time.time()

        nav_state     = get_nav_state(d, goal_pos)
        nav_cmd, _, _ = nav_ppo.get_action(nav_state, deterministic=True)

        walk_state    = get_walk_state(d, nav_cmd)
        action, _, _  = walk_ppo.get_action(walk_state, deterministic=True)

        d.ctrl[ctr_inds] = HOME_CTRL + action * ACTION_SCALE

        mujoco.mj_step(m, d)
        viewer.sync()

        steps_per_print = max(1, int(1.0 / m.opt.timestep))
        if int(d.time / m.opt.timestep) % steps_per_print == 0:
            robot_xy     = d.xpos[1][:2]
            dist_to_goal = np.linalg.norm(goal_pos - robot_xy)
            print(f"ep {ep:3d} | t={d.time:.1f}s | dist={dist_to_goal:.2f}m | h={d.xpos[1][2]:.3f}m")

        if _is_fallen(d):
            ep += 1
            print(f"--- fell (ep {ep}) ---")
            goal_pos   = random_goal()
            reset(goal_pos)
            reset_time = time.time()
            continue

        robot_xy = d.xpos[1][:2]
        if np.linalg.norm(goal_pos - robot_xy) < GOAL_RADIUS:
            goals_hit += 1
            print(f"*** GOAL {goals_hit} REACHED — resetting ***")
            goal_pos   = random_goal()
            reset(goal_pos)
            reset_time = time.time()
        elif time.time() - reset_time >= RESET_EVERY_S:
            ep        += 1
            goal_pos   = random_goal()
            reset(goal_pos)
            reset_time = time.time()
            print(f"--- timeout reset (ep {ep}) ---")

        elapsed   = time.time() - step_start
        remaining = m.opt.timestep - elapsed
        if remaining > 0:
            time.sleep(remaining)
