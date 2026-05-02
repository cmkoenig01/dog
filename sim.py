import os
import mujoco
import numpy as np
import torch
from mj_utils import *
from PPO import PPO, RolloutBuffer

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

STEPS_PER_UPDATE = 2048
ACTION_SCALE     = 0.25
MAX_EP_LEN       = 7500
N_ENVS           = 16

CIRCLE_RADIUS  = 3.0
GOAL_RADIUS    = 0.4
STAND_STEPS    = 100

NAV_STATE_DIM  = 15
WALK_STATE_DIM = 34  # 32 body features (4 orient + 3 ang_vel + 12 jpos + 12 jvel + 1 height) + 2 nav command

HOME_CTRL = np.array([0, 0.9, -1.8, 0, 0.9, -1.8, 0, 0.9, -1.8, 0, 0.9, -1.8])

scene_path = "unitree_a1/scene.xml"

envs_m = [mujoco.MjModel.from_xml_path(scene_path) for _ in range(N_ENVS)]
envs_d = [mujoco.MjData(em) for em in envs_m]
for i in range(N_ENVS):
    mujoco.mj_resetDataKeyframe(envs_m[i], envs_d[i], 0)
m = envs_m[0]

joint_inds = get_qpos_indices(m, JOINT_NAMES)
ctr_inds   = get_ctrl_indices(m, ACTUATOR_NAMES)

FOOT_BODY_NAMES = ("FR_calf", "FL_calf", "RR_calf", "RL_calf")
foot_body_id_to_idx = {m.body(name).id: i for i, name in enumerate(FOOT_BODY_NAMES)}
foot_body_ids       = [m.body(name).id for name in FOOT_BODY_NAMES]

goal_body_id  = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "goal_marker")
goal_mocap_id = m.body_mocapid[goal_body_id]

def random_goal():
    angle = np.random.uniform(0, 2 * np.pi)
    return np.array([CIRCLE_RADIUS * np.cos(angle), CIRCLE_RADIUS * np.sin(angle)])

def set_goal(d, goal_pos):
    d.mocap_pos[goal_mocap_id, 0] = goal_pos[0]
    d.mocap_pos[goal_mocap_id, 1] = goal_pos[1]

def get_foot_contacts(d):
    contacts = np.zeros(4, dtype=bool)
    for i in range(d.ncon):
        c  = d.contact[i]
        b1 = m.geom(c.geom1).bodyid[0]
        b2 = m.geom(c.geom2).bodyid[0]
        if b1 in foot_body_id_to_idx:
            contacts[foot_body_id_to_idx[b1]] = True
        if b2 in foot_body_id_to_idx:
            contacts[foot_body_id_to_idx[b2]] = True
    return contacts

# Nav state: 15D — orientation, ang_vel, height, goal direction, heading error, yaw
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
        d.qpos[3:7],           # orientation      — 4
        d.qvel[3:6],           # angular vel       — 3
        [d.xpos[1][2]],        # body height       — 1
        [dx_norm],             # dx to goal        — 1
        [dy_norm],             # dy to goal        — 1
        [dist_norm],           # dist to goal      — 1
        [np.cos(angle_err)],   # heading error cos — 1
        [np.sin(angle_err)],   # heading error sin — 1
        [np.cos(robot_yaw)],   # robot yaw cos     — 1
        [np.sin(robot_yaw)],   # robot yaw sin     — 1
    ])  # total: 15

# Walk state: 34D — body state + nav command from navigation policy
def get_walk_state(d, nav_cmd):
    return np.concatenate([
        d.qpos[3:7],           # orientation  — 4
        d.qvel[3:6],           # angular vel  — 3
        d.qpos[7:19],          # joint pos    — 12
        d.qvel[6:18],          # joint vel    — 12
        [d.xpos[1][2]],        # body height  — 1
        nav_cmd,               # nav command  — 2
    ])  # total: 34

def _is_fallen(d):
    qx, qy = d.qpos[4], d.qpos[5]
    upright_z = 1.0 - 2.0 * (qx**2 + qy**2)
    return upright_z < 0.5 or d.xpos[1][2] < 0.15

def compute_nav_reward(d, goal_pos, at_goal):
    robot_xy     = d.xpos[1][:2]
    to_goal      = goal_pos - robot_xy
    dist_to_goal = float(np.linalg.norm(to_goal))
    goal_dir     = to_goal / (dist_to_goal + 1e-6)
    body_speed   = float(np.linalg.norm(d.qvel[:2]))

    qw, qx, qy, qz = d.qpos[3], d.qpos[4], d.qpos[5], d.qpos[6]
    robot_yaw     = np.arctan2(2*(qw*qz + qx*qy), 1 - 2*(qy*qy + qz*qz))
    facing_dir    = np.array([np.cos(robot_yaw), np.sin(robot_yaw)])
    heading_align = float(np.dot(facing_dir, goal_dir))
    goal_vel      = float(np.dot(d.qvel[:2], goal_dir))
    perp_dir      = np.array([-goal_dir[1], goal_dir[0]])
    lateral_vel   = float(np.dot(d.qvel[:2], perp_dir))
    yaw_signed    = float(d.qvel[5])
    yaw_rate      = abs(yaw_signed)
    goal_angle    = np.arctan2(to_goal[1], to_goal[0] + 1e-6)
    angle_err     = (goal_angle - robot_yaw + np.pi) % (2 * np.pi) - np.pi

    if at_goal:
        still_reward = 4.0 * max(0.0, 1.0 - body_speed / 0.3)
        return still_reward + 50.0

    TARGET_SPEED  = 0.5
    ALIGN_THRESH  = 0.7  # heading_align threshold (~45 degrees)

    facing_reward = 3.0 * max(0.0, heading_align) ** 2
    err_norm      = float(np.clip(angle_err / np.pi, -1.0, 1.0))
    desired_yaw   = err_norm * 1.0
    dist_factor   = min(1.0, dist_to_goal / 0.6)
    goal_bonus    = 50.0 * float(dist_to_goal < GOAL_RADIUS)

    if heading_align >= ALIGN_THRESH:
        # ALIGNED: reward forward speed, gentle correction only
        goal_vel_reward = 5.0 * min(max(0.0, goal_vel), TARGET_SPEED) * dist_factor
        turn_reward     = 2.0 * desired_yaw * float(np.tanh(yaw_signed))
        fast_fwd_pen    = 0.0
    else:
        # MISALIGNED: reward turning hard, allow only crawl speed forward
        goal_vel_reward = 1.0 * min(max(0.0, goal_vel), 0.15)
        turn_reward     = 7.0 * desired_yaw * float(np.tanh(yaw_signed))
        fast_fwd_pen    = 6.0 * max(0.0, goal_vel - 0.15)

    # Hysteresis proxy: being close but misaligned = massive penalty
    misalign_mag          = min(1.0, max(0.0, ALIGN_THRESH - heading_align))
    proximity_misalign_pen = 12.0 * max(0.0, 1.0 - dist_to_goal / 1.5) * misalign_mag

    away_pen         = (8.0 + 8.0 * max(0.0, heading_align)) * max(0.0, -goal_vel)
    stationary_pen   = 2.0 * float(body_speed < 0.05) * max(0.0, 1.0 - abs(err_norm))
    lateral_pen      = 4.0 * lateral_vel ** 2
    speed_excess_pen = 3.0 * max(0.0, goal_vel - TARGET_SPEED)
    fast_spin_pen    = 2.0 * max(0.0, yaw_rate - 1.2)

    return (facing_reward + turn_reward + goal_vel_reward + goal_bonus
            - away_pen - stationary_pen - lateral_pen - speed_excess_pen
            - fast_spin_pen - fast_fwd_pen - proximity_misalign_pen)

def compute_walk_reward(d, action, prev_action, at_goal, nav_cmd):
    foot_contacts = get_foot_contacts(d)
    fr, fl, rr, rl = foot_contacts
    n_contacts  = int(np.sum(foot_contacts))
    trot_active = bool((fr == rl) and (fl == rr) and (fr != fl))
    hop_active  = bool((fr == fl) and (rr == rl) and (fr != rr))

    body_speed   = float(np.linalg.norm(d.qvel[:2]))
    qw, qx, qy, qz = d.qpos[3], d.qpos[4], d.qpos[5], d.qpos[6]
    robot_yaw    = np.arctan2(2*(qw*qz + qx*qy), 1 - 2*(qy*qy + qz*qz))
    facing_dir   = np.array([np.cos(robot_yaw), np.sin(robot_yaw)])
    perp_dir     = np.array([-facing_dir[1], facing_dir[0]])
    body_fwd_vel = float(np.dot(d.qvel[:2], facing_dir))
    body_lat_vel = float(np.dot(d.qvel[:2], perp_dir))
    tilt         = d.qpos[4] ** 2 + d.qpos[5] ** 2
    fwd_cmd      = float(nav_cmd[0])  # [-1, 1] from nav policy Tanh

    TARGET_SPEED = 0.5

    if at_goal:
        still_reward = 4.0 * max(0.0, 1.0 - body_speed / 0.3)
        upright      = 0.5 * d.qpos[3]
        height_pen   = 2.0 * max(0.0, 0.28 - d.xpos[1][2])
        torque_pen   = 0.001 * np.sum(action ** 2)
        return still_reward + upright - height_pen - torque_pen

    foot_heights     = [d.xpos[bid][2] for bid in foot_body_ids]
    clearance_reward = 0.4 * sum(
        max(0.0, h - 0.04) for h, c in zip(foot_heights, foot_contacts) if not c
    )
    turn_cmd          = float(nav_cmd[1])
    actual_yaw_rate   = float(d.qvel[5])

    # Diagonal pair sync — FR+RL and FL+RR thigh/calf should match throughout trot
    fr_tc = d.qpos[joint_inds[[1, 2]]]
    rl_tc = d.qpos[joint_inds[[10, 11]]]
    fl_tc = d.qpos[joint_inds[[4, 5]]]
    rr_tc = d.qpos[joint_inds[[7, 8]]]
    diag_diff = float(np.sum((fr_tc - rl_tc) ** 2) + np.sum((fl_tc - rr_tc) ** 2))

    trot_reward           = 3.0 * float(trot_active)
    even_gait_reward      = 1.0 * float(trot_active and n_contacts == 2)
    swing_momentum_reward = 0.15 * min(float(np.sum(np.abs(d.qvel[6:18]))), 20.0) * float(trot_active)
    diagonal_sync_pen     = 2.5 * diag_diff
    upright           = 0.3 * d.qpos[3]
    flatness_reward   = 0.5 * max(0.0, 1.0 - 20.0 * tilt)
    # Reward executing the nav forward command
    cmd_vel_reward    = 4.0 * min(max(0.0, body_fwd_vel), TARGET_SPEED) * max(0.0, fwd_cmd)
    # Reward executing the nav turn command — positive turn_cmd = turn right (positive yaw)
    turn_follow_reward = 3.0 * float(np.clip(actual_yaw_rate * turn_cmd, -2.0, 2.0))

    fwd_bias          = 1.5 * min(max(0.0, body_fwd_vel), TARGET_SPEED) * max(0.0, fwd_cmd)
    backward_pen      = 15.0 * max(0.0, -body_fwd_vel)
    lateral_scale     = max(0.7, 1.0 - 0.3 * abs(turn_cmd))
    lateral_pen       = (15.0 * abs(body_lat_vel) + 30.0 * body_lat_vel ** 2) * lateral_scale
    hop_pen           = 2.0 * float(hop_active)
    grounded_pen      = 2.0 * float(n_contacts == 4)
    height_pen        = 3.0 * max(0.0, 0.28 - d.xpos[1][2])
    tilt_pen          = 15.0 * tilt
    roll_rate_pen     = 3.0 * abs(d.qvel[3])
    torque_pen        = 0.001 * np.sum(action ** 2)
    smoothness        = 0.01  * np.sum((action - prev_action) ** 2)
    joint_vel_pen     = 0.001 * np.sum(d.qvel[6:18] ** 2)
    hip_pen           = 0.15  * np.sum(d.qpos[joint_inds[[0, 3, 6, 9]]] ** 2)
    stationary_pen    = 3.0 * float(body_speed < 0.05) * max(0.0, fwd_cmd)
    unwanted_turn_pen = 1.5 * actual_yaw_rate ** 2 * max(0.0, 1.0 - abs(turn_cmd))

    return (trot_reward + clearance_reward + even_gait_reward + swing_momentum_reward
            + upright + flatness_reward + cmd_vel_reward + turn_follow_reward + fwd_bias
            - backward_pen - lateral_pen - hop_pen - grounded_pen - height_pen
            - tilt_pen - roll_rate_pen
            - torque_pen - smoothness - joint_vel_pen - hip_pen - stationary_pen
            - unwanted_turn_pen - diagonal_sync_pen)

# --- Per-env state ---
nav_buffers    = [RolloutBuffer() for _ in range(N_ENVS)]
walk_buffers   = [RolloutBuffer() for _ in range(N_ENVS)]
ep_steps       = [0] * N_ENVS
prev_actions   = [np.zeros(12) for _ in range(N_ENVS)]
goal_positions = [random_goal() for _ in range(N_ENVS)]
stand_counts   = [0] * N_ENVS

for i in range(N_ENVS):
    set_goal(envs_d[i], goal_positions[i])

nav_ppo  = PPO(state_dim=NAV_STATE_DIM,  action_dim=2,  hidden_dim=128, critic_hidden_dim=256)
walk_ppo = PPO(state_dim=WALK_STATE_DIM, action_dim=12, hidden_dim=256, critic_hidden_dim=512)

if os.path.exists("nav_checkpoint.pt"):
    nav_ppo.load("nav_checkpoint.pt")
if os.path.exists("walk_checkpoint.pt"):
    walk_ppo.load("walk_checkpoint.pt")

nav_best_reward  = float("-inf")
walk_best_reward = float("-inf")
if os.path.exists("nav_checkpoint_best.pt"):
    saved = torch.load("nav_checkpoint_best.pt", weights_only=False)
    if "best_reward" in saved:
        nav_best_reward = saved["best_reward"]
        print(f"Loaded previous nav best reward: {nav_best_reward:.3f}")
if os.path.exists("walk_checkpoint_best.pt"):
    saved = torch.load("walk_checkpoint_best.pt", weights_only=False)
    if "best_reward" in saved:
        walk_best_reward = saved["best_reward"]
        print(f"Loaded previous walk best reward: {walk_best_reward:.3f}")

step_count        = 0
nav_reward_accum  = 0.0
walk_reward_accum = 0.0

while True:
    # Step 1: nav policy decides direction / speed command
    nav_states                        = np.array([get_nav_state(envs_d[i], goal_positions[i]) for i in range(N_ENVS)])
    nav_cmds, nav_log_probs, nav_vals = nav_ppo.get_actions_batch(nav_states)

    # Step 2: walk policy uses body state + nav command to produce joint actions
    walk_states                           = np.array([get_walk_state(envs_d[i], nav_cmds[i]) for i in range(N_ENVS)])
    walk_acts, walk_log_probs, walk_vals  = walk_ppo.get_actions_batch(walk_states)

    # Step 3: apply joint actions and step physics
    for i in range(N_ENVS):
        envs_d[i].ctrl[ctr_inds] = HOME_CTRL + walk_acts[i] * ACTION_SCALE
        mujoco.mj_step(envs_m[i], envs_d[i])

    # Step 4: compute rewards and store in separate buffers
    for i in range(N_ENVS):
        ep_steps[i] += 1

        robot_xy     = envs_d[i].xpos[1][:2]
        dist_to_goal = np.linalg.norm(goal_positions[i] - robot_xy)
        if dist_to_goal < GOAL_RADIUS:
            stand_counts[i] += 1
        elif stand_counts[i] > 0 and dist_to_goal < GOAL_RADIUS * 1.5:
            pass  # brief bounce — keep counting
        else:
            stand_counts[i] = 0
        at_goal = stand_counts[i] > 0

        nav_r  = compute_nav_reward(envs_d[i], goal_positions[i], at_goal)
        walk_r = compute_walk_reward(envs_d[i], walk_acts[i], prev_actions[i], at_goal, nav_cmds[i])

        prev_actions[i]    = walk_acts[i].copy()
        nav_reward_accum  += nav_r
        walk_reward_accum += walk_r

        stood      = stand_counts[i] >= STAND_STEPS
        terminated = _is_fallen(envs_d[i]) or stood or ep_steps[i] >= MAX_EP_LEN

        nav_buffers[i].add(nav_states[i],   nav_cmds[i],  nav_r,  nav_log_probs[i],  nav_vals[i],  terminated)
        walk_buffers[i].add(walk_states[i], walk_acts[i], walk_r, walk_log_probs[i], walk_vals[i], terminated)

        if terminated:
            mujoco.mj_resetDataKeyframe(envs_m[i], envs_d[i], 0)
            prev_actions[i]   = np.zeros(12)
            ep_steps[i]       = 0
            stand_counts[i]   = 0
            goal_positions[i] = random_goal()
            set_goal(envs_d[i], goal_positions[i])

    step_count += N_ENVS

    # Step 5: update both policies when buffers fill
    if len(nav_buffers[0].rewards) >= STEPS_PER_UPDATE:
        next_nav_states  = np.array([get_nav_state(envs_d[i], goal_positions[i]) for i in range(N_ENVS)])
        next_nav_cmds, _, _ = nav_ppo.get_actions_batch(next_nav_states)
        next_walk_states = np.array([get_walk_state(envs_d[i], next_nav_cmds[i]) for i in range(N_ENVS)])

        next_nav_vals  = nav_ppo.critic(
            torch.FloatTensor(next_nav_states)
        ).squeeze(-1).detach().numpy()
        next_walk_vals = walk_ppo.critic(
            torch.FloatTensor(next_walk_states)
        ).squeeze(-1).detach().numpy()

        nav_ppo.update_multi(nav_buffers, next_nav_vals)
        walk_ppo.update_multi(walk_buffers, next_walk_vals)

        nav_ppo.save("nav_checkpoint.pt")
        walk_ppo.save("walk_checkpoint.pt")

        nav_avg  = nav_reward_accum  / (STEPS_PER_UPDATE * N_ENVS)
        walk_avg = walk_reward_accum / (STEPS_PER_UPDATE * N_ENVS)
        nav_reward_accum  = 0.0
        walk_reward_accum = 0.0

        nav_tag  = ""
        walk_tag = ""
        if nav_avg > nav_best_reward:
            nav_best_reward = nav_avg
            nav_ppo.save("nav_checkpoint_best.pt", best_reward=nav_avg)
            nav_tag = " NEW BEST"
        if walk_avg > walk_best_reward:
            walk_best_reward = walk_avg
            walk_ppo.save("walk_checkpoint_best.pt", best_reward=walk_avg)
            walk_tag = " NEW BEST"

        print(f"Step {step_count} | Nav: {nav_avg:.3f}{nav_tag} | Walk: {walk_avg:.3f}{walk_tag}")
