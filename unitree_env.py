import gymnasium as gym
from gymnasium import spaces
import numpy as np
import mujoco
from mj_utils import * # Make sure this is in the same directory

class UnitreeStandEnv(gym.Env):
    def __init__(self):
        super().__init__()
        
        # Load MuJoCo model
        self.model = mujoco.MjModel.from_xml_path("./unitree_a1/scene.xml")
        self.data = mujoco.MjData(self.model)
        self.model.opt.timestep = 0.002
        
        # Define indices
        self.JOINT_NAMES = (
            "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
            "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
            "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
            "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
        )
        self.ACTUATOR_NAMES = (
            "FR_hip", "FR_thigh", "FR_calf",
            "FL_hip", "FL_thigh", "FL_calf",
            "RR_hip", "RR_thigh", "RR_calf",
            "RL_hip", "RL_thigh", "RL_calf",
        )
        
        self.joint_inds = get_qpos_indices(self.model, self.JOINT_NAMES)
        self.ctrl_inds = get_ctrl_indices(self.model, self.ACTUATOR_NAMES)
        self.qvel_inds = get_qvel_indices(self.model, self.JOINT_NAMES)
        
        # Base standing posture
        self.nominal_posture = np.array([0.0, 0.8, -1.5] * 4)
        
        # Action space: 12 continuous values bounded between -1 and 1
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(12,), dtype=np.float32)
        
        # Observation space: 12 joint angles + 12 joint vels + 3 body pos + 4 body quat = 31
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(31,), dtype=np.float32)
        
        self.step_count = 0
        self.max_steps = 1000 # 1000 steps * 0.02s (frameskip) = 20 seconds per episode

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        
        # Add slight random noise to the starting position to ensure robust learning
        noise = np.random.uniform(-0.1, 0.1, size=len(self.joint_inds))
        self.data.qpos[self.joint_inds] = self.nominal_posture + noise
        
        mujoco.mj_forward(self.model, self.data)
        self.step_count = 0
        
        return self._get_obs(), {}

    def step(self, action):
        self.step_count += 1
        
        # Scale action to +/- 0.5 radians and add to nominal posture
        target_angles = self.nominal_posture + (action * 0.5)
        set_ctrl_values(self.data, self.ctrl_inds, target_angles)
        
        # Control rate is usually lower than physics rate. 
        # Step physics 10 times per RL action (0.002 * 10 = 0.02s per RL step)
        for _ in range(10): 
            mujoco.mj_step(self.model, self.data)
            
        obs = self._get_obs()
        reward = self._compute_reward()
        
        # Terminate if the robot falls over
        trunk_z = self.data.qpos[2]
        terminated = bool(trunk_z < 0.2) # Fell over
        
        # Truncate if we reach the maximum time limit
        truncated = bool(self.step_count >= self.max_steps)
        
        return obs, reward, terminated, truncated, {}

    def _get_obs(self):
        # Gather sensor feedback
        qpos = self.data.qpos[self.joint_inds]
        qvel = self.data.qvel[self.qvel_inds]
        trunk_pos = self.data.qpos[0:3]   # x, y, z
        trunk_quat = self.data.qpos[3:7]  # orientation quaternion
        
        return np.concatenate([qpos, qvel, trunk_pos, trunk_quat]).astype(np.float32)

    def _compute_reward(self):
        # 1. Height Reward: Keep trunk near 0.3m
        trunk_z = self.data.qpos[2]
        target_height = 0.3
        height_reward = 1.0 - abs(trunk_z - target_height) 
        
        # 2. Velocity Penalty: INCREASED to stop the terrified jittering
        joint_vels = self.data.qvel[self.qvel_inds]
        vel_penalty = np.sum(np.square(joint_vels)) * 0.005 # Was 0.001
        
        # 3. Posture Penalty: INCREASED to stop the wide-legged splits
        qpos = self.data.qpos[self.joint_inds]
        posture_penalty = np.sum(np.square(qpos - self.nominal_posture)) * 0.15 # Was 0.05
        
        # 4. Drift Penalty: NEW! Penalize the trunk for sliding horizontally
        trunk_xy_vel = self.data.qvel[0:2] # x and y linear velocity of the trunk
        drift_penalty = np.sum(np.square(trunk_xy_vel)) * 0.05
        
        # 5. Alive Bonus
        alive_bonus = 2.0
        
        # Total Reward
        return float(height_reward + alive_bonus - vel_penalty - posture_penalty - drift_penalty)