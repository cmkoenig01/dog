import os
from stable_baselines3 import PPO
from unitree_env import UnitreeStandEnv

def main():
    # 1. Create the environment
    env = UnitreeStandEnv()
    
    # 2. Define directory to save the trained model
    models_dir = "models/PPO"
    os.makedirs(models_dir, exist_ok=True)
    
    # 3. Initialize the PPO algorithm (The RL Controller)
    # MlpPolicy creates a standard feedforward neural network
    model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="./ppo_unitree_tensorboard/")
    
    print("Starting training...")
    # 4. Train for 500,000 steps. 
    # (You may need to increase this to 1M-2M for perfect behavior)
    TIMESTEPS = 500000 
    model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False)
    
    # 5. Save the trained weights
    model.save(f"{models_dir}/stand_policy")
    print("Training complete and model saved!")
    
    env.close()

if __name__ == "__main__":
    main()