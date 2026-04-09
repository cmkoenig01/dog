import time
import mujoco
import mujoco.viewer
from stable_baselines3 import PPO
from unitree_env import UnitreeStandEnv

def main():
    print("Loading the best trained model...")
    
    # Load the highest-scoring neural network weights
    # (If you didn't use the callback version, change this path to "models/PPO/stand_policy")
    model = PPO.load("models/PPO/stand_policy")

    # Initialize our custom environment
    env = UnitreeStandEnv()
    obs, _ = env.reset()

    print("Launching viewer. Press ESC to quit.")
    
    # Open the interactive MuJoCo viewer
    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        while viewer.is_running():
            step_start = time.time()

            # 1. The AI looks at the sensors and decides how to move
            action, _states = model.predict(obs, deterministic=True)
            
            # 2. Apply the movement to the robot
            obs, reward, terminated, truncated, info = env.step(action)
            
            # 3. Update the graphics
            viewer.sync()
            
            # 4. If the dog falls over, reset it to try again
            if terminated or truncated:
                obs, _ = env.reset()

            # Throttle the loop so it plays at normal speed (not hyper-speed)
            time_until_next_step = 0.02 - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

if __name__ == "__main__":
    main()