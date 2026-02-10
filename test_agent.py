import os
import time
from stable_baselines3 import PPO
from robot_env import RobotEnv

# --- Model and Environment Setup ---
MODEL_NAME = "ppo_robot_continuous_model.zip"
model_path = os.path.join("models", MODEL_NAME)

print("--- Starting Test ---")

# 1. Check if model exists
if not os.path.exists(model_path):
    print(f"Error: Model file not found: {model_path}")
    print("Please run main.py first to train the model.")
    exit()

# 2. Create environment in 'human' mode
print("Creating environment in 'human' mode... (A window will open shortly)")
try:
    # Start the environment. If a PNG file exists, it uses it, otherwise it draws a circle.
    eval_env = RobotEnv(world_size=10.0, render_mode="human")
except Exception as e:
    print(f"Error creating environment: {e}")
    exit()


# 3. Load trained model
print(f"Loading trained model: {MODEL_NAME}")
try:
    model = PPO.load(model_path, env=eval_env)
except Exception as e:
    print(f"Error loading model: {e}")
    exit()


# 4. Run and visualize multiple episodes
print("\n--- Running Test Episodes ---")
print("Look at the opened window to watch the robot's movement.")
print("(Note: It might take a few seconds for the window to appear)")

num_episodes_to_test = 5 # Number of scenarios to test
time.sleep(2) # Preparation time for user

try:
    for episode_num in range(num_episodes_to_test):
        print(f"\n--- Episode {episode_num + 1} Starting ---")
        obs, _ = eval_env.reset() # Reset environment for each episode
        done = False
        total_reward = 0
        step_count = 0

        while not done:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = eval_env.step(action)
            
            total_reward += reward
            step_count += 1
            done = terminated or truncated
            
            # Slow down simulation for better visibility
            time.sleep(0.05) 

        print(f"Episode {episode_num + 1}: Completed in {step_count} steps. Total Reward: {total_reward:.2f}")
        time.sleep(2) # Longer wait between episodes

except Exception as e:
    if "Tcl_AsyncDelete: async handler deleted by the wrong thread" not in str(e):
         print(f"\nError during testing: {e}")
   
finally:
    print("\nWindow will close in 5 seconds...")
    time.sleep(5)
    eval_env.close()
    print("\n--- Test Completed ---")
