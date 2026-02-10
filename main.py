import os
import time
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env

from robot_env import RobotEnv

# --- Setup ---
# Create directories for logs and models
log_dir = "logs"
model_dir = "models"
os.makedirs(log_dir, exist_ok=True)
os.makedirs(model_dir, exist_ok=True)

model_path = os.path.join(model_dir, "ppo_robot_continuous_model.zip")

# --- 1. Environment Creation ---
print("Creating robot environment...")
env = RobotEnv(world_size=10.0)
print("Environment created.")

# --- 2. PPO Agent Creation ---
print("Creating PPO agent (with MlpPolicy)...")
# MlpPolicy is used for observation vectors (not images)
model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=log_dir)
print("Agent created.")

# --- 3. Training ---
# Training in continuous space is harder, increasing steps.
TIMESTEPS = 200000 
print(f"Starting training for {TIMESTEPS} steps...")
model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name="PPO_Robot_Continuous")
print("Training completed.")

# --- 4. Saving the Model ---
print(f"Saving model to: {model_path}")
model.save(model_path)
print("Model saved.")

del model  # Deleting model to demonstrate loading

# --- 5. Evaluating the Trained Model ---
print("\n--- Evaluating trained model ---")

# Create new environment in 'human' mode for evaluation
eval_env = RobotEnv(world_size=10.0, render_mode="human")
# Load the trained model into this environment
model = PPO.load(model_path, env=eval_env)


episodes = 5
for ep in range(episodes):
    obs, _ = eval_env.reset()
    done = False
    total_reward = 0
    while not done:
        # deterministic=True makes the agent select the most likely action
        action, _states = model.predict(obs, deterministic=True)
        # Sürekli aksiyon bir vektör olduğu için .item() KULLANILMAZ
        obs, reward, terminated, truncated, info = eval_env.step(action)
        total_reward += reward
        done = terminated or truncated
    print(f"Episode {ep + 1}: Total Reward: {total_reward:.2f}")
    time.sleep(2)  # Wait 2 seconds between episodes

eval_env.close()

print("\n--- Evaluation Completed ---")
print("\nTo see training logs, run the following command in terminal:")
print(f"tensorboard --logdir={log_dir}")