import os
import sys
import json
import gymnasium as gym
from stable_baselines3.dqn.dqn import DQN

# Read configuration (see config.json)
with open('config.json', 'r') as config_file:
    config = json.load(config_file)

if "SUMO_HOME" in os.environ:
    tools = os.path.join(os.environ["SUMO_HOME"], "tools")
    sys.path.append(tools)
else:
    sys.exit("Please declare the environment variable 'SUMO_HOME'")
import traci

# Use paths from config file
sys.path.append(config["project_base_path"])
from src.Environment.env import SumoEnvironment

# Number of experiments to run
num_experiments = 50

for i in range(num_experiments):
    env = SumoEnvironment(
        net_file="src/Intersection/Configuration/cross.net.xml",
        route_file="src/Intersection/Configuration/cross.rou.xml",
        out_csv_name=f"data/test_csv_run{i}",  # Unique file name for each run
        use_gui=True,
        num_seconds=6000,
    )

    model = DQN(
        env=env,
        policy="MlpPolicy",
        learning_rate=0.001,
        learning_starts=0,
        train_freq=1,
        target_update_interval=500,
        exploration_initial_eps=0.05,
        exploration_final_eps=0.01,
        verbose=1,
    )

    model.learn(total_timesteps=6000)

    # After learning, you may want to save the model
    # model.save(f"data/dqn_model_run{i}")
