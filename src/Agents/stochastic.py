import os
import sys
import json
import gymnasium as gym

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

class StochasticModel:
    def __init__(self, env):
        self.env = env
        self.action_space = env.action_space

    def learn(self, total_timesteps: int):
        obs = self.env.reset()
        for step in range(total_timesteps):
            action = self.action_space.sample()  # Random action
            obs, reward, terminated, truncated, info = self.env.step(action)
            if truncated:  # Check if the episode should end
                self.env.save_csv(self.env.out_csv_name, self.env.episode)  # Call the save_csv method
                break
        self.env.close()

    def predict(self, observation, state=None, mask=None, deterministic=False):
        return self.action_space.sample(), None

# Number of experiments to run
num_experiments = 50

for i in range(num_experiments):
    env = SumoEnvironment(
        net_file="src/Intersection/2way-single-intersection/single-intersection.net.xml",
        route_file="src/Intersection/2way-single-intersection/single-intersection-vhvh.rou.xml",
        out_csv_name=f"data/2way_stochastic_csv_run{i}",  # Unique file name for each run
        use_gui=True,
        num_seconds=6000,
    )
    
    model = StochasticModel(env)
    model.learn(total_timesteps=6000)
    
    # After running the model, you may want to save the results
    # Save results code here (if needed)
