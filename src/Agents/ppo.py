from stable_baselines3 import PPO
from agent import Agent

class PPOAgent(Agent):
    def __init__(self, num_experiments=1):
        super().__init__('PPO', num_experiments)

    def setup_model(self, env):
        self.model = PPO(
            policy="MlpPolicy",
            env=env,
            learning_rate=0.001,
            n_steps=512,  # lower n_steps seems to be better (512 at least way better than 2048)
            batch_size=64,
            n_epochs=10,  # Number of epochs to update the model
            gamma=0.99,
            gae_lambda=0.95,  # Adjusted for PPO
            clip_range=0.2,  # Clipping parameter
            ent_coef=0.0,
            vf_coef=0.5,
            max_grad_norm=0.5,
            verbose=1,
        )
        
ppo_agent = PPOAgent()
ppo_agent.run_experiments()




'''
import os
import sys
import json
import gymnasium as gym
from stable_baselines3 import PPO  # Importing PPO
from stable_baselines3.ppo import MlpPolicy  # Importing PPO's policy

# Read configuration (see config.json)
with open('config.json', 'r') as config_file:
    config = json.load(config_file)

# Ensure SUMO_HOME environment variable is set
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
    # Initialize the SumoEnvironment
    env = SumoEnvironment(
        net_file="src/Intersection/2way-single-intersection/single-intersection.net.xml",
        route_file="src/Intersection/2way-single-intersection/single-intersection-vhvh.rou.xml",
        out_csv_name=f"data/PPO_2way_test_csv_run{i}",  # Update file name for PPO runs
        use_gui=True,
        num_seconds=6000,
    )

    # Initialize the PPO model
    model = PPO(
        policy=MlpPolicy,
        env=env,
        learning_rate=0.001,
        n_steps=512,  # lower n_steps seems to be better (512 at least way better than 2048)
        batch_size=64,
        n_epochs=10,  # Number of epochs to update the model
        gamma=0.99,
        gae_lambda=0.95,  # Adjusted for PPO
        clip_range=0.2,  # Clipping parameter
        ent_coef=0.0,
        vf_coef=0.5,
        max_grad_norm=0.5,
        #tensorboard_log="./ppo_tensorboard/",  # Tensorboard logging directory for PPO
        verbose=1,
    )

    # Train the model
    model.learn(total_timesteps=6000)

    # Optional: Save the model after training
    # model.save(f"data/ppo_model_run{i}")

# Additional code for saving results or handling outputs

'''