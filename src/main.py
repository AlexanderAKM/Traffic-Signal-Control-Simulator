import os 
import sys
import json

# Load configuration from config_mine.json
with open('config_mine.json', 'r') as config_file:
    config = json.load(config_file)

# Set SUMO tools path from config file
if config["sumo_tools_path"]:
    tools = config["sumo_tools_path"]
    sys.path.append(tools)
else:
    if "SUMO_HOME" in os.environ:
        tools = os.path.join(os.environ["SUMO_HOME"], "tools")
        sys.path.append(tools)
    else:
        sys.exit("Please declare the environment variable 'SUMO_HOME' or set 'sumo_tools_path' in the config file.")

import traci

# Use project base path from config file
if config["project_base_path"]:
    sys.path.append(config["project_base_path"])
else:
    sys.exit("Please set 'project_base_path' in the config file.")

from src.environment.env import SumoEnvironment
from src.agents.dqn import DQN
from src.agents.a2c import A2C
from src.agents.stochastic import StochasticModel
from src.plotting.plot import plotWaitingTime
import numpy as np
import pandas as pd

# Run the experiments
if __name__ == '__main__':

    # Random agent
    stochastic_agent = StochasticModel()
    stochastic_agent.train(num_episodes = 5)
    
    # DQN
    env = SumoEnvironment(
        net_file = os.path.join(config["project_base_path"], "src/Intersection/2way-single-intersection/single-intersection.net.xml"),
        route_file = os.path.join(config["project_base_path"], "src/Intersection/2way-single-intersection/single-intersection-vhvh.rou.xml"),
        out_csv_name = os.path.join(config["project_base_path"], "data/DQN_2way"),
        use_gui = True,
        num_seconds = 10000,
    )

    dqn = DQN(env = env)
    dqn.train(num_episodes = 5)

    # A2C
    env = SumoEnvironment(
        net_file = os.path.join(config["project_base_path"], "src/Intersection/2way-single-intersection/single-intersection.net.xml"),
        route_file = os.path.join(config["project_base_path"], "src/Intersection/2way-single-intersection/single-intersection-vhvh.rou.xml"),
        out_csv_name = os.path.join(config["project_base_path"], "data/A2C_2way"),
        use_gui = True,
        num_seconds = 10000,
    )

    a2c = A2C(env = env)
    a2c.train(num_episodes = 6)

    # Plot the results
    plotWaitingTime()