import os 
import sys
import json

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
        net_file = "src/Intersection/2way-single-intersection/single-intersection.net.xml",
        route_file = "src/Intersection/2way-single-intersection/single-intersection-vhvh.rou.xml",
        out_csv_name = f"data/DQN_2way",
        use_gui = False,
        num_seconds = 6000,
    )

    dqn = DQN(env = env)
    dqn.train(num_episodes = 5)

    # A2C
    env = SumoEnvironment(
        net_file = "src/Intersection/2way-single-intersection/single-intersection.net.xml",
        route_file = "src/Intersection/2way-single-intersection/single-intersection-vhvh.rou.xml",
        out_csv_name = f"data/A2C_2way",
        use_gui = False,
        num_seconds = 6000,
    )

    a2c = A2C(env = env)
    a2c.train(num_episodes = 6)

    def average_csv(files):
        """Averages multiple CSV files into a single DataFrame."""
        data_frames = [pd.read_csv(f) for f in files]
        df_concatenated = pd.concat(data_frames)
        df_mean = df_concatenated.groupby(df_concatenated.index).mean()
        return df_mean

    # Paths to your CSV files
    dqn_files = ['data/DQN_run1.csv', 'data/DQN_run2.csv', 'data/DQN_run3.csv', 'data/DQN_run4.csv', 'data/DQN_run5.csv']
    a2c_files = ['data/A2C_run1.csv', 'data/A2C_run2.csv', 'data/A2C_run3.csv', 'data/A2C_run4.csv', 'data/A2C_run5.csv']
    random_files = ['data/Stochastic_run1.csv', 'data/Stochastic_run2.csv', 'data/Stochastic_run3.csv', 'data/Stochastic_run4.csv', 'data/Stochastic_run5.csv']

    # Average the results
    dqn_avg = average_csv(dqn_files)
    a2c_avg = average_csv(a2c_files)
    random_avg = average_csv(random_files)

    # Optionally, save the averaged results to new CSV files
    dqn_avg.to_csv('dqn_average.csv')
    a2c_avg.to_csv('a2c_average.csv')
    random_avg.to_csv('random_average.csv')
    # # Plot the results
    # plotWaitingTime(a2c_file = 'data/A2C_2way_ep2.csv', 
    #                 dqn_file = 'data/DQN_2way_ep2.csv', 
    #                 random_file = 'data/RANDOM_2way_ep1.csv')





