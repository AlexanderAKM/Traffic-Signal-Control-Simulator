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

# Run the experiments
if __name__ == '__main__':
    
    # DQN
    for i in range(1):
        env = SumoEnvironment(
            net_file="src/Intersection/2way-single-intersection/single-intersection.net.xml",
            route_file="src/Intersection/2way-single-intersection/single-intersection-vhvh.rou.xml",
            out_csv_name=f"data/dqn_2way_test_csv_run{i}",
            use_gui=True,
            num_seconds=6000,
        )

        dqn = DQN(env = env)
        dqn.train(num_episodes=1)

    # A2C
    for i in range(1):
        env = SumoEnvironment(
            net_file="src/Intersection/2way-single-intersection/single-intersection.net.xml",
            route_file="src/Intersection/2way-single-intersection/single-intersection-vhvh.rou.xml",
            out_csv_name=f"data/A2C_2way_test_csv_run{i}",
            use_gui=True,
            num_seconds=6000,
        )

        a2c = A2C(env = env)
        a2c.train(num_episodes = 3)
