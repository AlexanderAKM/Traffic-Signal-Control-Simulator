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
#from src.agents.ppo import PPOAgent
#from src.agents.stochastic import StochasticModel

# Run the experiments
if __name__ == '__main__':
    
    # # DQN
    # env = SumoEnvironment(
    #     net_file = "src/Intersection/2way-single-intersection/single-intersection.net.xml",
    #     route_file = "src/Intersection/2way-single-intersection/single-intersection-vhvh.rou.xml",
    #     out_csv_name = f"data/DQN_2way_",
    #     use_gui = False,
    #     num_seconds = 6000,
    # )

    # dqn = DQN(env = env)
    # dqn.train(num_episodes = 3)

    # A2C
    env = SumoEnvironment(
        net_file = "src/Intersection/2way-single-intersection/single-intersection.net.xml",
        route_file = "src/Intersection/2way-single-intersection/single-intersection-vhvh.rou.xml",
        out_csv_name = f"data/A2C_2way_",
        use_gui = False,
        num_seconds = 6000,
    )

    a2c = A2C(env = env)
    a2c.train(num_episodes = 3)

    # # PPO
    # ppo_agent = PPOAgent()
    # ppo_agent.train(num_episodes = 3)

    # # Random agent
    # stochastic_agent = StochasticModel()
    # stochastic_agent.train(num_episodes = 3)