import os
import sys
import json
from stable_baselines3 import A2C, PPO, DQN  # Import all the algorithms you intend to use
from stable_baselines3.a2c import MlpPolicy

class BaseAgent:
    def __init__(self, agent_type, num_experiments=50):
        self.config = self.load_configuration()
        self.setup_sumo_environment(self.config)
        self.agent_type = agent_type
        self.num_experiments = num_experiments

    @staticmethod
    def load_configuration():
        with open('config.json', 'r') as config_file:
            return json.load(config_file)

    @staticmethod
    def setup_sumo_environment(config):
        if "SUMO_HOME" not in os.environ:
            sys.exit("Please declare the environment variable 'SUMO_HOME'")
        tools = os.path.join(os.environ["SUMO_HOME"], "tools")
        sys.path.append(tools)
        sys.path.append(config["project_base_path"])

    def run_experiments(self):
        from src.Environment.env import SumoEnvironment

        for i in range(self.num_experiments):
            env = SumoEnvironment(
                net_file="src/Intersection/2way-single-intersection/single-intersection.net.xml",
                route_file="src/Intersection/2way-single-intersection/single-intersection-vhvh.rou.xml",
                out_csv_name=f"data/{self.agent_type}_2way_test_csv_run{i}",
                use_gui=True,
                num_seconds=6000,
            )

            self.setup_model(env)
            self.model.learn(total_timesteps=6000)
            

    def setup_model(self, env):
        raise NotImplementedError("This method should be implemented by subclasses")
