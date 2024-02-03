import os
import sys
import json 

class Agent:
    
    def __init__(self, agent_type):
        self.config = self.load_configuration()
        self.setup_sumo_environment(self.config)
        self.agent_type = agent_type

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

    def train(self, num_episodes):
        from src.environment.env import SumoEnvironment
        for i in range(num_episodes):
            env = SumoEnvironment(
                net_file="src/Intersection/2way-single-intersection/single-intersection.net.xml",
                route_file="src/Intersection/2way-single-intersection/single-intersection-vhvh.rou.xml",
                out_csv_name=f"data/{self.agent_type}_2way_test_csv_ep{i}",
                use_gui = False,
                num_seconds = 6000,
            )

            self.setup_model(env)
            self.model.learn(total_timesteps=6000)
            
    def setup_model(self):
        raise NotImplementedError("This method should be implemented by subclasses")
