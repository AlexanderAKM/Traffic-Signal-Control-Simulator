from stable_baselines3 import A2C
from stable_baselines3.a2c import MlpPolicy
from agent import Agent

class A2CAgent(Agent):
    def __init__(self, num_experiments=1):
        super().__init__('A2C', num_experiments)

    def setup_model(self, env):
        self.model = A2C(
            policy=MlpPolicy,
            env=env,
            learning_rate=0.001,
            n_steps=5,
            gamma=0.99,
            gae_lambda=1.0,
            ent_coef=0.01,
            vf_coef=0.5,
            max_grad_norm=0.5,
            verbose=1
        )
        
a2c_agent = A2CAgent()
a2c_agent.run_experiments()



'''
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
        net_file="src/Intersection/2way-single-intersection/single-intersection.net.xml",
        route_file="src/Intersection/2way-single-intersection/single-intersection-vhvh.rou.xml",
        out_csv_name=f"data/A2C_2way_test_csv_run{i}",  # Unique file name for each run
        use_gui=True,
        num_seconds=6000,
    )

    model = A2C(
        policy=MlpPolicy,
        env=env,
        learning_rate=0.001,
        n_steps=5,
        gamma=0.99,
        gae_lambda=1.0,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        #tensorboard_log="./a2c_tensorboard/",  # Optional: For tensorboard logging
        verbose=1
    )


    model.learn(total_timesteps=6000)

    # After learning, you may want to save the model
    # model.save(f"data/a2c_model_run{i}")

# Additional code for saving results or handling outputs
'''