import os
import sys

import gymnasium as gym
from stable_baselines3.dqn.dqn import DQN


if "SUMO_HOME" in os.environ:
    tools = os.path.join(os.environ["SUMO_HOME"], "tools")
    sys.path.append(tools)
else:
    sys.exit("Please declare the environment variable 'SUMO_HOME'")
import traci

sys.path.append("/UNI_STUFF/SECOND_YEAR/Block_1B/Reinforcement Learning/Assignments/final-40")
from src.Environment.env import SumoEnvironment


if __name__ == "__main__":
    env = SumoEnvironment(
        net_file="src/Intersection/Configuration/cross.net.xml",
        route_file="src/Intersection/Configuration/cross.rou.xml",
        out_csv_name="data/test_csv",
        use_gui=True,
        num_seconds=100000,
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
    model.learn(total_timesteps=100000)