from stable_baselines3 import DQN
from agent import Agent

class DQNAgent(Agent):
    def __init__(self, num_experiments=1):
        super().__init__('DQN', num_experiments)

    def setup_model(self, env):
        self.model = DQN(
            policy="MlpPolicy",
            env=env,
            learning_rate=0.001,
            learning_starts=0,
            train_freq=1,
            target_update_interval=500,
            exploration_initial_eps=0.05,
            exploration_final_eps=0.01,
            verbose=1,
        )
        
dqn_agent = DQNAgent()
dqn_agent.run_experiments()



# import torch
# import torch.nn as nn
# import torch.optim as optim
# import torch.nn.functional as F

# # Just a look on how can DQN be defined we're going to change a lot about this also 
# # to match with the algorithm
# class DQN(nn.Module):

#     def __init__(self, n_observations, n_actions):
#         super(DQN, self).__init__()
#         self.layer1 = nn.Linear(n_observations, 128)
#         self.layer2 = nn.Linear(128, 128)
#         self.layer3 = nn.Linear(128, n_actions)
    
#     def forward(self, x):
#         x = F.relu(self.layer1(x))
#         x = F.relu(self.layer2(x))
#         return self.layer3(x)