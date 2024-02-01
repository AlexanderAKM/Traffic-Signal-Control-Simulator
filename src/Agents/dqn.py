# # import torch
# # import torch.nn as nn
# # import torch.nn.functional as F
# # import torch.optim as optim
# # import numpy as np
# # from collections import deque, namedtuple
# # import random
# # from itertools import count
# # from agent import Agent

# # Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state'))

# # class DQNNetwork(nn.Module):

# #     def __init__(self, state_dim, action_dim):
# #         super(DQNNetwork, self).__init__()
# #         # Define the network layers
# #         # Example architecture (modify as needed)
# #         self.fc1 = nn.Linear(state_dim, 64)
# #         self.fc2 = nn.Linear(64, 64)
# #         self.fc3 = nn.Linear(64, action_dim)

# #     def forward(self, x):
# #         x = torch.relu(self.fc1(x))
# #         x = torch.relu(self.fc2(x))
# #         return self.fc3(x)

# # class ReplayBuffer:
# #     def __init__(self, capacity):
# #         self.buffer = deque(maxlen=capacity)

# #     def push(self, state, action, reward, next_state, done):
# #         self.buffer.append(Transition(state, action, reward, next_state))

# #     def sample(self, batch_size):
# #         return random.sample(self.buffer, batch_size)

# #     def __len__(self):
# #         return len(self.buffer)
        
# # class DQNAgent(Agent):

# #     def __init__(self, num_experiments=1):
# #         super().__init__('DQN', num_experiments)
# #         self.policy_net = None
# #         self.target_net = None
# #         self.memory = ReplayBuffer(10000)
# #         self.optimizer = None
# #         self.model = None

# #         # Hyperparameters
# #         self.batch_size = 64
# #         self.gamma = 0.99
# #         self.epsilon_start = 1.0
# #         self.epsilon_end = 0.01
# #         self.epsilon_decay = 500
# #         self.step_count = 0

# #     def setup_model(self, env):
# #         self.env = env
# #         state_dim = env.observation_space.shape[0]
# #         action_dim = env.action_space.n

# #         self.policy_net = DQNNetwork(state_dim, action_dim)
# #         self.target_net = DQNNetwork(state_dim, action_dim)
# #         self.target_net.load_state_dict(self.policy_net.state_dict())
# #         self.target_net.eval()

# #         self.optimizer = optim.Adam(self.policy_net.parameters(), lr=0.001)
# #         self.model = self

# #     def select_action(self, state, epsilon):
# #         sample = random.random()
# #         eps_threshold = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * \
# #                         np.exp(-1. * self.step_count / self.epsilon_decay)
# #         self.step_count += 1

# #         if sample > eps_threshold:
# #             with torch.no_grad():
# #                 return self.policy_net(state).max(1)[1].view(1, 1)
# #         else:
# #             return torch.tensor([[random.randrange(self.action_dim)]], dtype=torch.long)

# #     def learn(self, total_timesteps):
# #         if len(self.memory) < self.batch_size:
# #             return

# #         transitions = self.memory.sample(self.batch_size)
# #         batch = Transition(*zip(*transitions))

# #         # Compute a mask of non-final states and concatenate the batch elements
# #         non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
# #                                                 batch.next_state)), dtype=torch.bool)
# #         non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
# #         state_batch = torch.cat(batch.state)
# #         action_batch = torch.cat(batch.action)
# #         reward_batch = torch.cat(batch.reward)

# #         # Compute Q(s_t, a) - the model computes Q(s_t), then we select the columns of actions taken
# #         state_action_values = self.policy_net(state_batch).gather(1, action_batch)

# #         # Compute V(s_{t+1}) for all next states.
# #         next_state_values = torch.zeros(self.batch_size)
# #         next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0].detach()

# #         # Compute the expected Q values
# #         expected_state_action_values = (next_state_values * self.gamma) + reward_batch

# #         # Compute Huber loss
# #         loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

# #         # Optimize the model
# #         self.optimizer.zero_grad()
# #         loss.backward()
# #         for param in self.policy_net.parameters():
# #             param.grad.data.clamp_(-1, 1)
# #         self.optimizer.step()

# #     def run_experiments(self):
# #         super().run_experiments()
# #         TARGET_UPDATE = 10

# #         for i in range(self.num_experiments):
# #             # Reset environment and state
# #             self.env.reset()
# #             state = torch.tensor([self.env.observation_space()], dtype=torch.float)
# #             for t in count():
# #                 # Select and perform an action
# #                 action = self.select_action(state)
# #                 next_state, reward, done, _ = self.env.step(action.item())
# #                 reward = torch.tensor([reward], dtype=torch.float)

# #                 # Observe new state
# #                 if not done:
# #                     next_state = torch.tensor([next_state], dtype=torch.float)
# #                 else:
# #                     next_state = None

# #                 # Store the transition in memory
# #                 self.memory.push(state, action, reward, next_state)

# #                 # Move to the next state
# #                 state = next_state

# #                 # Perform one step of the optimization
# #                 self.optimize_model()
# #                 if done:
# #                     break

# #             # Update the target network
# #             if i % TARGET_UPDATE == 0:
# #                 self.target_net.load_state_dict(self.policy_net.state_dict())

# # dqn_agent = DQNAgent(num_experiments=1)
# # dqn_agent.run_experiments()

# from stable_baselines3 import DQN
# from agent import Agent

# class DQNAgent(Agent):
#     def __init__(self, num_experiments=1):
#         super().__init__('DQN', num_experiments)

#     def setup_model(self, env):
#         self.model = DQN(
#             policy="MlpPolicy",
#             env=env,
#             learning_rate=0.001,
#             learning_starts=0,
#             train_freq=1,
#             target_update_interval=500,
#             exploration_initial_eps=0.05,
#             exploration_final_eps=0.01,
#             verbose=1,
#         )
        
# dqn_agent = DQNAgent()
# dqn_agent.run_experiments()


# '''
# # import torch
# # import torch.nn as nn
# # import torch.optim as optim
# # import torch.nn.functional as F

# # # Just a look on how can DQN be defined we're going to change a lot about this also 
# # # to match with the algorithm
# # class DQN(nn.Module):

# #     def __init__(self, n_observations, n_actions):
# #         super(DQN, self).__init__()
# #         self.layer1 = nn.Linear(n_observations, 128)
# #         self.layer2 = nn.Linear(128, 128)
# #         self.layer3 = nn.Linear(128, n_actions)
    
# #     def forward(self, x):
# #         x = F.relu(self.layer1(x))
# #         x = F.relu(self.layer2(x))
# #         return self.layer3(x)'''

import torch
import torch.nn as nn
import torch.nn.functional as F 
from collections import deque, namedtuple
import random
import torch.optim as optim
import math
from itertools import count

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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen = capacity)
    
    def push(self, *args):
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
    
    def __len__(self):
        return len(self.memory)
    

class DQN(nn.Module):

    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()

        self.layer1 = nn.Linear(n_observations, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, n_actions)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))

        return self.layer3(x)

class DQNTrain():

    def __init__(self,
                 env = None,
                 batch_size = 128,
                 gamma = 0.99, 
                 eps_start = 0.9, 
                 eps_end = 0.05,
                 eps_decay = 1000, 
                 tau = 0.005,
                 lr = 1e-4):
        
        self.env = env
        self.batch_size = batch_size
        self.gamma = gamma
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay
        self.tau = tau
        self.lr = lr

        self.n_actions = 4
        self.state, self.info = self.env.reset()
        self.n_observations = len(self.state)

        self.policy_net = DQN(self.n_observations, self.n_actions).to(device)
        self.target_net = DQN(self.n_observations, self.n_actions).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())

        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr = self.lr, amsgrad = True)
        self.memory = ReplayMemory(1000)

        self.steps_done = 0

    def select_action(self, state):
        sample = random.random()
        eps_threshold = self.eps_end + (self.eps_start - self.eps_end) * \
            math.exp(-1. * self.steps_done / self.eps_decay)
        self.steps_done += 1
        if sample > eps_threshold:
            with torch.no_grad():
                return self.policy_net(state).max(1).indices.view(1, 1)
        else:
            return torch.tensor([[self.env.action_space.sample()]], device = device, dtype = torch.long)
        
    def optimize_model(self):
        if len(self.memory) < self.batch_size:
            return 
        
        transitions = self.memory.sample(self.batch_size)
        batch = Transition(*zip(*transitions))

        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                                 batch.next_state)), device = device, dtype = torch.bool)
        
        non_final_next_states = torch.cat([s for s in batch.next_state
                                           if s is not None])
        
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        next_state_values = torch.zeros(self.batch_size, device = device)
        with torch.no_grad():
            next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1).values
        
        expected_state_action_values = (next_state_values * self.gamma) + reward_batch

        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        self.optimizer.zero_grad()
        loss.backward()

        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()

    def train(self, num_episodes):

        for i in range(num_episodes):
            # Initialize the environment and get its state
            state, info = self.env.reset()
            state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
            for t in count():
                action = self.select_action(state)
                observation, reward, terminated, truncated, _ = self.env.step(action.item())
                reward = torch.tensor([reward], device=device)
                done = terminated or truncated

                if terminated:
                    next_state = None
                else:
                    next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)

                # Store the transition in memory
                self.memory.push(state, action, next_state, reward)

                # Move to the next state
                state = next_state

                # Perform one step of the optimization (on the policy network)
                self.optimize_model()

                # Soft update of the target network's weights
                # θ′ ← τ θ + (1 −τ )θ′
                target_net_state_dict = self.target_net.state_dict()
                policy_net_state_dict = self.policy_net.state_dict()
                for key in policy_net_state_dict:
                    target_net_state_dict[key] = policy_net_state_dict[key] * self.tau + target_net_state_dict[key] * (1 - self.tau)
                self.target_net.load_state_dict(target_net_state_dict)

                if done:
                    # episode_durations.append(t + 1)
                    # plot_durations()
                    break


for i in range(1):
    env = SumoEnvironment(
        net_file="src/Intersection/2way-single-intersection/single-intersection.net.xml",
        route_file="src/Intersection/2way-single-intersection/single-intersection-vhvh.rou.xml",
        out_csv_name=f"data/dqn_2way_test_csv_run{i}",
        use_gui=True,
        num_seconds=6000,
    )

    dqn = DQNTrain(env = env)
    dqn.train(num_episodes=6000)