import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from collections import deque, namedtuple
import random
from itertools import count
from agent import Agent


Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state'))

class DQNNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(DQNNetwork, self).__init__()
        # Define the network layers
        # Example architecture (modify as needed)
        self.fc1 = nn.Linear(state_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append(Transition(state, action, reward, next_state))

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)

class DQNAgent(Agent):
    def __init__(self, num_experiments=1):
        super().__init__('DQN', num_experiments)
        self.policy_net = None
        self.target_net = None
        self.memory = ReplayBuffer(10000)
        self.optimizer = None

        # Hyperparameters
        self.batch_size = 64
        self.gamma = 0.99
        self.epsilon_start = 1.0
        self.epsilon_end = 0.01
        self.epsilon_decay = 500
        self.step_count = 0

        
    def setup_model(self, env):
        self.env = env
        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.n

        self.policy_net = DQNNetwork(state_dim, action_dim)
        self.target_net = DQNNetwork(state_dim, action_dim)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=0.001)


    def select_action(self, state, epsilon):
        sample = random.random()
        eps_threshold = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * \
                        np.exp(-1. * self.step_count / self.epsilon_decay)
        self.step_count += 1

        if sample > eps_threshold:
            with torch.no_grad():
                return self.policy_net(state).max(1)[1].view(1, 1)
        else:
            return torch.tensor([[random.randrange(self.action_dim)]], dtype=torch.long)


    def optimize_model(self):
        if len(self.memory) < self.batch_size:
            return

        transitions = self.memory.sample(self.batch_size)
        batch = Transition(*zip(*transitions))

        # Compute a mask of non-final states and concatenate the batch elements
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                                batch.next_state)), dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the columns of actions taken
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        # Compute V(s_{t+1}) for all next states.
        next_state_values = torch.zeros(self.batch_size)
        next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0].detach()

        # Compute the expected Q values
        expected_state_action_values = (next_state_values * self.gamma) + reward_batch

        # Compute Huber loss
        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()


    def run_experiments(self):
        super().run_experiments()
        TARGET_UPDATE = 10

        for i in range(self.num_experiments):
            # Reset environment and state
            self.env.reset()
            state = torch.tensor([self.env.get_state()], dtype=torch.float)
            for t in count():
                # Select and perform an action
                action = self.select_action(state)
                next_state, reward, done, _ = self.env.step(action.item())
                reward = torch.tensor([reward], dtype=torch.float)

                # Observe new state
                if not done:
                    next_state = torch.tensor([next_state], dtype=torch.float)
                else:
                    next_state = None

                # Store the transition in memory
                self.memory.push(state, action, reward, next_state)

                # Move to the next state
                state = next_state

                # Perform one step of the optimization
                self.optimize_model()
                if done:
                    break

            # Update the target network
            if i % TARGET_UPDATE == 0:
                self.target_net.load_state_dict(self.policy_net.state_dict())

dqn_agent = DQNAgent(num_experiments=1)
dqn_agent.run_experiments()
'''
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
#         return self.layer3(x)'''