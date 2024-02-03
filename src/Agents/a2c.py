import sys
import torch  
import numpy as np  
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable

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

# hyperparameters
hidden_size = 256
learning_rate = 3e-4

class A2CNetwork(nn.Module):

    def __init__(self, num_inputs, num_actions, hidden_size):
        super(A2CNetwork, self).__init__()

        self.num_actions = num_actions
        self.critic_linear1 = nn.Linear(num_inputs, hidden_size)
        self.critic_linear2 = nn.Linear(hidden_size, 1)

        self.actor_linear1 = nn.Linear(num_inputs, hidden_size)
        self.actor_linear2 = nn.Linear(hidden_size, num_actions)
    
    def forward(self, state):
        state = Variable(torch.from_numpy(state).float().unsqueeze(0))
        value = F.relu(self.critic_linear1(state))
        value = self.critic_linear2(value)
        
        policy_dist = F.relu(self.actor_linear1(state))
        policy_dist = F.softmax(self.actor_linear2(policy_dist), dim=1)

        return value, policy_dist

class A2C():

    def __init__(self, env = None, lr = 0.001, n_steps = 6000, gamma = 0.99):
        self.env = env
        self.lr = lr
        self.n_steps = n_steps
        self.gamma = gamma

        self.num_inputs = self.env.observation_space.shape[0]
        self.num_outputs = self.env.action_space.n

        self.actor_critic = A2CNetwork(self.num_inputs, self.num_outputs, 128)
        self.ac_optimizer = optim.Adam(self.actor_critic.parameters(), lr = self.lr)
        
    def train(self, num_episodes):
        all_lengths = []
        average_lengths = []
        all_rewards = []
        entropy_term = 0

        for episode in range(num_episodes):
            log_probs = []
            values = []
            rewards = []

            state, _ = self.env.reset()
            for steps in range(self.n_steps):
                value, policy_dist = self.actor_critic.forward(state)
                value = value.detach().numpy()[0, 0]
                dist = policy_dist.detach().numpy() 

                action = np.random.choice(self.num_outputs, p = np.squeeze(dist))
                log_prob = torch.log(policy_dist.squeeze(0)[action])
                entropy = -np.sum(np.mean(dist) * np.log(dist))
                new_state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated

                rewards.append(reward)
                values.append(value)
                log_probs.append(log_prob)
                entropy_term += entropy
                state = new_state
                
                if done or steps == self.n_steps - 1:
                    Qval, _ = self.actor_critic.forward(new_state)
                    Qval = Qval.detach().numpy()[0, 0]
                    all_rewards.append(np.sum(rewards))
                    all_lengths.append(steps)
                    average_lengths.append(np.mean(all_lengths[-10:]))
                    break
            
            # compute Q values
            Qvals = np.zeros_like(values)
            for t in reversed(range(len(rewards))):
                Qval = rewards[t] + self.gamma * Qval
                Qvals[t] = Qval
    
            #update actor critic
            values = torch.FloatTensor(values)
            Qvals = torch.FloatTensor(Qvals)
            log_probs = torch.stack(log_probs)
            
            advantage = Qvals - values
            actor_loss = (-log_probs * advantage).mean()
            critic_loss = 0.5 * advantage.pow(2).mean()
            ac_loss = actor_loss + critic_loss + 0.001 * entropy_term

            self.ac_optimizer.zero_grad()
            ac_loss.backward()
            self.ac_optimizer.step()
        