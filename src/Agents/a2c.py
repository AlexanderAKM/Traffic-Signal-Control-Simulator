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

# Ensure SUMO tools are accessible by adding them to the system path
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

class A2CNetwork(nn.Module):
    """
    Implements the Actor-Critic network architecture.

    Attributes:
    - num_actions (int): Number of possible actions in the action space.
    - critic_linear1 (nn.Linear): First linear layer for the critic network.
    - critic_linear2 (nn.Linear): Second linear layer for the critic network, outputting state value.
    - actor_linear1 (nn.Linear): First linear layer for the actor network.
    - actor_linear2 (nn.Linear): Second linear layer for the actor network, outputting action probabilities.
    """
    
    def __init__(self, num_inputs, num_actions, hidden_size):
        """
        Initializes the A2C network with separate actor and critic pathways.

        Parameters:
        - num_inputs (int): The number of inputs in the input state space.
        - num_actions (int): The number of possible actions.
        - hidden_size (int): The number of neurons in the hidden layer.
        """
        super(A2CNetwork, self).__init__()

        self.num_actions = num_actions
        self.critic_linear1 = nn.Linear(num_inputs, hidden_size)
        self.critic_linear2 = nn.Linear(hidden_size, 1)

        self.actor_linear1 = nn.Linear(num_inputs, hidden_size)
        self.actor_linear2 = nn.Linear(hidden_size, num_actions)
    
    def forward(self, state):
        """
        Forward pass through the network.

        Parameters:
        - state (np.array): The current state of the environment.

        Returns:
        - Tuple[Tensor, Tensor]: The estimated state value and policy distribution.
        """
        state = Variable(torch.from_numpy(state).float().unsqueeze(0))
        value = F.relu(self.critic_linear1(state))
        value = self.critic_linear2(value)
        
        policy_dist = F.relu(self.actor_linear1(state))
        policy_dist = F.softmax(self.actor_linear2(policy_dist), dim=1)

        return value, policy_dist

class A2C():
    """
    Implements the Advantage Actor-Critic (A2C) algorithm.
    """

    def __init__(self, env = None, lr = 0.001, n_steps = 6000, gamma = 0.99):
        """
        Initializes the A2C agent.

        Parameters:
        - env(SumoEnvironment): The environment to interact with.
        - lr (float): Learning rate.
        - n_steps (int): Number of steps per episode.
        - gamma (float): Discount factor.
        """
        self.env = env
        self.lr = lr
        self.n_steps = n_steps
        self.gamma = gamma

        self.num_inputs = self.env.observation_space.shape[0]
        self.num_outputs = self.env.action_space.n

        self.actor_critic = A2CNetwork(self.num_inputs, self.num_outputs, 128)
        self.ac_optimizer = optim.Adam(self.actor_critic.parameters(), lr = self.lr)
        
    def train(self, num_episodes):
        """
        Trains the A2C agent for a specified number of episodes.

        During each episode, the agent selects actions based on the current policy, observes the reward
        and next state from the environment, and updates the policy by optimizing the actor and critic networks.
        The method tracks the length of each episode, the total rewards, and applies entropy regularization to encourage exploration.

        Parameters:
        - num_episodes (int): The number of episodes to train the agent.

        Updates:
            Updates the actor-critic network to optimize policy and value function estimation,
            aiming to maximize the expected return and minimize the value function error.
        """
        all_lengths = []  # Stores the length of each episode
        average_lengths = []  # Stores the average length of the last 10 episodes
        all_rewards = []  # Stores the total reward for each episode
        entropy_term = 0  # Accumulates entropy for all steps to encourage exploration

        for episode in range(num_episodes):
            log_probs = []  # Stores log probabilities of the actions taken
            values = []  # Stores value estimates from the critic
            rewards = []  # Stores rewards received at each step

            state, _ = self.env.reset()
            for steps in range(self.n_steps):
                value, policy_dist = self.actor_critic.forward(state)
                value = value.detach().numpy()[0, 0]  # Detach the value from the computation graph and convert to numpy
                dist = policy_dist.detach().numpy()  # Detach the policy distribution and convert to numpy

                action = np.random.choice(self.num_outputs, p = np.squeeze(dist))  # Sample an action from the policy distribution
                log_prob = torch.log(policy_dist.squeeze(0)[action])  # Compute the log probability of the selected action
                entropy = -np.sum(np.mean(dist) * np.log(dist))  # Calculate the entropy of the policy distribution to encourage exploration
                new_state, reward, terminated, truncated, _ = self.env.step(action)  # Take the action in the environment and observe the result
                done = terminated or truncated

                rewards.append(reward)
                values.append(value)
                log_probs.append(log_prob)
                entropy_term += entropy
                state = new_state
                
                if done or steps == self.n_steps - 1:  # Check if the episode is done or the max steps have been reached
                    Qval, _ = self.actor_critic.forward(new_state)
                    Qval = Qval.detach().numpy()[0, 0]  # Compute the final Q-value for the last state
                    all_rewards.append(np.sum(rewards))  # Accumulate total reward
                    all_lengths.append(steps)  # Record the length of the episode
                    average_lengths.append(np.mean(all_lengths[-10:]))  # Update the average length
                    break
            
            # Compute the Q values for each step
            Qvals = np.zeros_like(values)
            for t in reversed(range(len(rewards))):
                Qval = rewards[t] + self.gamma * Qval  # Update the Q-value backwards from the final state
                Qvals[t] = Qval

            # Compute the losses for the actor and critic networks
            values = torch.FloatTensor(values)
            Qvals = torch.FloatTensor(Qvals)
            log_probs = torch.stack(log_probs)
            
            advantage = Qvals - values  # Calculate the advantage
            actor_loss = (-log_probs * advantage).mean()  # Actor loss based on the policy gradient theorem
            critic_loss = 0.5 * advantage.pow(2).mean()  # Critic loss using mean squared error
            ac_loss = actor_loss + critic_loss + 0.001 * entropy_term  # Total loss with entropy regularization

            # Perform backpropagation and update the actor-critic networks
            self.ac_optimizer.zero_grad()
            ac_loss.backward()
            self.ac_optimizer.step()
        