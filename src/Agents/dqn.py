import torch
import torch.nn as nn
import torch.nn.functional as F 
from collections import deque, namedtuple
import random
import torch.optim as optim
import math
from itertools import count

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

class ReplayMemory(object):
    """
    A cyclic buffer of bounded size that holds the transitions observed by the agent.
    
    Attributes:
    - memory (deque): A double-ended queue that stores transitions.
    - capacity (int): The maximum size of the memory.
    """

    def __init__(self, capacity):
        """
        Initializes the ReplayMemory.
        
        Parameters:
        - capacity (int): The size of the memory.
        """
        self.memory = deque([], maxlen=capacity)
    
    def push(self, *args):
        """
        Saves a transition.
        
        Parameters:
            *args: The components of a transition.
        """
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        """
        Randomly samples a batch of transitions from memory.
        
        Parameters:
        - batch_size (int): The size of the sample.
            
        Returns:
        - list: A sampled batch of transitions.
        """
        return random.sample(self.memory, batch_size)
    
    def __len__(self):
        """
        Returns the current size of internal memory.
        
        Returns:
        - int: The current size of memory.
        """
        return len(self.memory)
    

class DQNet(nn.Module):
    """
    Implements a Deep Q-Network.
    
    A neural network that approximates the Q-value function.
    """
    def __init__(self, n_observations, n_actions):
        """
        Initializes the DQN network with a single hidden layer.
        
        Parameters:
        - n_observations (int): The dimension of the observation space.
        - n_actions (int): The dimension of the action space.
        """
        super(DQNet, self).__init__()

        self.layer1 = nn.Linear(n_observations, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, n_actions)

    def forward(self, x):
        """
        Defines the computation performed at every call.
        
        Parameters:
            x (Tensor): The state input.
            
        Returns:
            Tensor: The Q-values for each action.
        """
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)

class DQN():
    """
    Implements the DQN algorithm.
    """
    def __init__(self, env=None, batch_size=128, gamma=0.99, eps_start=0.9, eps_end=0.05, eps_decay=1000, tau=0.005, lr=1e-4):
        """
        Initializes the DQN agent.
        
        Parameters:
        - env: The environment the agent interacts with.
        - batch_size (int): The size of the batch for optimization.
        - gamma (float): The discount factor for future rewards.
        - eps_start (float): The starting value of epsilon for epsilon-greedy exploration.
        - eps_end (float): The minimum value of epsilon after decay.
        - eps_decay (int): The rate at which epsilon decays.
        - tau (float): The interpolation parameter for updating the target network.
        - lr (float): The learning rate for the optimizer.
        """
        self.env = env
        self.batch_size = batch_size
        self.gamma = gamma
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay
        self.tau = tau
        self.lr = lr

        self.n_actions = env.action_space.n
        self.state, self.info = self.env.reset()
        self.n_observations = env.observation_space.shape[0]

        self.policy_net = DQNet(self.n_observations, self.n_actions).to(device)
        self.target_net = DQNet(self.n_observations, self.n_actions).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())

        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=self.lr, amsgrad=True)
        self.memory = ReplayMemory(1000)

        self.steps_done = 0

    def select_action(self, state):
        """
        Selects an action according to the epsilon-greedy-decay exploration strategy.
        
        Parameters:
            state (Tensor): The current state of the environment.
            
        Returns:
            Tensor: The action selected by the agent.
        """
        sample = random.random()
        eps_threshold = self.eps_end + (self.eps_start - self.eps_end) * \
            math.exp(-1. * self.steps_done / self.eps_decay)
        self.steps_done += 1
        if sample > eps_threshold:
            with torch.no_grad():
                return self.policy_net(state).max(1).indices.view(1, 1)
        else:
            return torch.tensor([[random.randrange(self.n_actions)]], device=device, dtype=torch.long)
        
    def optimize_model(self):
        """
        Performs a single optimization step on the policy network using a sampled batch from replay memory.

        This method updates the policy network by calculating the loss between predicted Q-values
        and the target Q-values derived from the Bellman equation. The loss is calculated only for
        non-final states to ensure proper bootstrapping of future state values. The optimization
        uses the AdamW optimizer and employs gradient clipping to prevent large gradients.
        """
        if len(self.memory) < self.batch_size:
            # If there aren't enough samples in the memory, do not perform optimization.
            return 
        
        transitions = self.memory.sample(self.batch_size)
        batch = Transition(*zip(*transitions))

        # Mask for filtering out transitions that lead to a final state.
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), device=device, dtype=torch.bool)
        
        # Concatenate all non-final next states.
        non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
        
        # Concatenate batches of states, actions, and rewards.
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        # Compute Q values for each state-action pair in the batch.
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        # Compute V(s_{t+1}) for all next states.
        next_state_values = torch.zeros(self.batch_size, device=device)
        with torch.no_grad():
            next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1).values
        
        # Compute the expected Q values.
        expected_state_action_values = (next_state_values * self.gamma) + reward_batch

        # Compute Huber loss between current Q values and target Q values.
        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        # Optimize the model.
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()

    def train(self, num_episodes):
        """
        Trains the DQN agent over a specified number of episodes.

        For each episode, the agent interacts with the environment to collect experiences,
        which are stored in the replay memory. After each action, the agent performs a single
        optimization step on the policy network. Additionally, the target network's weights
        are softly updated to slowly track the policy network.

        Parameters:
            num_episodes (int): The number of episodes to train the agent.
        """
        for i in range(num_episodes):
            state, info = self.env.reset()
            state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
            for t in count():
                action = self.select_action(state)
                observation, reward, terminated, truncated, _ = self.env.step(action.item())
                reward = torch.tensor([reward], device=device)
                done = terminated or truncated

                next_state = None if terminated else torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)

                self.memory.push(state, action, next_state, reward)
                state = next_state

                self.optimize_model()

                # Soft update the target network.
                target_net_state_dict = self.target_net.state_dict()
                policy_net_state_dict = self.policy_net.state_dict()
                for key in policy_net_state_dict:
                    target_net_state_dict[key] = policy_net_state_dict[key] * self.tau + target_net_state_dict[key] * (1 - self.tau)
                self.target_net.load_state_dict(target_net_state_dict)

                if done:
                    break
