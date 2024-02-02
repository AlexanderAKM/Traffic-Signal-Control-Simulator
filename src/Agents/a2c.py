# from stable_baselines3 import A2C
# from agent import Agent

# class A2CAgent(Agent):
#     def __init__(self, num_experiments=1):
#         super().__init__('A2C', num_experiments)

#     def setup_model(self, env):
#         self.model = A2C(
#             policy="MlpPolicy",
#             env=env,
#             learning_rate=0.001,
#             n_steps=5,
#             gamma=0.99,
#             gae_lambda=1.0,
#             ent_coef=0.01,
#             vf_coef=0.5,
#             max_grad_norm=0.5,
#             verbose=1
#         )
        
# a2c_agent = A2CAgent()
# a2c_agent.run_experiments()



# '''
# # Read configuration (see config.json)
# with open('config.json', 'r') as config_file:
#     config = json.load(config_file)

# if "SUMO_HOME" in os.environ:
#     tools = os.path.join(os.environ["SUMO_HOME"], "tools")
#     sys.path.append(tools)
# else:
#     sys.exit("Please declare the environment variable 'SUMO_HOME'")
# import traci

# # Use paths from config file
# sys.path.append(config["project_base_path"])
# from src.Environment.env import SumoEnvironment

# # Number of experiments to run
# num_experiments = 50

# for i in range(num_experiments):
#     env = SumoEnvironment(
#         net_file="src/Intersection/2way-single-intersection/single-intersection.net.xml",
#         route_file="src/Intersection/2way-single-intersection/single-intersection-vhvh.rou.xml",
#         out_csv_name=f"data/A2C_2way_test_csv_run{i}",  # Unique file name for each run
#         use_gui=True,
#         num_seconds=6000,
#     )

#     model = A2C(
#         policy=MlpPolicy,
#         env=env,
#         learning_rate=0.001,
#         n_steps=5,
#         gamma=0.99,
#         gae_lambda=1.0,
#         ent_coef=0.01,
#         vf_coef=0.5,
#         max_grad_norm=0.5,
#         #tensorboard_log="./a2c_tensorboard/",  # Optional: For tensorboard logging
#         verbose=1
#     )


#     model.learn(total_timesteps=6000)

#     # After learning, you may want to save the model
#     # model.save(f"data/a2c_model_run{i}")

# # Additional code for saving results or handling outputs
# '''

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
from src.environment.env import SumoEnvironment

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

    def __init__(self, 
                 env = None,
                 lr = 0.001,
                 n_steps = 5,
                 gamma = 0.99,
                 gae_lambda = 1.0, 
                 ent_coef = 0.01, 
                 vf_coef = 0.5, 
                 max_grad_norm = 0.5):
        
        self.env = env
        self.lr = lr
        self.n_steps = n_steps
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm

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

            state, info = self.env.reset()
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
                    if episode % 10 == 0:                    
                        sys.stdout.write("episode: {}, reward: {}, total length: {}, average length: \
                                          {} \n".format(episode, np.sum(rewards), steps, average_lengths[-1]))
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


for i in range(1):
    env = SumoEnvironment(
        net_file="src/Intersection/2way-single-intersection/single-intersection.net.xml",
        route_file="src/Intersection/2way-single-intersection/single-intersection-vhvh.rou.xml",
        out_csv_name=f"data/a2c_2way_test_csv_run{i}",
        use_gui=True,
        num_seconds=6000,
    )

    a2c = A2C(env = env)
    a2c.train(num_episodes = 1)
        