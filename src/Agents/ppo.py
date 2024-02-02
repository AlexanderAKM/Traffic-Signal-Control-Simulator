from stable_baselines3 import PPO
from agent import Agent

class PPOAgent(Agent):
    def __init__(self, num_experiments=1):
        super().__init__('PPO', num_experiments)

    def setup_model(self, env):
        self.model = PPO(
            policy="MlpPolicy",
            env=env,
            learning_rate=0.001,
            n_steps=512,  # lower n_steps seems to be better (512 at least way better than 2048)
            batch_size=64,
            n_epochs=10,  # Number of epochs to update the model
            gamma=0.99,
            gae_lambda=0.95,  # Adjusted for PPO
            clip_range=0.2,  # Clipping parameter
            ent_coef=0.0,
            vf_coef=0.5,
            max_grad_norm=0.5,
            verbose=1,
        )
        
ppo_agent = PPOAgent()
ppo_agent.run_experiments()