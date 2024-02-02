from agent import Agent  

class StochasticModel(Agent):
    def __init__(self, num_experiments=1):
        super().__init__('Stochastic', num_experiments)

    def setup_model(self, env):
        self.env = env
        self.action_space = env.action_space
        self.model = self  # In this case, the model is the StochasticModel itself

    def learn(self, total_timesteps: int):
        obs = self.env.reset()
        for step in range(total_timesteps):
            action = self.action_space.sample()  # Random action
            obs, reward, terminated, truncated, info = self.env.step(action)
            if terminated or truncated:  # Check if the episode should end
                self.env.save_csv(self.env.out_csv_name, self.env.episode)  # Call the save_csv method
                break
        self.env.close()

    def predict(self, observation, state=None, mask=None, deterministic=False):
        return self.action_space.sample(), None