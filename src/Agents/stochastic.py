from src.agents.agent import Agent  

class StochasticModel(Agent):
    """
    A stochastic model that selects actions at random from the environment's action space.
    
    This model extends the Agent class and is designed to interact with an environment
    by taking random actions. It serves as a baseline to evaluate the performance of
    more sophisticated agents.
    
    Attributes:
        env (Environment): The environment the agent interacts with.
        action_space (Space): The space of possible actions the agent can take.
        model (StochasticModel): A reference to itself, indicating the model is this stochastic agent.
    """

    def __init__(self, num_experiments = 1):
        """
        Initializes the StochasticModel with a specified number of experiments.
        
        Parameters:
            num_experiments (int): The number of experiments to run with the model. Defaults to 1.
        """
        super().__init__('RANDOM')

    def setup_model(self, env):
        """
        Configures the model for interaction with the environment.
        
        Parameters:
            env (Environment): The environment the model will interact with.
        """
        self.env = env
        self.action_space = env.action_space
        self.model = self  # In this context, the model is the StochasticModel itself.

    def learn(self, total_timesteps: int):
        """
        Executes the learning process for the model over a given number of timesteps.
        
        In this model, learning consists of taking random actions within the environment
        and optionally saving the results to a CSV file if the episode ends.
        
        Parameters:
            total_timesteps (int): The total number of timesteps to run the learning process.
        """
        obs = self.env.reset()
        for _ in range(total_timesteps):
            action = self.action_space.sample()  # Selects a random action.
            obs, reward, terminated, truncated, info = self.env.step(action)
            if terminated or truncated:  # Checks if the episode has ended.
                self.env.save_csv(self.env.out_csv_name, self.env.episode)  # Saves the episode data if applicable.
                break
        self.env.close()

    def predict(self):
        """
        Predicts an action based on the current observation.
        
        For the StochasticModel, this method always selects an action at random, ignoring the observation.
        
        Parameters:
            observation: The current observation from the environment.
            state (optional): The current state for stateful agents. Not used in this model.
            mask (optional): A mask of valid actions. Not used in this model.
            deterministic (bool, optional): Whether to use a deterministic policy. Not applicable in this model.
            
        Returns:
            tuple: A randomly selected action and None (since the model does not maintain a state).
        """
        return self.action_space.sample(), None
