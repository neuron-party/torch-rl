class RandomAgent():
    def __init__(self, observation_space, action_space, **params):
        # Environment Params
        self.observation_space = observation_space
        self.action_space = action_space
        
        # Agent Common Params
        self.online = False
        
        self._build_agent()
        
    def _build_agent(self):
        self.optim_steps = 0
        self.memory = []
    
    def remember(self, state, action, reward, next_state, done):
        pass
    
    def get_action(self, state):
        return self.action_space.sample()

    def learn(self):
        pass

    def save(self, filename):
        pass

    def load(self, filename):
        pass