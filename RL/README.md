# Project Structure

- agents/  
  - agent_1/  
  - agent_2/  
  - ...  
  - agent_n/  
    - agent.py  
    - network.py  
  - utils/  
    - train.py  
    - test.py  
    - ...  

Each RL agent should have it's own directory. Within each agent directory, there should be a `network.py` and `agent.py`; all else  should be placed in utils. `network.py` contains deep neural network models defined as subclasses of `torch.nn.Module`. `agent.py` contains agent learning and processing functions that is required in `train.py` (more details defined below). This is an optimal architectural framework that allows for versatility in training various agent modules on different environments. 


# `agent.py` Structure

Below represents a generic layout for agents that uses a single network. The ??? denotes code to be filled out.

```
from RL.agents.Foo.network import FooNet
from RL.agents.utils.??? import ???

class FooAgent():
    def __init__(self, observation_space, action_space, **params):
        # Environment Params
        self.observation_space = observation_space
        self.action_space = action_space
        assert isinstance(observation_space, gym.spaces.???)
        self.state_dim = ???
        assert isinstance(action_space, gym.spaces.???)
        self.action_dim = ???
        
        # Agent Common Params (ordered by preference)
        self.online = ???
        self.gamma = params['gamma'] # Discount factor
        self.hidden_dim = params['hidden_dim']
        self.learning_rate = params['learning_rate']
        self.device = torch.device(params['device'])
        if params['dtype'] == 'float32':
            self.dtype = torch.float32
        if params['dtype'] == 'float64':
            self.dtype = torch.float64

        # Agent Specific Params (ordered alphabetically)
        ???
        
        self._build_agent()     
        
    def _build_agent(self):
        # Networks
        self.optim_steps = 0
        self.network = FooNet(self.state_dim, self.action_dim, self.hidden_dim, ???).to(self.device)
        self.optim = ???
        
        # Memory
        self.memory = ???
    
    def remember(self, state, action, reward, next_state, done):
        self.memory.???
    
    def get_action(self, state):
        state = torch.tensor(???)
        action = self.network(state)
        ???
        return action

    def learn(self):
        ???
        self.optim_steps += 1

    def save(self, filename):
        ??? # recommended to save the state_dict of both the optim and network

    def load(self, filename):
        ???
       
```

Note:
- ensure all torch tensors are created via `torch.tensor(???, dtype=self.dtype, device=self.device)