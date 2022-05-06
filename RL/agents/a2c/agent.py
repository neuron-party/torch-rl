import numpy as np
import gym
import copy
import torch
import torch.nn.functional as F

from RL.agents.a2c.network import *
from RL.agents.utils.misc import *


class A2CAgent():
    def __init__(self, observation_space, action_space, **params):       
        # Environment Params
        self.observation_space = observation_space
        self.action_space = action_space
        assert isinstance(observation_space, gym.spaces.Box)
        self.state_dim = observation_space.shape[0]
        if isinstance(action_space, gym.spaces.Discrete):
            self.discrete = True
            self.action_dim = action_space.n
        else:
            self.discrete = False
            self.action_dim = action_space.shape[0]
        
        # Agent Common Params (ordered by preference)
        self.online = False
        self.gamma = params['gamma'] # Discount factor
        self.hidden_dim = params['hidden_dim']
        self.learning_rate = params['learning_rate']
        self.device = torch.device(params['device'])
        if params['dtype'] == 'float32':
            self.dtype = torch.float32
        if params['dtype'] == 'float64':
            self.dtype = torch.float64
           
        # Agent Specific Params (ordered alphabetically)
        activation = params['activation']
        if activation == 'relu':
            self.activation = F.relu
        elif activation == 'tanh':
            self.activation = torch.tanh
        self.beta = params['beta']
        self.clip = params['clip']
        self.shared_network = params['shared_network']
        self.target_update_freq = params['target_update_freq'] # If None: MC-Learning; Else: TD-Learning
        
        self._build_agent()
        
    def _build_agent(self):
        # Networks
        self.optim_steps = 0
        self.network = ActorCriticNet(state_dim=self.state_dim, action_dim=self.action_dim, hidden_dim=self.hidden_dim, shared_network=self.shared_network, activation=self.activation, discrete=self.discrete).to(self.device)
        self.optim = torch.optim.Adam(self.network.parameters(), lr=self.learning_rate)
        if self.target_update_freq is not None:
            self.target_network = copy.deepcopy(self.network)
        
        # Memory
        self.memory = [] # a "trajectory", not a "replay buffer"; experiences are reset after episode
   
    def remember(self, state, action, reward, next_state, done):
        self.memory.append([state, action, reward, next_state, done])
    
    def get_action(self, state): 
        state = torch.tensor(state, dtype=self.dtype, device=self.device)
        pi, value = self.network(state)
        action = pi.sample().data.cpu().numpy() # TODO
        if self.discrete:
            action = action.item()
        else:
            action = np.clip(action, self.action_space.low, self.action_space.high)   
        return action
    
    def learn(self):
        """
        Notes:
            In stochastic policy gradients,
            The action `a` is a random variable with distribution pi(a|s) known as the policy, where s is the sate
            We denote `a = pi(s)` as random sampling an action `a` from pi(a|s) given state s
            Q(s, a) = r(s, a) + γ * V(s')
            V(s) = E[Q(s, .)] ; Expected Value of Q over all actions
                 = E[r(s, .) + γ * V(s')]
                 = ∑[pi(a|s) * r(s, a)] + γ * V(s')
                 = r + γ * V(s') ; denote r = ∑[pi(a|s) * r(s, a)]
                 = r + γ * [r + γ * V(s")]
                 = ∑ γ^i * r(i)
            A(s, a) = Q(s, a) - V(s) = [r(s, a) + γ * V(s')] - V(s) = [∑ γ^i * r(i)] - V(s)
                    = `total_discounted_reward - network(state)[1]`
        """
        states = torch.tensor([t[0] for t in self.memory], dtype=self.dtype, device=self.device)
        if self.discrete:
            actions = torch.tensor([t[1] for t in self.memory], dtype=torch.int64, device=self.device)
        else:
            actions = torch.tensor([t[1] for t in self.memory], dtype=self.dtype, device=self.device).view(-1, self.action_dim)
        rewards = [t[2] for t in self.memory]
        next_states = torch.tensor([t[3] for t in self.memory], dtype=self.dtype, device=self.device)
        dones = torch.tensor([t[4] for t in self.memory], dtype=torch.int64, device=self.device).view(-1, 1)
        self.memory = [] # reset memory

        pis, values = self.network(states)

        # Advantage
        if self.target_update_freq is None: # Monte Carlo Learning
            tdr = torch.tensor(get_total_discounted_rewards(rewards, self.gamma), dtype=self.dtype, device=self.device).view(-1, 1)
            advantage = tdr - values
        else: # Temporal Difference Learning
            rewards = torch.tensor(rewards, dtype=self.dtype, device=self.device).view(-1, 1)
            if self.optim_steps % self.target_update_freq == 0:
                self.target_network.load_state_dict(self.network.state_dict())
            next_pis, next_values = self.target_network(next_states)
            advantage = rewards + self.gamma*next_values*(1-dones) - values

        # Actor loss
        if self.discrete:
            log_probs = pis.log_prob(actions).view(-1, 1)
        else:
            log_probs = pis.log_prob(actions).view(-1, self.action_dim)
        entropies = pis.entropy()
        actor_loss = -(log_probs*advantage + self.beta*entropies).mean()

        # Critic loss
        critic_loss = advantage.pow(2).mean()

        loss = actor_loss + critic_loss
        self.optim.zero_grad()
        loss.backward()
        for param in self.network.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optim.step()

        self.optim_steps += 1
    
    def save(self, path):
        torch.save({'network': self.network.state_dict(),
                    'optim': self.optim.state_dict()},
                   path)

    def load(self, path):
        checkpoint = torch.load(path)
        
        self.network.load_state_dict(checkpoint['network'])
        self.optim.load_state_dict(checkpoint['optim'])
        if self.target_update_freq is not None:
            self.target_network = copy.deepcopy(self.network)