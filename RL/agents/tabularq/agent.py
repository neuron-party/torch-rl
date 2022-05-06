import numpy as np
import gym
import random
import pickle

from RL.agents.utils.misc import *
from RL.agents.random.agent import *


class TabularQ:
    def __init__(self, observation_space, action_space, **params):
        self.action_space = action_space
        self.observation_space = observation_space
        self.online = params['online']
        self.target_update_freq = params['target_update_freq']
        self.env = params['environment']
        
        self.epsilon = params['epsilon']
        self.epsilon_decay = params['epsilon_decay']
        self.epsilon_min = params['epsilon_min']
        self.gamma = params['gamma']
        self.learning_rate = params['learning_rate']
        self.lr_decay = params['lr_decay']
        self.lr_min = params['lr_min']
        
        self.bin_range = params['bin_range'] # list of tuples (high, low), or None
        self.numbins = params['numbins']
        self.sample_range = params['sample_range'] # tuple  
        
        self._build_agent()
        
    def _build_agent(self):
        self.memory = []
        self.bins = []
        self.bin_idx = []
        self.optim_steps = 0
        self.Q_matrix = np.random.uniform(low=-2, high=0, size=([self.numbins] * len(self.observation_space.high) + [self.action_space.n]))
        
        # user specified bins
        if self.bin_range is not None:
            for low, high in self.bin_range:
                self.bins.append(np.linspace(low, high, self.numbins))
        
        # auto-generated bins
        if self.sample_range is not None:
            idx = 0
            for low, high in zip(self.observation_space.low, self.observation_space.high):
                # set sampled lower and upper bound for bin range
                if (low < -1e10 or high > 1e10):
                    state_dis = sample(self.env)
                    lower, upper = self.sample_range
                    lower, upper = int(lower * len(state_dis[idx])), int(upper * (len(state_dis[idx])) - 1)
                    lower_bound, upper_bound = state_dis[idx][lower], state_dis[idx][upper]
                    self.bins.append(np.linspace(lower_bound, upper_bound, self.numbins))
                    self.bin_idx.append(idx)
                # set environment defined lower and upper bound for bin range
                else:
                    self.bins.append(np.linspace(low, high, self.numbins))
                idx += 1
        
    def get_action(self, state):
        state = self.get_discrete_state(state, self.bins, len(self.observation_space.high))
        if (self.epsilon > np.random.random()):
            return self.action_space.sample()
        else:
            return np.argmax(self.Q_matrix[state])
        
    def remember(self, state, action, reward, next_state, done):
        state = self.get_discrete_state(state, self.bins, len(self.observation_space.high))
        next_state = self.get_discrete_state(next_state, self.bins, len(self.observation_space.high))
        self.memory.append((state, action, reward, next_state))
        
    def learn(self):
        """
        Q-function: 
        Q(s, a) = R(s, a, s') + γ * max(Q(s', a'))
        
        Q-function iteration (scaled by a learning rate α):
        Q(s, a) = (1 - α)Q(s, a) + α[R(s, a, s') + γ * max(Q(s', a'))] 
        Q(s, a) = Q(s, a) + α[R(s, a, s') + γ * max(Q(s', a')) - Q(s, a)]
        
        Monte Carlo Learning:
        Q(s, a) = Q(s, a) + α * [G - Q(s, a)]
            where:
            G = R_t+1 + γ * R_t+2 + ... + γ^T-1 * R_T (total discounted rewards)
            T = termination timestep 
        
        Temporal Difference Learning:
        Q(s, a) = Q(s, a) + α * [R(s, a, s') + γ * max(Q(s', a')) - Q(s, a)]
            where:
            R(s, a, s') is the reward of taking action a in state s and ending up in state s'
            max(Q(s', a')) = Q-function value by acting optimally 
            max(Q(s', a')) - Q(s, a) = temporal difference error (TD error)
        """
        if self.target_update_freq is None: 
            for idx, (state, action, reward, next_state) in enumerate(self.memory):
                rewards = [t[2] for t in self.memory[idx:]]
                tdr = np.sum(get_total_discounted_rewards(rewards, self.gamma))
                index = state + (action, )
                self.Q_matrix[index] += self.learning_rate * (tdr - self.Q_matrix[index])      
        else:  
            for state, action, reward, next_state in self.memory:
                index = state + (action, )
                self.Q_matrix[index] += self.learning_rate * (reward + self.gamma * np.max(self.Q_matrix[next_state]) - self.Q_matrix[index])
            
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        
        if (self.lr_min and self.lr_decay) is not None:
            self.learning_rate = max(self.lr_min, self.lr_decay * self.learning_rate)
            
        self.optim_steps += 1
        self.memory = []
        
    def get_discrete_state(self, state, bins, observation_size):
        state_idx = []
        for i in range(observation_size):
            state_idx.append(np.digitize(state[i], bins[i]) - 1) # -1 will turn bin into index
        return tuple(state_idx)
    
    def save(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump({'Q_matrix': self.Q_matrix, 'bins': self.bins, 'bin_idx': self.bin_idx}, f)

    def load(self, filename):
        with open(filename, 'rb') as f:
            file = pickle.load(f)
            self.Q_matrix = file['Q_matrix']
            self.bins = file['bins']
            self.bin_idx = file['bin_idx']
            
        self.epsilon = 0.01 # for testing

'''
Samples the entire state of a given environment
Create new list for each state component 
    i.e [a0, b0, c0, d0], [a1, b1, c1, d1] --> [[a0, a1], [b0, b1], [c0, c1], [d0, d1]]
Returns sorted distributions of each state component 
'''
def sample(env, epochs=5000):
    env = gym.make(env)
    observation_space = env.observation_space
    action_space = env.action_space
    
    distributions = []
    
    for e in range(epochs):
        done = False
        state = env.reset()
        while not done:
            action = env.action_space.sample()
            next_state, reward, done, _ = env.step(action)
            state = next_state
        distributions.append(state)

    res = []
    for i in range(len(distributions[0])):
        temp = [item[i] for item in distributions]
        temp.sort()
        res.append(temp)
        
    return res