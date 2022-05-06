import numpy as np
import gym
import copy
import torch
import random
from collections import deque
from skimage import transform

from RL.agents.dqn.network import *
from RL.agents.utils.replay_buffer import *

def scale_lumininance(obs):
    return np.dot(obs[...,:3], [0.299, 0.587, 0.114])

def resize(obs):
    return transform.resize(obs, (84, 84))

def normalize(obs):
    return obs / 255
    # return (obs - obs.mean()) / np.std(obs)

def preprocess_observation(obs):
    obs_proc = scale_lumininance(obs)
    obs_proc = resize(obs_proc)
    obs_proc = normalize(obs_proc)
    return obs_proc

class DQNCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )
        # make a dynamic function to calculate output size, for now use 3136
        self.fc1 = nn.Linear(3136, 512) # tune parameters later, just testing functionality rn
        self.fc2 = nn.Linear(512, 4) # set to env.action_space.n later
        
    def forward(self, x):
        # x = torch.squeeze(x, 1) # [32, 1, 4, 84, 84] -> [32, 4, 84, 84]
        x = self.cnn(x)
        x = x.flatten()
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
class DQNAgent:
    def __init__(self, observation_space, action_space, **params):
        # Environment Params
        self.observation_space = observation_space
        self.action_space = action_space
        assert isinstance(observation_space, gym.spaces.Box)
        self.state_dim = observation_space.shape[0]
        assert isinstance(action_space, gym.spaces.Discrete)
        self.action_dim = action_space.n
        
        # Agent Common Params (ordered by preference)
        self.online = params['online']
        self.gamma = params['gamma'] # Discount factor
        self.hidden_dim = params['hidden_dim']
        self.learning_rate = params['learning_rate']
        self.device = torch.device(params['device'])
        if params['dtype'] == 'float32':
            self.dtype = torch.float32
        if params['dtype'] == 'float64':
            self.dtype = torch.float64
        
        # Agent Specific Params (ordered alphabetically)
        self.batch_size = params['batch_size'] # size to sample from memory
        self.clip = params['clip']
        self.dueling = params['dueling']
        self.epsilon = params['epsilon'] # initial exploration rate
        self.epsilon_min = params['epsilon_min']
        self.epsilon_decay = params['epsilon_decay']
        self.memory_maxlen = params['memory_maxlen']
        self.per = params['per']
        assert not (self.per == True and self.online == False)
        if self.per:
            self.memory_alpha = params['memory_alpha']
            self.memory_beta = params['memory_beta'] # initial beta; approaches 1 as epsilon approaches 0
        self.target_update_freq = params['target_update_freq'] # double network

        self.tau = params['tau']
        
        
        self._build_agent()
  
    def _build_agent(self):
        """
        initalize deque of size tau (4)
        before feeding into nn remember to np.stack() and convert into tensor (send to device too)
            this input becomes (batch_size, channels (frames), height, width)
        
        current_state_buffer -> last frame is current state
        next_state_buffer -> last frame is next state
        holistic_state_buffer -> current_state_buffer, next_state_buffer
            this is what you will use to calculate Q_predicted and Q_actual
            MDP tuple is (state, action, reward, next_state, done)
            so this buffer will hold:
                (current_state_buffer, action, reward, next_state_buffer, done)
                configure def remember() accordingly
                
        everything else should be the same?
        removed target_network 
        
        CURRENT ERROR:
        when agent gets action, there are no batches, so feeding it through NN doesnt work because the input size is different now for fully connected layers
        """
        # Networks
        self.optim_steps = 0
        self.network = DQNCNN().to(self.device) # cnndqn 
        self.optim = torch.optim.Adam(self.network.parameters(), lr=self.learning_rate)
        if self.target_update_freq is not None:
            self.target_network = copy.deepcopy(self.network)
        """
        building buffers
        """
        self.curr_state_buffer = deque(maxlen=self.tau)
        self.next_state_buffer = deque(maxlen=self.tau)
        for i in range(self.tau):
            self.curr_state_buffer.append(np.zeros((84, 84))) # configure this to dynamically adapt to input size
            self.next_state_buffer.append(np.zeros((84, 84)))
        
        # Memory
        if self.per:
            self.memory = PrioritizedReplayBuffer(size=self.memory_maxlen, alpha=self.memory_alpha)
        else:
            self.memory = ReplayBuffer(size=self.memory_maxlen)

    def remember(self, state, action, reward, next_state, done):
        # self.memory.add(state, action, reward, next_state, done)
        state = preprocess_observation(state)
        next_state = preprocess_observation(next_state)
        self.curr_state_buffer.append(state)
        self.next_state_buffer.append(next_state)
        transformed_currstate = np.stack([self.curr_state_buffer])
        transformed_nextstate = np.stack([self.next_state_buffer])
        self.memory.add(transformed_currstate, action, reward, transformed_nextstate, done)
        
    def get_action(self, state):
        """
        typically, calculate the best action given this state whether thats tabq matrix lookup or feed through nn
        -> env.step -> next_state
        1. get the action first
        2. get complete MDP tuple
        3. remember and learn
        so we will follow this same pattern
        """
        
        if random.random() <= self.epsilon:
            action = int(random.random()*self.action_dim) # random action
        # else:
        #     state = torch.tensor(state, dtype=self.dtype, device=self.device)
        #     act_values = self.network(state)
        #     action = np.argmax(act_values.data.cpu().numpy()) # predicted action
        # return action
        else:
            state = preprocess_observation(state)
            self.curr_state_buffer.append(state)
            state = np.stack([self.curr_state_buffer])
            state = torch.FloatTensor(state).to(self.device)
            act_values = self.network(state)
            action = np.argmax(act_values.data.cpu().numpy())
        return action
    
    def learn(self):
        n = self.batch_size
        if self.per:
            minibatch = self.memory.sample(batch_size=n, beta=max(self.memory_beta, 1-self.epsilon))
            weights = torch.tensor(minibatch[5], dtype=self.dtype, device=self.device).reshape(-1, 1)
            idx = minibatch[6]
        else:
            minibatch = self.memory.sample(batch_size=n)
        s0 = torch.tensor(minibatch[0], dtype=self.dtype, device=self.device)
        a0 = torch.tensor(minibatch[1], dtype=torch.int64, device=self.device).reshape(-1, 1)
        r = torch.tensor(minibatch[2], dtype=self.dtype, device=self.device).reshape(-1, 1)
        s1 = torch.tensor(minibatch[3], dtype=self.dtype, device=self.device)
        d = torch.tensor(minibatch[4], dtype=torch.int64, device=self.device).reshape(-1, 1)

        # let subscripts 0 := current and 1 := next
        # let Q' be the double network ("target_network") that lags behind Q
        # import pdb; pdb.set_trace()
        """
        changed torch.gather to 0
        changed a lot of squeeze/gather
        
        changes:
        a0 = a0.squeeze()
        a1 = Q_s1.argmax(dim=0).reshape(-1, 1)
        a1 = a1.squeeze()
        Q_actual = Q_actual.reshape((-1, ))
        
        LOOP THROUGH MDP TUPLE TO LEARN
        
        for s, a, sp in zip(s0, a0, s1):
            Q_predicted = torch.gather(self.network(s), 0, a)
            Q_s1 = self.network(sp)
            a1 = Q_s1.argmax(dim=0).reshape(1)
            Q_actual = r + self.gamma*(1-d)*torch.gather(Q_s1, 0, a1)
            errors = (Q_actual - Q_predicted)
            loss = errors.pow(2).mean()
            
            # put optim/backprop outside of loop? idk
            
            self.optim.zero_grad()
            loss.backward()
            self.optim.step()
        """
        #a0 = a0.squeeze() 
        for s, a, sp in zip(s0, a0, s1):
            s, a, sp = s.to(self.device), a.to(self.device), sp.to(self.device)
            Q_predicted = torch.gather(self.network(s), 0, a)
            Q_s1 = self.network(sp)
            a1 = Q_s1.argmax(dim=0).reshape(1)
            Q_actual = r + self.gamma*(1-d)*torch.gather(Q_s1, 0, a1)
            errors = (Q_actual - Q_predicted)
            loss = errors.pow(2).mean()
            """
            put optim/backprop outside of loop? idk
            """
            self.optim.zero_grad()
            loss.backward()
            self.optim.step()
            
#         Q_predicted = torch.gather(self.network(s0), 0, a0) # Q(s=s0, a=a0)
#         Q_s1 = self.network(s1) # Q(s=s1, a=.)
#         a1 = Q_s1.argmax(dim=0).reshape(-1, 1) # a1 = argmax_a Q(s=s1, a=.) ; `a1` is always chosen by original Q
#         # if self.target_update_freq is not None:
#         #     if self.optim_steps % self.target_update_freq == 0:
#         #         self.target_network.load_state_dict(self.network.state_dict())
#         #     Q_s1 = self.target_network(s1) # Q'(s=s1, a=.) ; if using double, Q' will be used to eval `a1` chosen by original Q 
            
#         a1 = a1.squeeze()
        
#         Q_actual = r + self.gamma*(1-d)*torch.gather(Q_s1, 0, a1) # Q_actual = r + gamma*(1-d)*{Q or Q'}(s=s1, a=a1)
        
#         Q_actual = Q_actual.reshape((-1, ))
        
#         errors = (Q_actual - Q_predicted)
        
#         if self.per: # TODO: results show it is not working as intended
#             priorities = np.abs(errors.data.cpu().numpy()) + 1e-3 # check dtype
#             self.memory.update_priorities(idx, priorities)
#             loss = (errors.pow(2)*weights).mean()
#         else:
#             loss = errors.pow(2).mean()
        
#         self.optim.zero_grad()
#         loss.backward()
        
#         if self.clip:
#             for param in self.network.parameters():
#                 param.grad.data.clamp_(-1, 1)
                
#         self.optim.step()
            
        if self.epsilon > self.epsilon_min:
            self.epsilon = max(self.epsilon*self.epsilon_decay, self.epsilon_min)
        
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
        
        self.epsilon = .01 # for testing


# class DQNAgent:
#     def __init__(self, observation_space, action_space, **params):
#         # Environment Params
#         self.observation_space = observation_space
#         self.action_space = action_space
#         assert isinstance(observation_space, gym.spaces.Box)
#         self.state_dim = observation_space.shape[0]
#         assert isinstance(action_space, gym.spaces.Discrete)
#         self.action_dim = action_space.n
        
#         # Agent Common Params (ordered by preference)
#         self.online = params['online']
#         self.gamma = params['gamma'] # Discount factor
#         self.hidden_dim = params['hidden_dim']
#         self.learning_rate = params['learning_rate']
#         self.device = torch.device(params['device'])
#         if params['dtype'] == 'float32':
#             self.dtype = torch.float32
#         if params['dtype'] == 'float64':
#             self.dtype = torch.float64
        
#         # Agent Specific Params (ordered alphabetically)
#         self.batch_size = params['batch_size'] # size to sample from memory
#         self.clip = params['clip']
#         self.dueling = params['dueling']
#         self.epsilon = params['epsilon'] # initial exploration rate
#         self.epsilon_min = params['epsilon_min']
#         self.epsilon_decay = params['epsilon_decay']
#         self.memory_maxlen = params['memory_maxlen']
#         self.per = params['per']
#         assert not (self.per == True and self.online == False)
#         if self.per:
#             self.memory_alpha = params['memory_alpha']
#             self.memory_beta = params['memory_beta'] # initial beta; approaches 1 as epsilon approaches 0
#         self.target_update_freq = params['target_update_freq'] # double network

#         self._build_agent()
  
#     def _build_agent(self):
#         # Networks
#         self.optim_steps = 0
#         self.network = DQN(state_dim=self.state_dim, action_dim=self.action_dim, hidden_dim=self.hidden_dim, dueling=self.dueling).to(self.device)
#         self.optim = torch.optim.Adam(self.network.parameters(), lr=self.learning_rate)
#         if self.target_update_freq is not None:
#             self.target_network = copy.deepcopy(self.network)
        
#         # Memory
#         if self.per:
#             self.memory = PrioritizedReplayBuffer(size=self.memory_maxlen, alpha=self.memory_alpha)
#         else:
#             self.memory = ReplayBuffer(size=self.memory_maxlen)

#     def remember(self, state, action, reward, next_state, done):
#         self.memory.add(state, action, reward, next_state, done)
        
#     def get_action(self, state):
#         if random.random() <= self.epsilon:
#             action = int(random.random()*self.action_dim) # random action
#         else:
#             state = torch.tensor(state, dtype=self.dtype, device=self.device)
#             act_values = self.network(state)
#             action = np.argmax(act_values.data.cpu().numpy()) # predicted action
#         return action
    
#     def learn(self):
#         n = self.batch_size
#         if self.per:
#             minibatch = self.memory.sample(batch_size=n, beta=max(self.memory_beta, 1-self.epsilon))
#             weights = torch.tensor(minibatch[5], dtype=self.dtype, device=self.device).reshape(-1, 1)
#             idx = minibatch[6]
#         else:
#             minibatch = self.memory.sample(batch_size=n)
#         s0 = torch.tensor(minibatch[0], dtype=self.dtype, device=self.device)
#         a0 = torch.tensor(minibatch[1], dtype=torch.int64, device=self.device).reshape(-1, 1)
#         r = torch.tensor(minibatch[2], dtype=self.dtype, device=self.device).reshape(-1, 1)
#         s1 = torch.tensor(minibatch[3], dtype=self.dtype, device=self.device)
#         d = torch.tensor(minibatch[4], dtype=torch.int64, device=self.device).reshape(-1, 1)

#         # let subscripts 0 := current and 1 := next
#         # let Q' be the double network ("target_network") that lags behind Q
#         Q_predicted = torch.gather(self.network(s0), 1, a0) # Q(s=s0, a=a0)
#         Q_s1 = self.network(s1) # Q(s=s1, a=.)
#         a1 = Q_s1.argmax(dim=1).reshape(-1, 1) # a1 = argmax_a Q(s=s1, a=.) ; `a1` is always chosen by original Q
#         if self.target_update_freq is not None:
#             if self.optim_steps % self.target_update_freq == 0:
#                 self.target_network.load_state_dict(self.network.state_dict())
#             Q_s1 = self.target_network(s1) # Q'(s=s1, a=.) ; if using double, Q' will be used to eval `a1` chosen by original Q 
        
#         Q_actual = r + self.gamma*(1-d)*torch.gather(Q_s1, 1, a1) # Q_actual = r + gamma*(1-d)*{Q or Q'}(s=s1, a=a1)
        
#         errors = (Q_actual - Q_predicted)
        
#         if self.per: # TODO: results show it is not working as intended
#             priorities = np.abs(errors.data.cpu().numpy()) + 1e-3 # check dtype
#             self.memory.update_priorities(idx, priorities)
#             loss = (errors.pow(2)*weights).mean()
#         else:
#             loss = errors.pow(2).mean()
        
#         self.optim.zero_grad()
#         loss.backward()
        
#         if self.clip:
#             for param in self.network.parameters():
#                 param.grad.data.clamp_(-1, 1)
                
#         self.optim.step()
            
#         if self.epsilon > self.epsilon_min:
#             self.epsilon = max(self.epsilon*self.epsilon_decay, self.epsilon_min)
        
#         self.optim_steps += 1

#     def save(self, path):
#         torch.save({'network': self.network.state_dict(),
#                     'optim': self.optim.state_dict()},
#                    path)

#     def load(self, path):
#         checkpoint = torch.load(path)
        
#         self.network.load_state_dict(checkpoint['network'])
#         self.optim.load_state_dict(checkpoint['optim'])
#         if self.target_update_freq is not None:
#             self.target_network = copy.deepcopy(self.network)
        
#         self.epsilon = .01 # for testing