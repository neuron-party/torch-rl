# Inspired by https://github.com/sfujim/TD3

import numpy as np
import gym
import copy
import torch

from RL.agents.td3.network import *
from RL.agents.utils.replay_buffer import *


class TD3Agent():
    def __init__(self, observation_space, action_space, **params):
        # Environment Params
        self.observation_space = observation_space
        self.action_space = action_space
        assert isinstance(observation_space, gym.spaces.Box)
        self.state_dim = observation_space.shape[0]
        assert isinstance(action_space, gym.spaces.Box)
        self.action_dim = action_space.shape[0] 
        
        # Agent Common Params (ordered by preference)
        self.online = True
        self.gamma = params['gamma'] # Discount factor
        self.hidden_dim = params['hidden_dim']
        self.learning_rate = params['learning_rate']
        self.device = torch.device(params['device'])
        if params['dtype'] == 'float32':
            self.dtype = torch.float32
        if params['dtype'] == 'float64':
            self.dtype = torch.float64

        # Agent Specific Params (ordered alphabetically)
        self.batch_size = params['batch_size'] # Batch size for both actor and critic
        self.expl_noise = params['expl_noise'] # Std of Gaussian exploration noise
        self.memory_maxlen = params['memory_maxlen']
        self.noise_clip = params['noise_clip'] # Range to clip target policy noise
        self.policy_freq = params['policy_freq'] # Frequency of delayed policy updates
        self.policy_noise = params['policy_noise'] # Noise added to target policy during critic update
        self.tau = params['tau'] # Target network update rate
        
        self._build_agent()
        
    def _build_agent(self):
        # Networks
        self.optim_steps = 0
        self.actor = Actor(self.state_dim, self.action_dim, self.hidden_dim, self.action_space).to(self.device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.learning_rate)
        self.critic = Critic(self.state_dim, self.action_dim, self.hidden_dim).to(self.device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.learning_rate)
        
        # Memory
        self.memory = ReplayBuffer(size=self.memory_maxlen)
    
    def remember(self, state, action, reward, next_state, done):
        self.memory.add(state, action, reward, next_state, done)
    
    def get_action(self, state):
        state = torch.tensor(state.reshape(1, -1), dtype=self.dtype, device=self.device)
        action = self.actor(state).cpu().data.numpy().flatten()
        location = (self.action_space.high + self.action_space.low) / 2
        scale = self.expl_noise * (self.action_space.high - self.action_space.low) / 2
        clipped_noisy_action = (action + np.random.normal(location, scale, size=self.action_dim)).clip(self.action_space.low, self.action_space.high)
        return clipped_noisy_action

    def learn(self):
        # Sample replay buffer 
        state, action, reward, next_state, done = self.memory.sample(self.batch_size)
        
        state = torch.tensor(state, dtype=self.dtype, device=self.device)
        action = torch.tensor(action, dtype=self.dtype, device=self.device)
        reward = torch.tensor(reward, dtype=self.dtype, device=self.device).reshape(-1, 1)
        next_state = torch.tensor(next_state, dtype=self.dtype, device=self.device)
        done = torch.tensor(done, dtype=self.dtype, device=self.device).reshape(-1, 1)

        with torch.no_grad():
            # Select action according to policy and add clipped noise
            noise = (torch.randn_like(action) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
            
            # Clip action to [action_space.low, action_space.high]
            next_action = (self.actor_target(next_state) + noise)
            low = torch.tensor(self.action_space.low, dtype=self.dtype, device=self.device)
            high = torch.tensor(self.action_space.high, dtype=self.dtype, device=self.device)
            next_action = torch.where(next_action < low, low, next_action)
            next_action = torch.where(next_action > high, high, next_action)

            # Compute the target Q value
            target_Q1, target_Q2 = self.critic_target(next_state, next_action)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward + (1 - done) * self.gamma * target_Q

        # Get current Q estimates
        current_Q1, current_Q2 = self.critic(state, action)

        # Compute critic loss
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Delayed policy updates
        if self.optim_steps % self.policy_freq == 0:

            # Compute actor losse
            actor_loss = -self.critic.Q1(state, self.actor(state)).mean()
            
            # Optimize the actor 
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Update the frozen target networks
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        
        self.optim_steps += 1

    def save(self, path):
        torch.save({'critic': self.critic.state_dict(),
                    'critic_optim': self.critic_optimizer.state_dict(),
                    'actor': self.actor.state_dict(),
                    'actor_optim': self.actor_optimizer.state_dict()},
                   path)

    def load(self, path):
        checkpoint = torch.load(path)
        
        self.critic.load_state_dict(checkpoint['critic'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optim'])
        self.critic_target = copy.deepcopy(self.critic)

        self.actor.load_state_dict(checkpoint['actor'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optim'])
        self.actor_target = copy.deepcopy(self.actor)