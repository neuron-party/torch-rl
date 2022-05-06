import gym
import numpy as np
from RL.agents.random.agent import *
from RL.agents import *

def get_total_discounted_rewards(rewards, gamma):
    """
    parameters:
        rewards (r) = [1, 2, 3]
        gamma (γ) = .95
    
    R(t) = ∑ γ^i * r(i+t) from i=0 to inf
    
    returns:
        total_discounted_rewards (tdr) = [R(0), R(1), R(2)]
        = [(.95^0)(1)+(.95^1)(2)+(.95^2)(3), (.95^0)(2)+(.95^1)(3), (.95^0)(3)]
        = [5.6075, 4.85, 3]
    """
    tdr = 0
    total_discounted_rewards = []
    for reward in rewards[::-1]:
        tdr = reward + tdr * gamma
        total_discounted_rewards.append(tdr)
    
    return total_discounted_rewards[::-1]

def sample(env, epochs=5000):
    env = gym.make(env)
    observation_space = env.observation_space
    action_space = env.action_space
    agent = RandomAgent(observation_space, action_space)
    
    distributions = []
    
    for e in range(epochs):
        done = False
        state = env.reset()
        while not done:
            action = agent.get_action(state)
            next_state, reward, done, _ = env.step(action)
            state = next_state
        distributions.append(state)

    res = []
    for i in range(len(distributions[0])):
        temp = [item[i] for item in distributions]
        temp.sort()
        res.append(temp)
        
    return res