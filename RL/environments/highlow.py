import numpy as np
import gym


class HighLow(gym.Env):
    def __init__(self):
        high = np.array([10, 10], dtype=np.float32)
        self.action_space = gym.spaces.Box(
            low=-high,
            high=high,
            dtype=np.float32
        )
        
        high = np.array([-np.inf, np.inf], dtype=np.float32)
        self.observation_space = gym.spaces.Box(
            low=-high,
            high=high,
            dtype=np.float32
        )
    
    def reset(self):
        self.state = np.array([0, 0], dtype=np.float32)
        self.prev_state = np.array([0, 0], dtype=np.float32)
        return self.state
    
    def render(self):
        pass
    
    def step(self, action):
        self.state =  self.state + action
        reward = (self.state[0] - self.prev_state[0]) + (self.prev_state[1] - self.state[1])
        self.prev_state = self.state
        return self.state, reward, False, {}


def test_HighLow():
    sum_rewards = []
    for i in range(100):
        env = HighLow()
        state = env.reset()
        done = False
        traj = []
        sum_reward = 0

        for i in range(100): 
            action = env.action_space.sample()
            next_state, reward, done, info = env.step(action)
            sum_reward += reward
    #         t = (state, action, reward, next_state, done)
    #         traj.append(t)
            state = next_state
        sum_rewards.append(sum_reward)
    print(sum_rewards)
    print(sum(sum_rewards)/100)