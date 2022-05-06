import torch
import torch.nn as nn
import torch.nn.functional as F

class DQN(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim, dueling):
        super(DQN, self).__init__()
        self.dueling = dueling
        
        self.layer1 = nn.Linear(state_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, hidden_dim)
        
        if self.dueling:
            self.value = nn.Linear(hidden_dim, 1)
            self.advantage = nn.Linear(hidden_dim, action_dim)
        else:
            self.layer3 = nn.Linear(hidden_dim, action_dim)
        
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        if self.dueling:        
            v = self.value(x)
            a = self.advantage(x)
            q = v + a - a.mean(dim=a.dim()-1, keepdim=True)
        else:
            q = self.layer3(x)
        return q