import torch
import torch.nn as nn
import torch.nn.functional as F


class ActorCriticNet(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim, shared_network, activation, discrete):
        super(ActorCriticNet, self).__init__()
        self.shared_network = shared_network
        self.activation = activation
        self.discrete = discrete
        
        if self.shared_network:
            self.shared = nn.Linear(state_dim, hidden_dim)
            self.actor1 = nn.Linear(hidden_dim, hidden_dim)
            self.critic1 = nn.Linear(hidden_dim, hidden_dim)
        else:
            self.actor1 = nn.Linear(state_dim, hidden_dim)
            self.critic1 = nn.Linear(state_dim, hidden_dim)

            
        self.actor2 = nn.Linear(hidden_dim, hidden_dim)
        self.actor3 = nn.Linear(hidden_dim, action_dim)

        self.critic2 = nn.Linear(hidden_dim, hidden_dim)
        self.critic3 = nn.Linear(hidden_dim, 1)
        
        if not self.discrete:
#             logstds_param = nn.Parameter(torch.zeros(action_dim)) #torch.full((self.model(action_dim,), 0.1)))
#             self.register_parameter("logstds", logstds_param)
            self.log_stds = nn.Parameter(torch.zeros(action_dim))
            
        
    def forward(self, state):
        if self.shared_network:
            shared = self.activation(self.shared(state))
            pi = self.activation(self.actor1(shared))
            value = self.activation(self.critic1(shared))
        else:
            pi = self.activation(self.actor1(state))
            value = self.activation(self.critic1(state))


            
        pi = self.activation(self.actor2(pi))
        pi = self.actor3(pi)
        if self.discrete:
            pi = F.softmax(pi, dim=pi.dim()-1)
            pi = torch.distributions.Categorical(pi)
        else:
#             stds = torch.clamp(self.logstds.exp(), 1e-3, 50)
            pi = torch.distributions.Normal(pi, self.log_stds.exp())


        value = self.activation(self.critic2(value))
        value = self.critic3(value)

        return pi, value