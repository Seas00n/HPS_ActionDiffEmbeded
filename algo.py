import numpy as np
import torch
import torch.nn as nn   
import torch.optim as optim
import torch.nn.functional as F 
from collections import defaultdict
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 

class ActChangeNN(nn.Module):
    def __init__(self, state_dim, context_dim, action_dim, hidden_dim=128):
        super(ActChangeNN, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim + context_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh()
        )
    
    def forward(self, state, context):
        x = torch.cat([state, context], dim=-1)
        return self.fc(x)


class VAE(nn.Module):
    def __init__(self, context_dim, action_dim, latent_dim, hidden_dim=128, max_action=1.0):
        super(VAE, self).__init__()
        self.fe = nn.Sequential(
            nn.Linear(context_dim+action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.fmean = nn.Linear(hidden_dim, latent_dim)
        self.flog_std = nn.Linear(hidden_dim, latent_dim)
        self.fd = nn.Sequential(
            nn.Linear(latent_dim+context_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Sigmoid()
        )
        self.max_action = max_action
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        z = mu+eps*std
        return z
    
    def forward(self, context, action):
        z = self.fe(torch.cat([context, action],1))
        mean = self.fmean(z)
        logvar = self.flog_std(z).clamp(-4,15)
        z = self.reparameterize(mean, logvar)
        u = self.fd(torch.cat([context, z],1))*self.max_action
        return u, mean, logvar

    def encode(self, context, action):
        z = self.fe(torch.cat([context, action],1))
        mean = self.fmean(z)
        return mean
    
    def decode(self, context, z):
        u = self.fd(torch.cat([context, z],1))*self.max_action
        return u


class LatentDynNN(nn.Module):
    def __init__(self, state_dim, context_dim, latent_dim, hidden_dim=128) -> None:
        super(LatentDynNN, self).__init__()
        self.fe = nn.Sequential(
            nn.Linear(state_dim+context_dim+latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
            nn.Tanh()
        )
    
    def forward(self, state_diff, context, z):
        u = self.fe(torch.cat([state_diff, context, z],1))
        return u
    

