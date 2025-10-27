import numpy as np
import torch
from torch import nn

class SineLayer(nn.Module):    
    def __init__(self, in_features, out_features, bias=True,
                 is_first=False, omega_0=30):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first
        
        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        
        self.init_weights()
    
    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_features, 
                                             1 / self.in_features)      
            else:
                self.linear.weight.uniform_(-np.sqrt(6 / self.in_features) / self.omega_0, 
                                             np.sqrt(6 / self.in_features) / self.omega_0)
        
    def forward(self, x):
        return torch.sin(self.omega_0 * self.linear(x))

class Siren(nn.Module):
    def __init__(self, in_features, hidden_features, hidden_layers, out_features, outermost_linear=False, 
                 first_omega_0=30, hidden_omega_0=30.):
        super().__init__()
        
        self.net = []
        self.net.append(SineLayer(in_features, hidden_features, 
                                  is_first=True, omega_0=first_omega_0))

        for i in range(hidden_layers):
            self.net.append(SineLayer(hidden_features, hidden_features, 
                                      is_first=False, omega_0=hidden_omega_0))

        if outermost_linear:
            final_linear = nn.Linear(hidden_features, out_features)
            
            with torch.no_grad():
                final_linear.weight.uniform_(-np.sqrt(6 / hidden_features) / hidden_omega_0, 
                                              np.sqrt(6 / hidden_features) / hidden_omega_0)
                
            self.net.append(final_linear)
        else:
            self.net.append(SineLayer(hidden_features, out_features, 
                                      is_first=False, omega_0=hidden_omega_0))
        
        self.net = nn.Sequential(*self.net)
    
    def forward(self, x):
        return self.net(x)

class NeuTex(nn.Module):
    def __init__(self, resolution, embedding_channels, bound=10e-4):
        super().__init__()
        self.resolution = resolution
        self.embedding_channels = embedding_channels
        tex = torch.empty((1, embedding_channels, resolution, resolution), dtype=torch.float32, requires_grad=True)
        torch.nn.init.uniform_(tex, a=-bound, b=bound)
        self.tex = nn.Parameter(tex)
    
    def forward(self, uvs):
        channels = torch.nn.functional.grid_sample(self.tex, uvs, mode='nearest', padding_mode='zeros', align_corners=True)
        channels = channels.permute(0, 2, 3, 1).reshape(-1, self.embedding_channels)

        return channels

class SirenTex(nn.Module):
    def __init__(self, resolution, embedding_channels, hidden_features, hidden_layers, out_features, outermost_linear=False, 
                 first_omega_0=30, hidden_omega_0=30):
        super().__init__()
        self.resolution = resolution
        self.embedding_channels = embedding_channels

        self.neu_tex = NeuTex(resolution, embedding_channels)

        self.siren_net = Siren(embedding_channels, hidden_features, hidden_layers, out_features, outermost_linear, 
                               first_omega_0, hidden_omega_0)
        
    def forward(self, uvs):
        net_input = self.neu_tex(uvs)
        return self.siren_net(net_input), net_input
