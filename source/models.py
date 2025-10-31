import numpy as np
import torch
from torch import nn
import math

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
        # I have no clue why the shape of this needs to be N H W 2
        # it complains if the N is different but the H and W don't even have to match the input
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
        net_input = self.neu_tex(uvs.reshape((1, 1, -1, 2)))
        return self.siren_net(net_input)
    
class BCNeuTex(nn.Module):
    def __init__(self, resolution, block_sidelength, block_embedding_channels, pixel_embedding_channels, mode='nearest', bound=10e-4):
        super().__init__()

        self.mode = mode
        self.block_sidelength = block_sidelength

        while resolution % self.block_sidelength != 0:
            self.block_sidelength -= 1

        self.width_in_blocks = int(resolution / self.block_sidelength)
        self.width_in_pixels = resolution
        
        self.block_embedding_channels = block_embedding_channels
        self.pixel_embedding_channels = pixel_embedding_channels

        block_tex = torch.empty((1, self.block_embedding_channels, self.width_in_blocks, self.width_in_blocks), dtype=torch.float32, requires_grad=True)
        torch.nn.init.uniform_(block_tex, a=-bound, b=bound)
        self.block_tex = nn.Parameter(block_tex)
        
        pixel_tex = torch.empty((1, self.pixel_embedding_channels, self.width_in_pixels, self.width_in_pixels), dtype=torch.float32, requires_grad=True)
        torch.nn.init.uniform_(pixel_tex, a=-bound, b=bound)
        self.pixel_tex = nn.Parameter(pixel_tex)
    
    def forward(self, uvs):
        block_channels = torch.nn.functional.grid_sample(self.block_tex, uvs, mode=self.mode, padding_mode='zeros', align_corners=True)
        block_channels = block_channels.permute(0, 2, 3, 1).reshape(-1, self.block_embedding_channels)

        pixel_channels = torch.nn.functional.grid_sample(self.pixel_tex, uvs, mode=self.mode, padding_mode='zeros', align_corners=True)
        pixel_channels = pixel_channels.permute(0, 2, 3, 1).reshape(-1, self.pixel_embedding_channels)

        channels = torch.cat((block_channels, pixel_channels), 1)

        return channels
    
class SirenBCTex(nn.Module):
    def __init__(self, resolution, block_sidelength, block_embedding_channels, pixel_embedding_channels, hidden_features, hidden_layers, out_features, mode='nearest', outermost_linear=False, 
                 first_omega_0=30, hidden_omega_0=30):
        super().__init__()
        self.resolution = resolution
        self.pixel_embedding_channels = pixel_embedding_channels

        self.neu_tex = BCNeuTex(resolution, block_sidelength, block_embedding_channels, pixel_embedding_channels, mode=mode)

        self.siren_net = Siren(block_embedding_channels + pixel_embedding_channels, hidden_features, hidden_layers, out_features, outermost_linear, 
                               first_omega_0, hidden_omega_0)
        
    def forward(self, uvs):
        net_input = self.neu_tex(uvs.reshape((1, 1, -1, 2)))
        return self.siren_net(net_input)
