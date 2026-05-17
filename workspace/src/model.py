import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


# Autoencoder
class DFPAutoEncoder(nn.Module):
    def __init__(self, input_dim, latent_dim, hidden_dims, dropout: float=0.2):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        
        # encoder
        enc_layers = []
        prev = self.input_dim
        for h in hidden_dims:
            enc_layers += [nn.Linear(prev, h), nn.LayerNorm(h), nn.GELU(), nn.Dropout(dropout)]
            prev = h
        enc_layers.append(nn.Linear(prev, latent_dim))

        self.encoder = nn.Sequential(*enc_layers)

        # decoder
        dec_layers = []
        prev = latent_dim
        for h in reversed(hidden_dims):
            dec_layers += [nn.Linear(prev, h), nn.LayerNorm(h), nn.GELU(), nn.Dropout(dropout)]
            prev = h
        dec_layers.append(nn.Linear(prev, input_dim))

        self.decoder = nn.Sequential(*dec_layers)

    def forward(self, x):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat
    
    def encode(self, x):
        # to return latent representation only
        return self.encoder(x)