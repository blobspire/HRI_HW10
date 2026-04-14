import torch
import torch.nn as nn


# control policy
class Autoencoder(nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim, latent_dim):
        super(Autoencoder, self).__init__()

        ## define encoder

        ## define decoder

        ## helper functions
        # loss function
        self.mse_loss = nn.MSELoss()

    # encoder
    def encoder(self, state, action):
        x = torch.cat((state, action), 1)

    # decoder
    def decoder(self, state, z):
        x = torch.cat((state, z), 1)

    # autoencoder
    def forward(self, state, action):
        z = self.encoder(state, action)
        return self.decoder(state, z)
