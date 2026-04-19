import torch
import torch.nn as nn


# control policy
class Autoencoder(nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim, latent_dim):
        super(Autoencoder, self).__init__()

        # Convert state and action to latent dim
        self.encoder_1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.encoder_2 = nn.Linear(hidden_dim, hidden_dim)
        self.encoder_3 = nn.Linear(hidden_dim, latent_dim)

        ## Conditional autodecoder: uses state and action
        self.decoder_1 = nn.Linear(latent_dim + state_dim, hidden_dim)
        self.decoder_2 = nn.Linear(hidden_dim, hidden_dim)
        self.decoder_3 = nn.Linear(hidden_dim, action_dim)

        ## helper functions
        # loss function
        self.mse_loss = nn.MSELoss()

    # encoder
    def encoder(self, state, action):
        x = torch.cat((state, action), 1)
        x = torch.tanh(self.encoder_1(x)) # Prof recommend tanh bc it worked better than relu in his grad
        x = torch.tanh(self.encoder_2(x))
        return torch.tanh(self.encoder_3(x)) # tanh activation: -1 to 1 (helps us know where the latest space is)


    # decoder
    def decoder(self, state, z): # z is the latent variable from the bottleneck (low dimensional action representation)
        x = torch.cat((state, z), 1)
        x = torch.tanh(self.decoder_1(x)) # Prof recommend tanh bc it worked better than relu in his grad
        x = torch.tanh(self.decoder_2(x))
        return self.decoder_3(x) # Don't need bound on output space

    # autoencoder
    def forward(self, state, action):
        z = self.encoder(state, action) # Encode action to latent space
        return self.decoder(state, z) # Decode action from latent space
