import torch
import torch.nn as nn
from torch.nn import functional as F

class VAE(nn.Module):
    def __init__(self, one_side=28, z_dim=2):
        super().__init__()
        self.one_side = one_side
        self.fc1 = nn.Linear(self.one_side**2, 500)
        self.fc21 = nn.Linear(500, z_dim)  # fc21 for mean of Z
        self.fc22 = nn.Linear(500, z_dim)  # fc22 for log variance of Z
        self.fc3 = nn.Linear(z_dim, 500)
        self.fc4 = nn.Linear(500, self.one_side**2)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        mu = self.fc21(h1)
        logvar = self.fc22(h1)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.rand_like(std)
        return mu + eps*std

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))

    def forward(self, x):
        # x: [batch size, 1, 28,28] -> x: [batch size, 784]
        x = x.view(-1, self.one_side**2)
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar
