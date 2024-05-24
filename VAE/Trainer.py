import torch
from torch.nn import functional as F

def loss_function(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')
    KLD = 0.5 * torch.sum(mu.pow(2) + logvar.exp() - logvar - 1)
    return BCE + KLD, BCE, KLD

def train(model, optimizer, sample, device):
    model.train()

    # define input and output
    input = sample[0].float().to(device)
    input = input.flatten(start_dim=1)

    # get prediction
    x_recon, z_mean, z_logvar = model(input)

    # calculate the loss
    total_loss, MSEloss, KLloss = loss_function(x_recon, input, z_mean, z_logvar)

    # run 1-step training iteration
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

    return total_loss.item(), MSEloss.item(), KLloss.item()

def validate(model, sample, device):
    model.eval()

    with torch.no_grad():
        input = sample[0].to(device)
        input = input.flatten(start_dim=1)
        x_recon, z_mean, z_logvar = model(input)

        total_loss, MSEloss, KLloss = loss_function(x_recon, input, z_mean, z_logvar)

    return x_recon, total_loss.item(), MSEloss.item(), KLloss.item()
