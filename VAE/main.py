import argparse
import random
import numpy as np

import torch
import torch.optim as optim
from torchvision.utils import save_image

from Dataloader import MakeDataloader
from Model import VAE
from Trainer import train, validate

parser = argparse.ArgumentParser(description='PyTorch VAE Training')
parser.add_argument('-v', '--version', default="grid",
                    help='version of the test, grid or train (default: grid)')
parser.add_argument('-s', '--model-seed', default=7, type=int,
                    help='random seed for training (default: 7)')
parser.add_argument('-b', '--batch-size', default=128, type=int,
                    help='mini-batch size (default: 128)')
parser.add_argument('-e', '--epoch', default=20, type=int,
                    help='number of epochs (default: 20)')
parser.add_argument('-lr', '--learning_rate', default=0.01, type=float,
                    help='learning rate (default: 0.01)')
parser.add_argument('-wd', '--weight-decay', default=0.001, type=float,
                    help='weight decay (default: 0.001)')
parser.add_argument('-optim', '--optimizer', default='Adam',
                    help='Adam or AdamW (default: Adam)')
parser.add_argument('-z', '--z-dim', default=4, type=int,
                    help='dimension of the latent variable (default: 4)')

args = parser.parse_args()
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def reset(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def main():
    reset(args.model_seed)

    # Load file
    fname = f'./Stats/VAE_{args.version}_{args.optimizer}_{args.learning_rate}_{args.weight_decay}_{args.epoch}_{args.model_seed}.txt'
    f = open(fname, 'w')

    # Call Fashion MNIST dataset
    train_loader, test_loader = MakeDataloader(args.batch_size)

    # Dense layer VAE
    model = VAE(z_dim=args.z_dim)
    model.to(device)

    # Optimizer
    if args.optimizer == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    elif args.optimizer == 'AdamW':
        optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    else:
        raise Exception("Adam or AdamW")

    for epoch in range(args.epoch):
        ###Train Phase

        # Initialize Loss and Accuracy
        train_total_loss = 0.0
        train_recon_loss = 0.0
        train_KL_loss = 0.0

        # Iterate over the train_loader
        for idx, sample in enumerate(train_loader):
            total_loss, recon_loss, KL_loss = train(model, optimizer, sample, device)
            train_total_loss += total_loss
            train_recon_loss += recon_loss
            train_KL_loss += KL_loss

        train_total_loss = train_total_loss / len(train_loader)
        train_recon_loss = train_recon_loss / len(train_loader)
        train_KL_loss = train_KL_loss / len(train_loader)

        ### Validation phase
        # Initialize Loss and Accuracy
        val_total_loss = 0.0
        val_recon_loss = 0.0
        val_KL_loss = 0.0

        # Iterate over the val_loader
        for idx, sample in enumerate(test_loader):
            x_recon, total_loss, recon_loss, KL_loss = validate(model, sample, device)
            val_total_loss += total_loss
            val_recon_loss += recon_loss
            val_KL_loss += KL_loss

            if epoch % 10 == 0 and idx == 0:
                num_samples = min(args.batch_size, 16)

                input = sample[0].to(device)
                compare_pics = torch.cat(
                    [input[:num_samples//2], x_recon.view(args.batch_size, 1, 28, 28)[:num_samples//2],
                     input[num_samples//2:num_samples], x_recon.view(args.batch_size, 1, 28, 28)[num_samples//2:num_samples]]).cpu()
                save_image(
                    compare_pics,
                    f'./ReconResults/VAE_{args.version}_{args.optimizer}_{args.learning_rate}_{args.weight_decay}_{args.epoch}_{args.model_seed}_{epoch+1}.png',
                    n_row=4)

        val_total_loss = val_total_loss / len(test_loader)
        val_recon_loss = val_recon_loss / len(test_loader)
        val_KL_loss = val_KL_loss / len(test_loader)

        if epoch % 5 == 0 or epoch == args.epoch-1:
            torch.save(model.state_dict(),
                       f'./Weight/VAE_{args.version}_{args.optimizer}_{args.learning_rate}_{args.weight_decay}_{args.epoch}_{args.model_seed}_{epoch+1}.pth')

        f.write('[epoch {}] train_total_loss : {:.4f}, train_recon_loss : {:.4f}, train_KL_loss : {:.4f}, val_total_loss : {:.4f}, val_recon_loss : {:.4f}, val_KL_loss : {:.4f}\n'
                .format(epoch+1, train_total_loss, train_recon_loss, train_KL_loss, val_total_loss, val_recon_loss, val_KL_loss))
    f.close()


if __name__ == '__main__':
    main()
