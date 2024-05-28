import argparse
import random
import numpy as np

import torch
import torch.optim as optim

from Dataloader import MakeDataloader
from Trainer import train, validate

parser = argparse.ArgumentParser(description='PyTorch VGGNet Training')
parser.add_argument('-v', '--version', default="grid",
                    help='version of the test, grid or train (default: grid)')
parser.add_argument('-s', '--model-seed', default=7, type=int,
                    help='random seed for training (default: 7)')
parser.add_argument('-b', '--batch-size', default=128, type=int,
                    help='mini-batch size (default: 128)')
parser.add_argument('-e', '--epoch', default=100, type=int,
                    help='number of epochs (default: 100)')
parser.add_argument('-lr', '--learning_rate', default=0.01, type=float,
                    help='learning rate (default: 0.01)')
parser.add_argument('-wd', '--weight-decay', default=0.001, type=float,
                    help='weight decay (default: 0.001)')
parser.add_argument('-optim', '--optimizer', default='Adam',
                    help='Adam or AdamW (default: Adam)')

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

    #Load file
    fname = f'./Stats/VGG_{args.version}_{args.optimizer}_{args.learning_rate}_{args.weight_decay}_{args.epoch}_{args.model_seed}.txt'
    f = open(fname, 'w')

    # Call CIFAR dataset
    train_loader, test_loader, train_data_len, test_data_len = MakeDataloader(args.batch_size)

    # VGG net
    model = torch.hub.load('pytorch/vision:v0.10.0', 'vgg11_bn', pretrained=False)
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
        train_loss = 0.0
        train_accu = 0.0

        # Iterate over the train_loader
        for idx, sample in enumerate(train_loader):
            curr_loss, num_correct = train(model, optimizer, sample, device)
            train_loss += curr_loss
            train_accu += num_correct

        train_loss = train_loss / len(train_loader)
        train_accu = train_accu / train_data_len

        ### Validation phase
        # Initialize Loss and Accuracy
        val_loss = 0.0
        val_accu = 0.0

        # Iterate over the val_loader
        for idx, sample in enumerate(test_loader):
            curr_loss, num_correct = validate(model, sample, device)
            val_loss += curr_loss
            val_accu += num_correct

        val_loss = val_loss / len(test_loader)
        val_accu = val_accu / test_data_len

        torch.save(model.state_dict(),
                   f'./Weight/VGG_{args.version}_{args.optimizer}_{args.learning_rate}_{args.weight_decay}_{args.epoch}_{args.model_seed}.pth')

        f.write('[epoch {}] train_loss : {:.4f}, train accu : {:.4f} , val_loss : {:.4f}, val accu : {:.4f}\n' \
              .format(epoch + 1, train_loss, train_accu, val_loss, val_accu))
    f.close()

if __name__ == '__main__':
    main()
