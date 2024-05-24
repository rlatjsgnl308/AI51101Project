import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets

def MakeDataloader(batch_size=128, num_workers=0):
    train_data = datasets.FashionMNIST('./data', train=True, download=True,
                                       transform=transforms.ToTensor())
    test_data = datasets.FashionMNIST('./data', train=False,
                                      transform=transforms.ToTensor())

    train_loader = torch.utils.data.DataLoader(train_data,
                                               batch_size=batch_size, shuffle=True,
                                               num_workers=num_workers, pin_memory=True)

    test_loader = torch.utils.data.DataLoader(test_data,
                                              batch_size=batch_size, shuffle=False,
                                              num_workers=num_workers, pin_memory=True)
    return train_loader, test_loader