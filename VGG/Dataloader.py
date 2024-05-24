import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets

def MakeDataloader(batch_size=128, num_workers=0):
    # Load the CIFAR-10 dataset

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_data =  datasets.CIFAR10(root='./data', train=True, transform=transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, 4),
            transforms.ToTensor(),
            normalize,
        ]), download=True)

    test_data = datasets.CIFAR10(root='./data', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ]))

    train_loader = torch.utils.data.DataLoader(
        train_data,
        batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True)

    test_loader = torch.utils.data.DataLoader(
        test_data,
        batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True)

    return train_loader, test_loader, len(train_data), len(test_data)
