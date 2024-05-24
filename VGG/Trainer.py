import torch
import torch.nn as nn

def train(model, optimizer, sample, device):
    model.train()

    # define loss function
    criterion = nn.CrossEntropyLoss()

    # define input and output
    input = sample[0].float().to(device)
    label = sample[1].long().to(device)

    # get prediction
    pred = model(input)

    # count number of correct answers from training dataset
    num_correct = sum(torch.argmax(pred, dim=1) == label)

    # calculate the loss
    pred_loss = criterion(pred, label)

    # run 1-step training iteration
    optimizer.zero_grad()
    pred_loss.backward()
    optimizer.step()

    return pred_loss.item(), num_correct.item()

def validate(model, sample, device):
    model.eval()
    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        input, label = sample[0].to(device), sample[1].to(device)
        pred = model(input)

        num_correct = torch.sum(torch.argmax(pred, dim=-1) == label)
        pred_loss = criterion(pred, label)

    return pred_loss.item(), num_correct.item()