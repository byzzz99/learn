import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from DataLoader import *
from mymodel import *
import numpy as np


def train(model, device):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    num_epoches =20
    cost =[]

    for epoch in range(num_epoches):
        running_loss =0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            # print(inputs, inputs.shape)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i%2000 == 1999:
                print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 2000))
                cost.append(running_loss / 2000)
                running_loss = 0.0

    # plt.plot(cost)
    # plt.ylabel('loss')
    # plt.show()

def test(model, device):
    total =0
    correct = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, pred = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (pred == labels).sum().item()
    print('Accuracy of the network on the 10000 test images: %.3f %%' % (100 * correct / total))

def save_model(model, ckpt_path):
    torch.save(model.state_dict(), ckpt_path)


if __name__ == '__main__':
    device = "cuda"
    model = LeNet_5().to(device)
    train(model, device)
    test(model, device)
    save_model(model, 'Models/leNet_5.pth')

