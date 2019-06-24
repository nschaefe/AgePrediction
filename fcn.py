import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter


# Model: Simple fully connected neural network
class FCN(nn.Module):
    def __init__(self, hiddens=[64, 8], indim=128):
        super(FCN, self).__init__()
        layers = []
        layers.append(nn.Linear(indim, hiddens[0]))
        layers.append(nn.ReLU())
        for i in range(1, len(hiddens)):
            layers.append(nn.Linear(hiddens[i - 1], hiddens[i]))
            layers.append(nn.ReLU())
        # Output layer should be 1 as this is regression
        layers.append(nn.Linear(hiddens[-1], 1))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


# In[141]:
# Train function
def train(epoch, net, dataloader, criterion, optimizer, batch_size, writer, global_iteration):
    print('Epoch[{}]'.format(epoch))
    for i_batch, batch in enumerate(dataloader):
        global_iteration += 1 # Needed to store a scalar in tensorboard
        optimizer.zero_grad()
        xbatch = batch[0]
        ybatch = batch[1]
        ypred = net(xbatch)
        loss = criterion(ypred, ybatch.view((ybatch.shape[0], 1)))
        writer.add_scalar('train_loss_batch', loss.item(), global_iteration)
        if i_batch%50 == 0:
            print(' Training Batch[{}] Loss = {}'.format(i_batch, loss))
        loss.backward()
        optimizer.step()
    return global_iteration


def validate(epoch, net, dataloader, criterion, batch_size, writer, global_iteration):
    print('Epoch[{}] Validating'.format(epoch))
    total_loss = 0.0
    age_diff = 0.0
    for i_batch, batch in enumerate(dataloader):
        with torch.no_grad():
            global_iteration += 1  # Needed to store a scalar in tensorboard
            xbatch = batch[0]
            ybatch = batch[1]
            ypred = net(xbatch)
            loss = criterion(ypred, ybatch.view((ybatch.shape[0], 1)))
            age_diff += torch.mean(torch.abs(ypred - ybatch))
            writer.add_scalar('validation_loss_batch', loss.item(), global_iteration)
            total_loss += loss.item()
    avg_agediff = age_diff / (i_batch + 1)
    writer.add_scalar('age_difference', avg_agediff, epoch)
    print('Epoch[{}] average validation loss = {}, average age difference = {}'.format(
        epoch,
        total_loss/ ( i_batch + 1),
        avg_agediff))


def test(epoch, net, dataloader, criterion, batch_size, writer, global_iteration):
    print('Epoch[{}] Validating'.format(epoch))
    total_loss = 0.0
    age_diff = 0.0
    for i_batch, batch in enumerate(dataloader):
        with torch.no_grad():
            global_iteration += 1  # Needed to store a scalar in tensorboard
            xbatch = batch[0]
            ybatch = batch[1]
            ypred = net(xbatch)
            loss = criterion(ypred, ybatch.view((ybatch.shape[0], 1)))
            age_diff += torch.mean(torch.abs(ypred - ybatch))
            writer.add_scalar('validation_loss_batch', loss.item(), global_iteration)
            total_loss += loss.item()

    avg_agediff = age_diff / (i_batch + 1)
    # age difference on the test set
    writer.add_scalar('test_age_difference', avg_agediff, epoch)
    print(avg_agediff)


def main():
    df = pd.read_csv('./dataset/full_filtered.csv')
    # For stats
    writer = SummaryWriter()
    train_iteration , val_iteration = 0, 0
    # Create dataset
    x = torch.tensor(df.values[:, 2:130].astype(np.float32)).cuda()
    y = torch.tensor(df.values[:, -1].astype(np.float32)).cuda()
    tensordata = torch.utils.data.TensorDataset(x, y)
    # Split the dataset
    train_size = int(0.8 * len(x))
    val_size = int(01.* len(x))
    test_size = len(x) -  train_size - val_size
    trainset, valset, testset = random_split(tensordata, [train_size, val_size, test_size])
    #  Some parameters
    batch_size = 2048
    total_epochs = 200
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
    valloader = DataLoader(valset, batch_size=batch_size)
    net = FCN()
    net.cuda()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-4)
    # This is the main training loop
    for i in range(total_epochs):
        validate(i, net, valloader, criterion, batch_size, writer, train_iteration)
        train_iteration = train(i, net, trainloader, criterion, optimizer, batch_size, writer, train_iteration)

    # TODO: Create a file to dump predictions on the test set




if __name__=='__main__':
    main()
