import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import os


# Model: Simple fully connected neural network
class FCN(nn.Module):
    def __init__(self, hiddens=[256, 16], indim=128):
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
    return avg_agediff  # Used as a measure of accuracy


def test(epoch, net, dataloader, criterion, batch_size, writer, global_iteration):
    print('Epoch[{}] Validating'.format(epoch))
    total_loss = 0.0
    age_diff = 0.0
    recorded_predictions = []
    for i_batch, batch in enumerate(dataloader):
        with torch.no_grad():
            global_iteration += 1  # Needed to store a scalar in tensorboard
            xbatch = batch[0]
            ybatch = batch[1]
            ypred = net(xbatch)
            recorded_predictions.append(torch.stack((ybatch, ypred.squeeze(1)), 1)) # save all the predictions ang ground truth
            loss = criterion(ypred, ybatch.view((ybatch.shape[0], 1)))
            age_diff += torch.mean(torch.abs(ypred - ybatch))
            writer.add_scalar('validation_loss_batch', loss.item(), global_iteration)
            total_loss += loss.item()
    avg_agediff = age_diff / (i_batch + 1)
    avg_loss = total_loss / (i_batch + 1)
    # age difference on the test set
    writer.add_scalar('test_age_difference', avg_agediff, epoch)
    print(avg_agediff)
    return {'avg_age_diff': avg_agediff,
            'avg_loss': avg_loss,
            'predictions': recorded_predictions}


def write_test_output(epoch, test_out, save_dir):
    """
    :param epoch: global epoch number
    :param test_out: dictionary of the form
        {
            'avg_age_diff': avg_agediff,
            'avg_loss': avg_loss,
            'predictions': recorded_predictions
        }
    :param save_dir: directory in which outputs will be dumped
    :return:
    """
    predictions = test_out['predictions']
    if len(predictions) < 1:
        print('Nothing in the test predictions')
    np_predictions = predictions[0].data.cpu().numpy()
    for i in range(1, len(predictions)):
        np_predictions = np.concatenate( (np_predictions, predictions[i].data.cpu().numpy()), axis=0)

    print(np_predictions[:10])  # output first 10 predictions
    np.savetxt(os.path.join(save_dir,'test_output', 'epoch{}test.output'.format(epoch)), np_predictions, fmt='%1.4f')


def save_model(net, save_dir,  final=False):
    if final:
        # This is the last model
        torch.save(net.state_dict(), os.path.join(save_dir, 'final_model.pt'))
    else:
        torch.save(net.state_dict(), os.path.join(save_dir, 'best_model.pt'))
        # This is the best model sofar


def main():
    #  Some parameters
    batch_size = 2048
    total_epochs = 200
    valid_epoch, save_epoch = 5, 5
    save_dir = os.path.join('./runs/', 'run-'+datetime.now().strftime('%d_%m_%y_%H%M%S'))
    os.mkdir(save_dir)
    os.mkdir(os.path.join(save_dir, 'test_output'))  # This directory stores output of inference on the test set
    # For stats
    writer = SummaryWriter(log_dir=save_dir)
    train_iteration , val_iteration = 0, 0

    # Create dataset
    df = pd.read_csv('./dataset/full_filtered.csv')
    x = torch.tensor(df.values[:, 2:130].astype(np.float32)).cuda()
    y = torch.tensor(df.values[:, -1].astype(np.float32)).cuda()
    tensordata = torch.utils.data.TensorDataset(x, y)

    # Split the dataset
    train_size = int(0.8 * len(x))
    val_size = int(0.1* len(x))
    test_size = len(x) -  train_size - val_size
    trainset, valset, testset = random_split(tensordata, [train_size, val_size, test_size])
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
    valloader = DataLoader(valset, batch_size=batch_size)
    testloader = DataLoader(testset, batch_size=batch_size)
    net = FCN()
    net.cuda()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-4)

    best_error = 9e10
    # Main training loop
    for i in range(total_epochs):
        train_iteration = train(i, net, trainloader, criterion, optimizer, batch_size, writer, train_iteration)

        # Check if this is the epoch to validate
        if i % valid_epoch == 0:
            error = validate(i, net, valloader, criterion, batch_size, writer, train_iteration)
            if error < best_error:
                best_error = error
                save_model(net, save_dir, final=False)  # save best model similar to early stopping

            # Run on test set and save predictions
            test_out = test(i, net, testloader, criterion, batch_size, writer, train_iteration)
            write_test_output(i, test_out, save_dir)
            i -= 1
    # Save the last model
    save_model(net, save_dir, final=True)



if __name__=='__main__':
    main()
