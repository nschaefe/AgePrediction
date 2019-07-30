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
    def __init__(self, hiddens=[128], indim=128):
        super(FCN, self).__init__()
        layers = []
        layers.append(nn.Linear(indim, hiddens[0]))
        # layers.append(nn.BatchNorm1d(hiddens[0]))
        layers.append(nn.ReLU())
        for i in range(1, len(hiddens)):
            layers.append(nn.Linear(hiddens[i - 1], hiddens[i]))
            # layers.append(nn.BatchNorm1d(hiddens[i]))
            layers.append(nn.ReLU())
        # Output layer should be 1 as this is regression
        layers.append(nn.Linear(hiddens[-1], 1))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

# Model: Simple fully connected neural network
class FCNDropout(nn.Module):
    def __init__(self, hiddens=[128], indim=128):
        super(FCNDropout, self).__init__()
        layers = []
        layers.append(nn.Linear(indim, hiddens[0]))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(p=0.3))
        for i in range(1, len(hiddens)):
            layers.append(nn.Linear(hiddens[i - 1], hiddens[i]))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(p=0.5))
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
    net.train()
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
    net.eval()
    total_loss = 0.0
    age_diff = 0.0
    for i_batch, batch in enumerate(dataloader):
        with torch.no_grad():
            global_iteration += 1  # Needed to store a scalar in tensorboard
            xbatch = batch[0]
            ybatch = batch[1]
            ypred = net(xbatch)
            loss = criterion(ypred, ybatch.view((ybatch.shape[0], 1)))
            age_diff += torch.mean(torch.abs(ypred - ybatch.view((ybatch.shape[0], 1))))
            writer.add_scalar('validation_loss_batch', loss.item(), global_iteration)
            total_loss += loss.item()
    avg_agediff = age_diff / (i_batch + 1)
    avg_loss = total_loss/ ( i_batch + 1)
    writer.add_scalar('age_difference', avg_agediff, epoch)
    print('Epoch[{}] average validation loss = {}, average age difference = {}'.format(
        epoch,
        avg_loss,
        avg_agediff))
    return {'avg_age_diff': avg_agediff,
            'avg_loss': avg_loss,
            'predictions': None}


def test(epoch, net, dataloader, criterion, batch_size, writer, global_iteration):
    print('Epoch[{}] Validating'.format(epoch))
    net.eval()
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
            age_diff += torch.mean(torch.abs(ypred - ybatch.view((ybatch.shape[0], 1))))
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
    Helper function that writes age predictions on a test set to a file
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
        # This is the best model so far


def save_run(save_dict, save_dir):
    """
    Saves all the stats in the save_dict of a run to a file.
    :param save_dict: dictionary containing
    :param save_dir:
    :return:
    """
    with open(os.path.join(save_dir, 'run_description.txt'), 'w') as f:

        keys_to_print = ['net', 'total_epochs', 'lr', 'weight_decay', 'batch_size', 'best_validation_loss']
        for k in keys_to_print:
            print("_"*65, file=f)
            print(k, file=f)
            print(save_dict[k], file=f)
        print("_" * 65, file=f)
        print('test_loss', file=f)
        print(save_dict['test_out']['avg_loss'], file=f)
        print("_" * 65, file=f)
        print('avg_age_diff', file=f)
        print(save_dict['test_out']['avg_age_diff'], file=f)

    os.mkdir(os.path.join(save_dir, 'test_output'))  # This directory stores output of inference on the test set
    write_test_output(save_dict['total_epochs'], save_dict['test_out'], save_dir)
    save_model(save_dict['net'], save_dir, final=True)


def normalize_features(df, features):
    """
    Normalize the features of the dataframe. Only the features in the features list are normalized
    :param df: input dataframe
    :param features: list of features to normalize
    :return: normalized dataframe
    """
    for feature in features:
        df[feature] = (df[feature] - df[feature].min()) / (df[feature].max() - df[feature].min())
    return df


def main(hiddens, run_dir):
    """
    Does one complete run with the supplied hyperparameters (train, validate and test)
    :param hiddens: Number of hidden layers
    :return:
    """
    #  Some parameters
    batch_size = 2048
    total_epochs = 750
    valid_epoch, save_epoch = 1, 5
    run_id = 'run-' + datetime.now().strftime('%d_%m_%y_%H%M%S')
    save_dict = {}  # this dict stores the parameters we want to save
    save_dict['batch_size'] = batch_size
    save_dict['total_epochs'] = total_epochs
    save_dir = os.path.join('./runs/', run_dir, run_id)
    os.mkdir(save_dir)
    # For stats
    writer = SummaryWriter(log_dir=save_dir)
    train_iteration , val_iteration = 0, 0

    # Create dataset
    df = pd.read_csv('dataset/age_pred_data_withage.csv')
    df_train = df[df.is_train == 1]
    df_test = df[df.is_train == 0]
    # features = ['public', 'completion_percentage', 'gender',
    #             'last_login', 'registration', 'height', 'weight', 'comp_edu', 'smoking',
    #             'martial']
    # df = normalize_features(df, features)
    df_train = df_train.drop(['Unnamed: 0', 'user_id', 'X', 'X.Intercept.', 'is_train'], axis=1)
    df_test = df_test.drop(['Unnamed: 0', 'user_id', 'X', 'X.Intercept.', 'is_train'], axis=1)
    x_train = torch.tensor(df_train.values[:, :-2].astype(np.float32)).cuda()
    indims = x_train.shape[1]
    print(indims)
    y_train = torch.tensor(df_train.values[:, -1].astype(np.float32)).cuda()
    tensordata = torch.utils.data.TensorDataset(x_train, y_train)

    x_test = torch.tensor(df_test.values[:, :-2].astype(np.float32)).cuda()
    y_test = torch.tensor(df_test.values[:, -1].astype(np.float32)).cuda()
    tensordata_test = torch.utils.data.TensorDataset(x_test, y_test)

    # Split the dataset
    train_size = int(0.9 * len(x_train))
    val_size = len(x_train) -  train_size
    trainset, valset = random_split(tensordata, [train_size, val_size])
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
    valloader = DataLoader(valset, batch_size=batch_size)
    testloader = DataLoader(tensordata_test, batch_size=batch_size)
    net = FCN(hiddens, indim=indims)
    net.cuda()
    criterion = nn.MSELoss()
    # optimizer = torch.optim.SGD(net.parameters(), lr=1e-4, momentum=0.9)
    # Write parameters to a file
    save_dict['lr'] = 1e-4
    save_dict['weight_decay'] = 1e-5
    optimizer = torch.optim.Adam(net.parameters(), lr=save_dict['lr'],  weight_decay=save_dict['weight_decay'])
    best_valloss = 9e10  # Intitialize this to  be very large
    # Main training loop
    for i in range(total_epochs):
        train_iteration = train(i, net, trainloader, criterion, optimizer, batch_size, writer, train_iteration)

        # Check if this is the epoch to validate
        if i % valid_epoch == 0:
            val_out = validate(i, net, valloader, criterion, batch_size, writer, train_iteration)
            error = val_out['avg_loss']
            if error < best_valloss:
                best_valloss = error
                save_model(net, save_dir, final=False)  # save best model similar to early stopping
                # This is a hack. we do inference for the model that gives the best validation error
                test_out = test(i, net, testloader, criterion, batch_size, writer, train_iteration)
            # Run on test set and save predictions
    # Save the last model
    save_dict['best_validation_loss'] = best_valloss
    save_dict['test_out'] = test_out  # this is a dict containing test loss, test me, and predictions
    save_dict['net'] = net
    save_run(save_dict, save_dir)


# Hyper parameter optimization.
# run main passing in hyper parameter that we optimize for
def optimize_layers(indim=128):
    run_dir = 'wtdecay-featsemb2'
    pth = os.path.join('runs', run_dir)
    if not os.path.exists(pth):
        os.mkdir(pth)

    num_layers = [1, 2, 3, 4, 8, 12]

    for layer_count in num_layers:
        hiddens = [2048]  # first layer always has 128
        # Subsequent layers have neurons decreasing by a factor of 2
        neuron_num = indim
        for i in range(1, layer_count):
            neuron_num /= 2
            if neuron_num < 2:
                break
            hiddens += [int(neuron_num)]
        # Build neural network with this config
        print(hiddens)
        main(hiddens, run_dir=run_dir)


if __name__=='__main__':
    optimize_layers()