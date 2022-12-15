# Common Imports
import argparse
import os
import sys
from time import time

# Settings
parser = argparse.ArgumentParser(description='Single Node PyTorch MNIST Example')
parser.add_argument('--batch-size', type=int, default=128,
                    help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=1,
                    help='number of epochs to train (default: 1)')
parser.add_argument('--trials', type=int, default=2,
                    help='number of trails to parameter tuning (default: 2)')
parser.add_argument('--fsdir', default=None,
                    help='the file system dir (default: None)')
args = parser.parse_args()

param_fsdir = args.fsdir


# CloudTik cluster preparation or information
from cloudtik.runtime.spark.api import ThisSparkCluster
from cloudtik.runtime.ml.api import ThisMLCluster

cluster = ThisSparkCluster()

default_storage = cluster.get_default_storage()
if not param_fsdir:
    param_fsdir = default_storage.get("default.storage.uri") if default_storage else None
    if not param_fsdir:
        print("Must specify storage filesystem dir using -f.")
        sys.exit(1)

ml_cluster = ThisMLCluster()
mlflow_url = ml_cluster.get_services()["mlflow"]["url"]


# Checkpoint utilities
CHECKPOINT_HOME = "/tmp/ml/checkpoints"


def get_checkpoint_file(log_dir, file_id):
    return os.path.join(log_dir, 'checkpoint-{file_id}.pth.tar'.format(file_id=file_id))


def save_checkpoint(log_dir, model, optimizer, file_id):
    filepath = get_checkpoint_file(log_dir, file_id)
    print('Written checkpoint to {}'.format(filepath))
    state = {
        'model': model.state_dict(),
    }
    if optimizer is not None:
        state['optimizer'] = optimizer.state_dict()
    torch.save(state, filepath)


def load_checkpoint(log_dir, file_id):
    filepath = get_checkpoint_file(log_dir, file_id)
    return torch.load(filepath)


def create_log_dir(experiment_name):
    log_dir = os.path.join(CHECKPOINT_HOME, str(time()), experiment_name)
    os.makedirs(log_dir)
    return log_dir


# Define training function and pytorch model
import torch
import torch.nn as nn
import torch.nn.functional as F

batch_size = args.batch_size
print("Train batch size: {}".format(batch_size))

epochs = args.epochs
print("Train epochs: {}".format(epochs))


# Define the PyTorch model
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output


def train_one_epoch(model, device, data_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(data_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(data_loader) * len(data),
                100. * batch_idx / len(data_loader), loss.item()))


# Single node train function
import torch.optim as optim
from torchvision import datasets, transforms

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train(learning_rate):
    train_dataset = datasets.MNIST(
        'data',
        train=True,
        download=True,
        transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]))
    train_data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    model = Net().to(device)

    optimizer = optim.Adadelta(model.parameters(), lr=learning_rate)

    local_checkpoint_dir = create_log_dir('pytorch-mnist-local')
    print("Log directory:", local_checkpoint_dir)

    for epoch in range(1, epochs + 1):
        train_one_epoch(model, device, train_data_loader, optimizer, epoch)
        save_checkpoint(local_checkpoint_dir, model, optimizer, epoch)
    return model


def test_model(model):
    model.eval()

    test_dataset = datasets.MNIST(
        'data',
        train=False,
        download=True,
        transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]))
    test_data_loader = torch.utils.data.DataLoader(test_dataset)

    test_loss = 0
    for data, target in test_data_loader:
        data, target = data.to(device), target.to(device)
        output = model(data)
        test_loss += F.nll_loss(output, target)

    test_loss /= len(test_data_loader.dataset)
    print("Average test loss: {}".format(test_loss.item()))
    return test_loss


def load_model_of_checkpoint(log_dir, file_id):
    model = Net().to(device)
    checkpoint = load_checkpoint(log_dir, file_id)
    model.load_state_dict(checkpoint['model'])
    return model


def test_checkpoint(log_dir, file_id):
    model = load_model_of_checkpoint(log_dir, file_id)
    return test_model(model)


#  Hyperopt training function
import mlflow

checkpoint_dir = create_log_dir('pytorch-mnist')
print("Log directory:", checkpoint_dir)


def hyper_objective(learning_rate):
    with mlflow.start_run():
        model = train(learning_rate)

        # Write checkpoint
        save_checkpoint(checkpoint_dir, model, None, learning_rate)

        model = load_model_of_checkpoint(checkpoint_dir, learning_rate)
        test_loss = test_model(model)

        mlflow.log_metric("learning_rate", learning_rate)
        mlflow.log_metric("loss", test_loss)
    return {'loss': test_loss, 'status': STATUS_OK}


# Do a super parameter tuning with hyperopt
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials

trials = args.trials
print("Hyper parameter tuning trials: {}".format(trials))

search_space = hp.uniform('learning_rate', 0, 1)
mlflow.set_tracking_uri(mlflow_url)
mlflow.set_experiment("MNIST: Single Node Hyperopt + PyTorch")
argmin = fmin(
    fn=hyper_objective,
    space=search_space,
    algo=tpe.suggest,
    max_evals=trials)
print("Best parameter found: ", argmin)


# Train final model with the best parameters and save the model
best_model = load_model_of_checkpoint(checkpoint_dir, argmin.get('learning_rate'))
model_name = "pytorch-mnist-model-single-node"
mlflow.pytorch.log_model(
    best_model, model_name, registered_model_name=model_name)


# Load the model from MLflow and run an evaluation
model_uri = "models:/{}/latest".format(model_name)
print('Inference with model: {}'.format(model_uri))
saved_model = mlflow.pytorch.load_model(model_uri)
test_model(saved_model)
