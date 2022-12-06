# Common Imports
import getopt
import os
import sys
from time import time

# Parse and get parameters
try:
    opts, args = getopt.getopt(sys.argv[1:], "f:b:e:")
except getopt.GetoptError:
    print("Invalid options. Support -f for storage filesystem dir, -b for batch size,  -e for epochs.")
    sys.exit(1)

param_batch_size = None
param_epochs = None
param_fsdir = None
for opt, arg in opts:
    if opt in ['-f']:
        param_fsdir = arg
    elif opt in ['-b']:
        param_batch_size = arg
    elif opt in ['-e']:
        param_epochs = arg

from cloudtik.runtime.spark.api import ThisSparkCluster
from cloudtik.runtime.ml.api import ThisMLCluster

cluster = ThisSparkCluster()

# Scale the cluster as need
cluster.scale(workers=1)

# Wait for all cluster workers to be ready
cluster.wait_for_ready(min_workers=1)

# Total worker cores
cluster_info = cluster.get_info()
total_workers = cluster_info.get("total-workers")
if not total_workers:
    total_workers = 1

default_storage = cluster.get_default_storage()
if not param_fsdir:
    param_fsdir = default_storage.get("default.storage.uri") if default_storage else None
    if not param_fsdir:
        print("Must specify storage filesystem dir using -f.")
        sys.exit(1)

ml_cluster = ThisMLCluster()
mlflow_url = ml_cluster.get_services()["mlflow"]["url"]


# Initialize SparkSession
from pyspark import SparkConf
from pyspark.sql import SparkSession

spark_conf = SparkConf().setAppName('spark-horovod-pytorch').set('spark.sql.shuffle.partitions', '16')
spark = SparkSession.builder.config(conf=spark_conf).getOrCreate()
conf = spark.conf


# Checkpoint utilities
CHECKPOINT_HOME = "/tmp/ml/checkpoints"


def get_checkpoint_file(log_dir, file_id):
    return os.path.join(log_dir, 'checkpoint-{file_id}.pth.tar'.format(file_id=file_id))


def save_checkpoint(log_dir, model, optimizer, file_id):
    filepath = get_checkpoint_file(log_dir, file_id)
    state = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
    }
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

num_proc = total_workers
print("Train processes: {}".format(num_proc))

batch_size = int(param_batch_size) if param_batch_size else 128
print("Train batch size: {}".format(batch_size))

epochs = int(param_epochs) if param_epochs else 1
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


# Horovod train function passed to horovod.spark.run
import torch.optim as optim
from torchvision import datasets, transforms

import horovod.torch as hvd


def train_horovod(learning_rate):
    # Initialize Horovod
    hvd.init()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda':
        # Pin GPU to local rank
        torch.cuda.set_device(hvd.local_rank())

    train_dataset = datasets.MNIST(
        # Use different root directory for each worker to avoid conflicts
        root='data-%d' % hvd.rank(),
        train=True,
        download=True,
        transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    )

    from torch.utils.data.distributed import DistributedSampler

    # Configure the sampler so that each worker gets a distinct sample of the input dataset
    train_sampler = DistributedSampler(train_dataset, num_replicas=hvd.size(), rank=hvd.rank())
    # Use train_sampler to load a different sample of data on each worker
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler)

    model = Net().to(device)

    # The effective batch size in synchronous distributed training is scaled by the number of workers
    # Increase learning_rate to compensate for the increased batch size
    optimizer = optim.Adadelta(model.parameters(), lr=learning_rate)

    # Wrap the local optimizer with hvd.DistributedOptimizer so that Horovod handles the distributed optimization
    optimizer = hvd.DistributedOptimizer(optimizer, named_parameters=model.named_parameters())

    # Broadcast initial parameters so all workers start with the same parameters
    hvd.broadcast_parameters(model.state_dict(), root_rank=0)

    local_checkpoint_dir = create_log_dir('pytorch-mnist-local')
    print("Log directory:", local_checkpoint_dir)

    for epoch in range(1, epochs + 1):
        train_one_epoch(model, device, train_loader, optimizer, epoch)
        # Save checkpoints only on worker 0 to prevent conflicts between workers
        if hvd.rank() == 0:
            save_checkpoint(local_checkpoint_dir, model, optimizer, epoch)

    if hvd.rank() == 0:
        # Return the model bytes of the last checkpoint
        checkpoint_file = get_checkpoint_file(local_checkpoint_dir, epochs)
        with open(checkpoint_file, 'rb') as f:
            return f.read()


# Local test functions
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


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
import horovod.spark

checkpoint_dir = create_log_dir('pytorch-mnist')
print("Log directory:", checkpoint_dir)


def hyper_objective(learning_rate):
    with mlflow.start_run():
        model_bytes = horovod.spark.run(
            train_horovod, args=(learning_rate,), num_proc=num_proc,
            stdout=sys.stdout, stderr=sys.stderr, verbose=2,
            prefix_output_with_timestamp=True)[0]

        # Write checkpoint
        checkpoint_file = get_checkpoint_file(checkpoint_dir, learning_rate)
        with open(checkpoint_file, 'wb') as f:
            f.write(model_bytes)
        print('Written checkpoint to {}'.format(checkpoint_file))

        loaded_model = load_model_of_checkpoint(checkpoint_dir, learning_rate)
        test_loss = test_model(loaded_model)

        mlflow.log_metric("learning_rate", learning_rate)
        mlflow.log_metric("loss", test_loss)

    return {'loss': test_loss, 'status': STATUS_OK}


# Do a super parameter tuning with hyperopt
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials

search_space = hp.uniform('learning_rate', 0, 1)
mlflow.set_tracking_uri(mlflow_url)
mlflow.set_experiment("MNIST: Distributed Hyperopt + Horovod + PyTorch")
argmin = fmin(
    fn=hyper_objective,
    space=search_space,
    algo=tpe.suggest,
    max_evals=16)
print("Best parameter found: ", argmin)


# Train final model with the best parameters and save the model
best_model = load_model_of_checkpoint(checkpoint_dir, argmin.get('learning_rate'))
model_name = "pytorch-mnist-model-distributed"
mlflow.pytorch.log_model(
    best_model, model_name, registered_model_name=model_name)


# Load the model from MLflow and run an evaluation
model_uri = "models:/model_name/1".format(model_name)
loaded_model = mlflow.pytorch.load_model(model_uri)
test_model(loaded_model)
