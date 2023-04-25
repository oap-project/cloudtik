# Common Imports
import argparse
import os
import sys
from time import time

# Settings
parser = argparse.ArgumentParser(description='Horovod on Spark PyTorch MNIST Example')
parser.add_argument('--num-proc', type=int,
                    help='number of worker processes for training')
parser.add_argument('--batch-size', type=int, default=128,
                    help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=2,
                    help='number of epochs to train (default: 2)')
parser.add_argument('--trials', type=int, default=2,
                    help='number of trails to parameter tuning (default: 2)')
parser.add_argument('--fsdir', default=None,
                    help='the file system dir (default: None)')
parser.add_argument('--gloo', action='store_true', dest='use_gloo',
                    help='Run Horovod using the Gloo controller. This will '
                         'be the default if Horovod was not built with MPI support.')
parser.add_argument('--mpi', action='store_true', dest='use_mpi',
                    help='Run Horovod using the MPI controller. This will '
                         'be the default if Horovod was built with MPI support.')

if __name__ == '__main__':
    args = parser.parse_args()
    fsdir = args.fsdir

    # CloudTik cluster preparation or information
    from cloudtik.runtime.spark.api import ThisSparkCluster
    from cloudtik.runtime.ml.api import ThisMLCluster

    cluster = ThisSparkCluster()

    # Scale the cluster as need
    # cluster.scale(workers=1)

    # Wait for all cluster workers to be ready
    cluster.wait_for_ready(min_workers=1)

    # Total worker cores
    cluster_info = cluster.get_info()

    if not args.num_proc:
        total_worker_cpus = cluster_info.get("total-worker-cpus-ready")
        if total_worker_cpus:
            args.num_proc = int(total_worker_cpus / 4)
        if not args.num_proc:
            args.num_proc = 1

    default_storage = cluster.get_default_storage()
    if not fsdir:
        fsdir = default_storage.get("default.storage.uri") if default_storage else None
        if not fsdir:
            print("Must specify storage filesystem dir using --fsdir.")
            sys.exit(1)

    ml_cluster = ThisMLCluster()
    mlflow_url = ml_cluster.get_services()["mlflow"]["url"]

    # Initialize SparkSession
    from pyspark import SparkConf
    from pyspark.sql import SparkSession

    conf = SparkConf().setAppName('spark-horovod-pytorch').set('spark.sql.shuffle.partitions', '16')
    spark = SparkSession.builder.config(conf=conf).getOrCreate()

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
        os.makedirs(log_dir, exist_ok=True)
        return log_dir


    # Define training function and pytorch model
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    num_proc = args.num_proc
    print("Train processes: {}".format(num_proc))

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

        local_checkpoint_dir = create_log_dir(
            "pytorch-mnist-{}".format(hvd.rank()))
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
                use_gloo=args.use_gloo, use_mpi=args.use_mpi,
                prefix_output_with_timestamp=True)[0]

            # Write checkpoint
            checkpoint_file = get_checkpoint_file(checkpoint_dir, learning_rate)
            with open(checkpoint_file, 'wb') as f:
                f.write(model_bytes)
            print('Written checkpoint to {}'.format(checkpoint_file))

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
    mlflow.set_experiment("MNIST: PyTorch + Spark + Horovod Run")
    argmin = fmin(
        fn=hyper_objective,
        space=search_space,
        algo=tpe.suggest,
        max_evals=trials)
    print("Best parameter found: ", argmin)

    # Train final model with the best parameters and save the model
    best_model = load_model_of_checkpoint(checkpoint_dir, argmin.get('learning_rate'))
    model_name = "mnist-pytorch-spark-horovod-run"
    mlflow.pytorch.log_model(
        best_model, model_name, registered_model_name=model_name)

    # Load the model from MLflow and run an evaluation
    model_uri = "models:/{}/latest".format(model_name)
    print('Inference with model: {}'.format(model_uri))
    saved_model = mlflow.pytorch.load_model(model_uri)
    test_model(saved_model)
