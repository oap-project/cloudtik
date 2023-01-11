import argparse
import os
from filelock import FileLock
import tempfile


# Training settings
parser = argparse.ArgumentParser(description='Horovod PyTorch Lightning MNIST Example')
parser.add_argument('--num-proc', type=int,
                    help='number of worker processes for training')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=2, metavar='N',
                    help='number of epochs to train (default: 2)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=42, metavar='S',
                    help='random seed (default: 42)')
parser.add_argument('--data-dir',
                    help='location of the training dataset in the local filesystem (will be downloaded if needed)')
parser.add_argument('--gloo', action='store_true', dest='use_gloo',
                    help='Run Horovod using the Gloo controller. This will '
                         'be the default if Horovod was not built with MPI support.')
parser.add_argument('--mpi', action='store_true', dest='use_mpi',
                    help='Run Horovod using the MPI controller. This will '
                         'be the default if Horovod was built with MPI support.')


import torch
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger


# Define the PyTorch model without any Horovod-specific parameters
class Net(LightningModule):
    def __init__(self, lr=0.01):
        super(Net, self).__init__()
        self.lr = lr
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

    def configure_optimizers(self):
        return optim.SGD(self.parameters(), lr=self.lr, momentum=0.5)

    def training_step(self, batch, batch_nb):
        x, y = batch[0], batch[1]
        y_hat = self(x)
        loss = F.nll_loss(y_hat, y.long())
        tensorboard_logs = {'train_loss': loss}
        return {'loss': loss, 'log': tensorboard_logs}

    def validation_step(self, batch, batch_nb):
        x, y = batch[0], batch[1]
        y_hat = self(x)
        return {'val_loss': F.nll_loss(y_hat, y.long())}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        tensorboard_logs = {'val_loss': avg_loss}
        return {'avg_val_loss': avg_loss, 'log': tensorboard_logs}


if __name__ == '__main__':
    args = parser.parse_args()

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
        total_worker_cpus = cluster_info.get("total-worker-cpus")
        if total_worker_cpus:
            args.num_proc = int(total_worker_cpus / 2)
        if not args.num_proc:
            args.num_proc = 1

    worker_ips = cluster.get_worker_node_ips()

    ml_cluster = ThisMLCluster()
    mlflow_url = ml_cluster.get_services()["mlflow"]["url"]

    # Checkpoint utilities
    CHECKPOINT_HOME = "/tmp/ml/checkpoints"


    def get_checkpoint_file(log_dir, file_id):
        return os.path.join(log_dir, 'checkpoint-{file_id}.torch'.format(file_id=file_id))


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


    import io

    def serialize_torch_model(model):
        serialized = io.BytesIO()
        output = {'model': model.state_dict()}
        torch.save(output, serialized)
        return serialized.getvalue()

    def deserialize_torch_model(model_bytes):
        """Deserialize model from byte array encoded in base 64."""
        serialized = io.BytesIO(model_bytes)
        checkpoint = torch.load(serialized, map_location=torch.device('cpu'))

        model = Net()
        model.load_state_dict(checkpoint['model'])
        return model


    import horovod.torch as hvd


    def train_horovod(learning_rate):
        torch.manual_seed(args.seed)
        hvd.init()
        args.cuda = not args.no_cuda and torch.cuda.is_available()
        if args.cuda:
            torch.cuda.set_device(hvd.local_rank())
            torch.cuda.manual_seed(args.seed)

        kwargs = {'num_workers': 2}
        # When supported, use 'forkserver' to spawn dataloader workers instead of 'fork' to prevent
        # issues with Infiniband implementations that are not fork-safe
        if (kwargs.get('num_workers', 0) > 0 and hasattr(mp, '_supports_context') and
                mp._supports_context and 'forkserver' in mp.get_all_start_methods()):
            kwargs['multiprocessing_context'] = 'forkserver'

        # get data
        data_dir = args.data_dir or './data'
        with FileLock(os.path.expanduser("~/.horovod_lock")):
            train_dataset = \
                datasets.MNIST(data_dir, train=True, download=True,
                               transform=transforms.Compose([
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.1307,), (0.3081,))
                               ]))

        # set training data loader
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_dataset, num_replicas=hvd.size(), rank=hvd.rank())
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=args.batch_size, sampler=train_sampler, **kwargs)

        test_dataset = \
            datasets.MNIST(data_dir, train=False, transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ]))

        # set validation data loader
        test_sampler = torch.utils.data.distributed.DistributedSampler(
            test_dataset, num_replicas=hvd.size(), rank=hvd.rank())
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.test_batch_size,
                                                  sampler=test_sampler, **kwargs)

        epochs = args.epochs
        with tempfile.TemporaryDirectory() as run_output_dir:
            ckpt_path = os.path.join(run_output_dir, "checkpoint")
            os.makedirs(ckpt_path, exist_ok=True)

            logs_path = os.path.join(run_output_dir, "logger")
            os.makedirs(logs_path, exist_ok=True)
            logger = TensorBoardLogger(logs_path)

            train_percent = 1.0
            val_percent = 1.0

            model = Net(learning_rate)
            setattr(model, 'train_dataloader', lambda: train_loader)
            setattr(model, 'val_dataloader', lambda: test_loader)

            from pytorch_lightning.callbacks import Callback

            class MyDummyCallback(Callback):
                def __init__(self):
                    self.epcoh_end_counter = 0
                    self.train_epcoh_end_counter = 0

                def on_init_start(self, trainer):
                    print('Starting to init trainer!')

                def on_init_end(self, trainer):
                    print('Trainer is initialized.')

                def on_epoch_end(self, trainer, model):
                    print('A epoch ended.')
                    self.epcoh_end_counter += 1

                def on_train_epoch_end(self, trainer, model, unused=None):
                    print('A train epoch ended.')
                    self.train_epcoh_end_counter += 1

                def on_train_end(self, trainer, model):
                    print('Training ends')
                    assert self.epcoh_end_counter == 2 * epochs
                    assert self.train_epcoh_end_counter == epochs

            callbacks = [MyDummyCallback(), ModelCheckpoint(dirpath=ckpt_path)]

            trainer = Trainer(accelerator='horovod',
                              gpus=(1 if args.cuda else 0),
                              callbacks=callbacks,
                              max_epochs=epochs,
                              limit_train_batches=train_percent,
                              limit_val_batches=val_percent,
                              logger=logger,
                              num_sanity_val_steps=0)

            trainer.fit(model)
            if args.cuda:
                model = model.cuda()

            def metric_average(val, name):
                tensor = torch.tensor(val)
                avg_tensor = hvd.allreduce(tensor, name=name)
                return avg_tensor.item()

            def validate():
                model.eval()
                test_loss = 0.
                test_accuracy = 0.
                for data, target in test_loader:
                    if args.cuda:
                        data, target = data.cuda(), target.cuda()
                    output = model(data)
                    # sum up batch loss
                    test_loss += F.nll_loss(output, target, size_average=False).item()
                    # get the index of the max log-probability
                    pred = output.data.max(1, keepdim=True)[1]
                    test_accuracy += pred.eq(target.data.view_as(pred)).cpu().float().sum()

                # Horovod: use test_sampler to determine the number of examples in
                # this worker's partition.
                test_loss /= len(test_sampler)
                test_accuracy /= len(test_sampler)

                # Horovod: average metric values across workers.
                test_loss = metric_average(test_loss, 'avg_loss')
                test_accuracy = metric_average(test_accuracy, 'avg_accuracy')

                # Horovod: print output only on first rank.
                if hvd.rank() == 0:
                    print('\nTest set: Average loss: {:.4f}, Accuracy: {:.2f}%\n'.format(
                        test_loss, 100. * test_accuracy))
            validate()

            if hvd.rank() == 0:
                return serialize_torch_model(model)


    def test_model(model):
        data_dir = args.data_dir or './data'
        test_dataset = \
            datasets.MNIST(data_dir, train=False, transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ]))

        # set validation data loader
        test_sampler = torch.utils.data.distributed.DistributedSampler(
            test_dataset, num_replicas=1, rank=0)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.test_batch_size,
                                                  sampler=test_sampler)

        cuda = not args.no_cuda and torch.cuda.is_available()

        model.eval()
        test_loss = 0.
        test_accuracy = 0.
        for data, target in test_loader:
            if cuda:
                data, target = data.cuda(), target.cuda()
            output = model(data)
            # sum up batch loss
            test_loss += F.nll_loss(output, target, size_average=False).item()
            # get the index of the max log-probability
            pred = output.data.max(1, keepdim=True)[1]
            test_accuracy += pred.eq(target.data.view_as(pred)).cpu().float().sum()

        # Horovod: use test_sampler to determine the number of examples in
        # this worker's partition.
        test_loss /= len(test_sampler)
        test_accuracy /= len(test_sampler)

        print('\nTest set: Average loss: {:.4f}, Accuracy: {:.2f}%\n'.format(
            test_loss, 100. * test_accuracy))


    import horovod

    num_proc = args.num_proc
    print("Train processes: {}".format(num_proc))

    # Generate the host list
    worker_num_proc = int(num_proc / len(worker_ips))
    if not worker_num_proc:
        worker_num_proc = 1
    host_slots = ["{}:{}".format(worker_ip, worker_num_proc) for worker_ip in worker_ips]
    hosts = ",".join(host_slots)
    print("Hosts to run:", hosts)

    learning_rate = args.lr
    print("Train learning rate: {}".format(learning_rate))

    model_bytes = horovod.run(
        train_horovod, args=(learning_rate,),
        num_proc=num_proc, hosts=hosts,
        use_gloo=args.use_gloo, use_mpi=args.use_mpi,
        verbose=2)[0]

    model = deserialize_torch_model(model_bytes)

    print("Final testing of the trained model")
    test_model(model)
