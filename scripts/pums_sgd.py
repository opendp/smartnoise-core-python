import json
import os
import sys
from multiprocessing import Queue
from random import randint

import pandas as pd
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.multiprocessing import Process
from torch.utils.data import DataLoader, TensorDataset

from opendp.smartnoise.network.optimizer import PrivacyAccountant
from scripts.pums_downloader import get_pums_data_path, download_pums_data, datasets

# defaults to predicting ambulatory difficulty based on age, weight and cognitive difficulty
ACTIVE_PROBLEM = 'ambulatory'

problem = {
    'ambulatory': {
        'description': 'predict ambulatory difficulty based on age, weight and cognitive difficulty',
        'predictors': ['AGEP', 'PWGTP', 'DREM'],
        'target': 'DPHY'
    },
    'marital': {
        'description': 'predict marital status as a function of income and education',
        'predictors': ['PERNP', 'SCHL'],
        'target': 'MAR'
    },
    'medicare': {
        'description': 'predict medicare status based on mode of transporation to work (JWTR), '
                       'hours worked per week (WKHP), and number of weeks worked in past 12 months (WKW)',
        'predictors': ['JWTR', 'WKHP', 'WKW'],
        'target': 'HINS4'
    }
}[ACTIVE_PROBLEM]

debug = True


def printf(x, force=False):
    """
    overkill flushing
    :param x:
    :param force:
    :return:
    """
    if debug or force:
        print(x, flush=True)
        sys.stdout.flush()


def load_pums(dataset):
    download_pums_data(**dataset)
    data_path = get_pums_data_path(**dataset)

    data = pd.read_csv(data_path, usecols=problem['predictors'] + [problem['target']], engine='python')
    data.dropna(inplace=True)
    if ACTIVE_PROBLEM == 'marital':
        data['MAR'] = (data['MAR'] == 1) + 1
    return TensorDataset(
        torch.from_numpy(data[problem['predictors']].to_numpy()).type(torch.float32),
        torch.from_numpy(data[problem['target']].to_numpy()).type(torch.LongTensor) - 1)


class PumsModule(nn.Module):

    def __init__(self, input_size, output_size):
        """
        Example NN module
        :param input_size:
        :param output_size:
        """
        super().__init__()
        internal_size = 5
        self.linear1 = nn.Linear(input_size, internal_size)
        self.linear2 = nn.Linear(internal_size, output_size)

    def forward(self, x):
        x = self.linear1(x)
        x = F.relu(x)
        x = self.linear2(x)
        return x

    def loss(self, batch):
        inputs, targets = batch
        outputs = self(inputs)
        return F.cross_entropy(outputs, targets)

    def score(self, batch):
        with torch.no_grad():
            inputs, targets = batch
            outputs = self(inputs)
            loss = F.cross_entropy(outputs, targets)
            pred = torch.argmax(outputs, dim=1)
            accuracy = torch.sum(pred == targets) / len(pred)
            return [accuracy, loss]


def evaluate(model, loader):
    """Compute average of scores on each batch (unweighted by batch size)"""
    return torch.mean(torch.tensor([model.score(batch) for batch in loader]), dim=0)


class ModelCoordinator(object):
    def __init__(self, model, rank, size, step_limit, federation_scheme='shuffle'):

        assert federation_scheme in ('shuffle', 'ring')

        self.model = model
        self.rank = rank
        self.size = size
        self.federation_scheme = federation_scheme
        self._requests = {}

        self.step = torch.tensor(0)
        self.step_limit = step_limit

    def recv(self):
        # Only block on receive when not in the initial step
        # otherwise this process will wait forever for another process to communicate with it
        if self.step == 0 and self.rank == 0:
            self.step += 1
            return

        if self.federation_scheme == 'shuffle':
            prev_rank = None
        elif self.federation_scheme == 'ring':
            prev_rank = (self.rank - 1) % self.size

        dist.recv(tensor=self.step, src=prev_rank)
        if self.step == self.step_limit:
            return

        for param in self.model.parameters():
            dist.recv(tensor=param, src=prev_rank)

        self.step += 1

        # kill all other processes
        if self.step == self.step_limit:
            for rank in range(self.size):
                if rank == self.rank:
                    continue
                dist.send(tensor=self.step, dst=rank)

    def send(self):
        if self.federation_scheme == 'shuffle':
            next_rank = self.rank
            while next_rank == self.rank:
                next_rank = randint(0, self.size - 1)
        elif self.federation_scheme == 'ring':
            next_rank = (self.rank + 1) % self.size

        dist.send(tensor=self.step, dst=next_rank)
        for param in self.model.parameters():
            dist.send(tensor=param, dst=next_rank)


def train(
        model, optimizer, private_step_limit,
        train_loader, test_loader,
        coordinator, accountant,
        rank, public):

    epoch = 0
    while True:
        for batch in train_loader:

            if not public or epoch != 0:
                # synchronize weights with the previous worker
                coordinator.recv()

            if coordinator.step == private_step_limit:
                return epoch

            loss = model.loss(batch)
            loss.backward()

            # privatize the gradient and record usage
            accountant.privatize_grad()

            optimizer.step()
            optimizer.zero_grad()

            # send weights to the next worker
            if not public or epoch != 0:
                coordinator.send()

            accuracy, loss = evaluate(model, test_loader)
            printf(f"{rank: 4d} | {epoch: 5d} | {accuracy.item():.2f}     | {loss.item():.2f}", force=True)

        # privacy book-keeping
        epoch += 1
        accountant.increment_epoch()


def run_pums_worker(rank, size, private_step_limit=None, federation_scheme='shuffle', queue=None, model_filepath=None):
    """
    Perform federated learning over pums data

    :param rank: index for specific data set
    :param size: total ring size
    :param federation_scheme:
    :param private_step_limit:
    :param queue: stores values and privacy accountant usage
    :param model_filepath: indicating where to save the model checkpoint
    :return:
    """

    public = datasets[rank]['public']

    # load train data specific to the current rank
    train_loader = DataLoader(load_pums(datasets[rank]), batch_size=1000, shuffle=True)
    test_loader = DataLoader(load_pums(datasets[1]), batch_size=1000)

    model = PumsModule(len(problem['predictors']), 2)

    accountant = PrivacyAccountant(model, step_epsilon=0.001, disable=public)
    coordinator = ModelCoordinator(model, rank, size, private_step_limit, federation_scheme)

    optimizer = torch.optim.SGD(model.parameters(), .1)

    epoch = train(
        model, optimizer,
        private_step_limit,
        train_loader, test_loader,
        coordinator, accountant,
        rank, public)

    # Only save if filename is given
    if rank == size - 1 and model_filepath:
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': evaluate(model, test_loader)
        }, model_filepath)

    if queue:
        queue.put((tuple(datasets[rank].values()), accountant.compute_usage()))


def init_process(rank, size, fn, args, backend='gloo'):
    """
    Initialize the distributed environment.
    """
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'
    dist.init_process_group(backend, rank=rank, world_size=size)
    fn(rank, size, **args)


def main(worker):
    """
    Example method demonstrating ring structure running on
    multiple processes. __main__ entrypoint.
    :return:
    """
    size = len(datasets)
    processes = []
    queue = Queue()

    # Model checkpoints will be saved here
    model_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', 'model_checkpoints')
    if not os.path.exists(model_path):
        os.mkdir(model_path)

    for rank in range(size):
        p = Process(target=init_process, args=(rank, size, worker, {
            'queue': queue,
            'private_step_limit': 100,
            'model_filepath': os.path.join(model_path, 'model.pt'),
            'federation_scheme': 'shuffle'
        }))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    usage = {}
    while not queue.empty():
        rank, (epsilon, delta) = queue.get()
        usage[str(rank)] = {'epsilon': epsilon, 'delta': delta}
    print(json.dumps(usage, indent=4))


if __name__ == "__main__":
    print("Rank | Epoch | Accuracy | Loss")
    main(worker=run_pums_worker)
