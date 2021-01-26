import json
from multiprocessing import Queue

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import random_split, DataLoader, TensorDataset
import sys


import os
import torch.distributed as dist
from torch.multiprocessing import Process

from opendp.smartnoise.network.optimizer import PrivacyAccountant
from scripts.pums_downloader import get_pums_data_path, download_pums_data, datasets

# defaults to predicting ambulatory difficulty based on age, weight and cognitive difficulty
predictors = ['AGEP', 'PWGTP', 'DREM']
target = 'DPHY'

debug = False

# overkill flushing
def printf(x, force=False):
    if debug or force:
        print(x, flush=True)
        sys.stdout.flush()


class PumsModule(nn.Module):
    def __init__(self, input_size, output_size):
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


def run_ring(rank, size, epochs, queue=None):

    # load the data specific to the current rank
    download_pums_data(**datasets[rank])
    data_path = get_pums_data_path(**datasets[rank])

    data = pd.read_csv(data_path, usecols=predictors + [target], engine='python')
    data.dropna(inplace=True)
    data = TensorDataset(
        torch.from_numpy(data[predictors].to_numpy()).type(torch.float32),
        torch.from_numpy(data[target].to_numpy()).type(torch.LongTensor) - 1)

    # split
    test_split = int(len(data) * .2)
    train_split = len(data) - test_split
    train_set, test_set = random_split(data, [train_split, test_split])

    train_loader = DataLoader(train_set, batch_size=1000, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=1)

    model = PumsModule(len(predictors), 2)

    next_rank, prev_rank = ((rank + offset) % size for offset in (1, -1))

    first = True

    with PrivacyAccountant(model, step_epsilon=0.01) as accountant:
        optimizer = torch.optim.Adam(model.parameters(), .1)

        for epoch in range(epochs):
            # Only receive when not in the first run around the ring,
            # otherwise process will sit and wait for rank "-1" to finish
            if not first or rank != 0:
                printf(f'rank {rank} is blocking waiting for prev_rank {prev_rank}')
                for param in model.parameters():
                    dist.recv(tensor=param, src=prev_rank)
                printf(f'rank {rank} is unblocked')
            printf(f'starting {rank}, epoch {epoch}')

            first = False
            for batch in train_loader:
                loss = model.loss(batch)
                loss.backward()

                # before
                accountant.privatize_grad()

                optimizer.step()
                optimizer.zero_grad()
            accountant.increment_epoch()

            accuracy, loss = evaluate(model, test_loader)
            printf(f"{rank: 4d} | {epoch: 5d} | {accuracy.item():.2f}     | {loss.item():.2f}", force=True)

            # Ensure that send() does not happen on the last epoch of the last node,
            # since this would send back to the first node (which is done) and hang
            if rank == size - 1 and epoch == epochs - 1:
                # TODO: checkpoint model to disk
                pass
            else:
                for param in model.parameters():
                    # https://pytorch.org/docs/stable/distributed.html#torch.distributed.send
                    dist.send(tensor=param, dst=next_rank)

    if queue:
        queue.put((tuple(datasets[rank].values()), accountant.compute_usage()))


def init_process(rank, size, fn, args, backend='gloo'):
    """ Initialize the distributed environment. """
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'
    dist.init_process_group(backend, rank=rank, world_size=size)
    fn(rank, size, **args)


def run_parallel():
    size = 3
    processes = []
    queue = Queue()
    print("Rank | Epoch | Accuracy | Loss")
    for rank in range(size):
        p = Process(target=init_process, args=(rank, size, run_ring, {
            'queue': queue,
            'epochs': 3
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
    run_parallel()
