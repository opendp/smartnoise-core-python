import copy
import datetime
import json
import math
import os
import pickle
import sys
from multiprocessing import Queue
from random import randint

import pandas as pd
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.multiprocessing import Process
from torch.utils.data import DataLoader, TensorDataset, SubsetRandomSampler

from gradient_transfer import GradientTransfer
from scripts.pums_downloader import get_pums_data_path, download_pums_data, datasets

import random
random.seed(0)

import numpy as np
np.random.seed(0)

torch.manual_seed(5)


# defaults to predicting ambulatory difficulty based on age, weight and cognitive difficulty
ACTIVE_PROBLEM = 'medicare'

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
        x = torch.sigmoid(x)
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


class StateComparison(object):

    _dataset_dicts = [
        {'year': 2010, 'record_type': 'person', 'state': 'al'},
        {'year': 2010, 'record_type': 'person', 'state': 'ms'},
        {'year': 2010, 'record_type': 'person', 'state': 'ct'},
        {'year': 2010, 'record_type': 'person', 'state': 'ma'},
        {'year': 2010, 'record_type': 'person', 'state': 'vt'},
        {'year': 2010, 'record_type': 'person', 'state': 'ri'},
        {'year': 2010, 'record_type': 'person', 'state': 'nh'},
    ]

    def __init__(self, batches=10, batch_size=10, epochs=1, learning_rate=0.001, shuffle=True, test_data_size=None):
        self.batches = batches
        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.shuffle = shuffle
        self.datasets = dict((x['state'], load_pums(x)) for x in self._dataset_dicts)
        self.models = [
            {
                'name': 'initial',
                'model': PumsModule(len(problem['predictors']), 2),
                'training_history': pd.DataFrame([])
            }
        ]
        data = self.datasets['vt']
        self.test_data_size = test_data_size if test_data_size else batch_size
        self.test_data = [next(iter(DataLoader(data, batch_size=1, shuffle=self.shuffle)))
                          for _ in range(self.test_data_size)]
        self.trainer = None

    def burn_in(self, state='al', burn_in_batches=10, burn_in_epochs=1):
        model = copy.deepcopy(self.models[-1]['model'])
        model.cuda()

        burn_in_data = self.datasets[state]

        data_loaders = {
            'burn_in_loaders': [(state, DataLoader(burn_in_data, batch_size=1, shuffle=self.shuffle), )],
        }
        optimizer = torch.optim.SGD(model.parameters(), self.learning_rate)
        trainer = GradientTransfer(data_loaders, model, optimizer, epochs=epochs)
        train_results = trainer.train(self.test_data, batches=self.batches, batch_size=self.batch_size,
                                      burn_in_epochs=burn_in_epochs, burn_in_batches=burn_in_batches)
        self.models.append({
            'name': f'burn_in_{state}',
            'model': trainer.model,
            'training_history': pd.DataFrame(train_results)
        })
        return train_results

    def train(self, state_names, burn_in_states=None, model_index=None):
        if burn_in_states is None:
            burn_in_states = []
        model_index = model_index if model_index else -1
        model = copy.deepcopy(self.models[model_index]['model'])
        model.cuda()

        burn_in_data = [(x, self.datasets.get(x),) for x in burn_in_states]
        burn_in = True if burn_in_data else False
        train_data = [(x, self.datasets[x],) for x in state_names]

        data_loaders = {
            'burn_in_loaders': [(state, DataLoader(x, batch_size=1, shuffle=self.shuffle),) for state, x in burn_in_data],
            'tr_loaders': [(state, DataLoader(x, batch_size=1, shuffle=self.shuffle),) for state, x in train_data]
        }
        optimizer = torch.optim.SGD(model.parameters(), self.learning_rate)
        self.trainer = GradientTransfer(data_loaders, model, optimizer, epochs=epochs)
        train_results = self.trainer.train(self.test_data,
                                      batches=self.batches, batch_size=self.batch_size,
                                      burn_in=burn_in, burn_in_epochs=burn_in_epochs, burn_in_batches=burn_in_batches)
        self.models.append({
            'name': ''.join(['train_on_', '_'.join(state_names)]),
            'model': self.trainer.model,
            'training_history': pd.DataFrame(train_results)
        })
        return train_results

    def save(self, base_dir='.'):
        out_path = os.path.join(base_dir, str(datetime.datetime.now()).replace(' ', '-'))
        os.makedirs(out_path)
        for i, model_dict in enumerate(self.models):
            base_filename = ''.join([model_dict['name'], '_', str(i)])
            torch.save(model_dict['model'].state_dict(), os.path.join(out_path, base_filename + '.pth'))
            model_dict['training_history'].to_csv(os.path.join(out_path, base_filename + '.csv'))
        with open(os.path.join(out_path, 'summary.json'), 'w') as outfile:
            json.dump({
                'batches': self.batches,
                'batch_size': self.batch_size,
                'epochs': self.epochs
            }, outfile)


if __name__ == "__main__":

    epochs = 4
    batch_size = 5
    batches = 250

    burn_in_epochs = 4
    burn_in_batch_size = 5
    burn_in_batches = 250

    learning_rate = 0.001

    state_comparison = StateComparison(batches=batches,
                                       batch_size=batch_size,
                                       epochs=epochs,
                                       learning_rate=learning_rate,
                                       shuffle=False)

    alabama_burn_in_result = state_comparison.burn_in(state='al',
                                                      burn_in_batches=burn_in_batches,
                                                      burn_in_epochs=burn_in_epochs)

    run_order = [
        {
            'state': 'ct',
            'model_index': 1
        },
        {
            'state': 'nh'
        },
        {
            'state': 'ma'
        },
        {
            'state': 'ri'
        },
        {
            'state': 'ct'
        },
        {
            'state': 'nh'
        },
        {
            'state': 'ma'
        },
        {
            'state': 'ri'
        }
    ]

    results = []
    for state_dict in run_order:
        results.append(state_comparison.train(['al']))
    for state_dict in run_order:
        results.append(state_comparison.train([state_dict['state']], model_index=state_dict.get('model_index')))

    state_comparison.save(base_dir='results')
