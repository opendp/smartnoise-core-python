import json
import os
import sys

from multiprocessing import Queue

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

from torch.utils.data import random_split, DataLoader, TensorDataset
from torch.multiprocessing import Process

from opendp.smartnoise.network.optimizer import PrivacyAccountant
from scripts.pums_downloader import get_pums_data_path, download_pums_data, datasets

# defaults to predicting ambulatory difficulty based on age, weight and cognitive difficulty
predictors = ['AGEP', 'PWGTP', 'DREM']
target = 'DPHY'

debug = False


# overkill flushing
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


class LstmModule(nn.Module):

    def __init__(self, embedding_size, internal_size, vocab_size, tagset_size):
        super().__init__()
        self.embedding_size = embedding_size
        self.hidden_size = internal_size
        self.vocab_size = vocab_size
        self.tagset_size = tagset_size

        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.lstm = nn.LSTM(embedding_size, internal_size)
        self.hidden2tag = nn.Linear(internal_size, tagset_size)
        self.lstm_out = []
        self.lstm_hidden = []

    def forward(self, x):
        embed = self.embedding(x)
        hidden = self._init_hidden()

        # the second dimension refers to the batch size, which we've hard-coded
        # it as 1 throughout the example
        out, hidden = self.lstm(embed.view(len(x), 1, -1), hidden)
        self.lstm_out.append(out)
        self.lstm_hidden.append(hidden)
        output = self.hidden2tag(out.view(len(x), -1))
        return output

    def _init_hidden(self):
        # the dimension semantics are [num_layers, batch_size, hidden_size]
        return (torch.rand(1, 1, self.hidden_size),
                torch.rand(1, 1, self.hidden_size))


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


def run_ring(rank, size, epochs, queue=None, model_filepath=None):
    """
    Perform federated learning in a ring structure
    :param rank: index for specific data set
    :param size: total ring size
    :param epochs: number of training epochs
    :param queue: stores values and privacy accountant usage
    :param model_filepath: indicating where to save the model checkpoint
    :return:
    """

    # load the data specific to the current rank
    download_pums_data(**datasets[rank])
    data_path = get_pums_data_path(**datasets[rank])

    data = pd.read_csv(data_path, usecols=predictors + [target], engine='python')
    data.dropna(inplace=True)
    data = TensorDataset(
        torch.from_numpy(data[predictors].to_numpy()).type(torch.float32),
        torch.from_numpy(data[target].to_numpy()).type(torch.LongTensor) - 1)

    # split data into training and testing
    test_split = int(len(data) * .2)
    train_split = len(data) - test_split
    train_set, test_set = random_split(data, [train_split, test_split])

    train_loader = DataLoader(train_set, batch_size=1000, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=1)

    model = PumsModule(len(predictors), 2)

    # Get the next and previous indices
    next_rank, prev_rank = ((rank + offset) % size for offset in (1, -1))

    first = True

    accountant = PrivacyAccountant(model, step_epsilon=0.01)

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
            # Only save if filename is given
            if model_filepath:
                torch.save({'epoch': epoch,
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'loss': loss
                            }, model_filepath)
        else:
            for param in model.parameters():
                # https://pytorch.org/docs/stable/distributed.html#torch.distributed.send
                dist.send(tensor=param, dst=next_rank)

    if queue:
        queue.put((tuple(datasets[rank].values()), accountant.compute_usage()))


def run_lstm_ring(rank, size, epochs, queue=None, model_filepath=None):
    """
    Perform federated learning in a ring structure
    :param rank: index for specific data set
    :param size: total ring size
    :param epochs: number of training epochs
    :param queue: stores values and privacy accountant usage
    :param model_filepath: indicating where to save the model checkpoint
    :return:
    """

    # Every node gets same data for now
    EMBEDDING_SIZE = 6
    HIDDEN_SIZE = 6

    training_data = [
        ("The dog ate the apple".split(), ["DET", "NN", "V", "DET", "NN"]),
        ("Everybody read that book".split(), ["NN", "V", "DET", "NN"])
    ]

    idx_to_tag = ['DET', 'NN', 'V']
    tag_to_idx = {'DET': 0, 'NN': 1, 'V': 2}

    word_to_idx = {}
    for sent, tags in training_data:
        for word in sent:
            if word not in word_to_idx:
                word_to_idx[word] = len(word_to_idx)

    def prepare_sequence(seq, to_idx):
        """Convert sentence/sequence to torch Tensors"""
        idxs = [to_idx[w] for w in seq]
        return torch.LongTensor(idxs)

    seq = training_data[0][0]
    inputs = prepare_sequence(seq, word_to_idx)

    model = LstmModule(EMBEDDING_SIZE, HIDDEN_SIZE, len(word_to_idx), len(tag_to_idx))
    criterion = nn.CrossEntropyLoss()

    # Get the next and previous indices
    next_rank, prev_rank = ((rank + offset) % size for offset in (1, -1))

    first = True

    with PrivacyAccountant(model, step_epsilon=0.01) as accountant:
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

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
            for sentence, tags in training_data:
                sentence = prepare_sequence(sentence, word_to_idx)
                target = prepare_sequence(tags, tag_to_idx)

                output = model(sentence)
                loss = criterion(output, target)
                loss.backward()

                # before
                accountant.privatize_grad()

                optimizer.step()
                optimizer.zero_grad()
            accountant.increment_epoch()

            inputs = prepare_sequence(training_data[0][0], word_to_idx)
            tag_scores = model(inputs)
            tag_scores = tag_scores.detach().numpy()
            tag = [idx_to_tag[idx] for idx in np.argmax(tag_scores, axis = 1)]
            printf(f"{rank: 4d} | {epoch: 5d} | {tag}     | {training_data[0][1]}", force=True)

            # Ensure that send() does not happen on the last epoch of the last node,
            # since this would send back to the first node (which is done) and hang
            if rank == size - 1 and epoch == epochs - 1:
                # Only save if filename is given
                if model_filepath:
                    torch.save({'epoch': epoch,
                                'model_state_dict': model.state_dict(),
                                'optimizer_state_dict': optimizer.state_dict(),
                                'loss': loss
                                }, model_filepath)
            else:
                for param in model.parameters():
                    # https://pytorch.org/docs/stable/distributed.html#torch.distributed.send
                    dist.send(tensor=param, dst=next_rank)

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


def run():
    """
    Example method demonstrating ring structure running on
    multiple processes. ___main__ entrypoint.
    :return:
    """
    size = 3
    processes = []
    queue = Queue()

    # Model checkpoints will be saved here
    model_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', 'model_checkpoints')
    if not os.path.exists(model_path):
        os.mkdir(model_path)

    print("Rank | Epoch | Accuracy | Loss")
    for rank in range(size):
        p = Process(target=init_process, args=(rank, size, run_ring, {
            'queue': queue,
            'epochs': 3,
            'model_filepath': os.path.join(model_path, 'model.pt')

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


def run_lstm():
    """
    Example method demonstrating ring structure running on
    multiple processes. ___main__ entrypoint.
    :return:
    """
    size = 3
    processes = []
    queue = Queue()

    # Model checkpoints will be saved here
    model_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', 'model_checkpoints')
    if not os.path.exists(model_path):
        os.mkdir(model_path)

    print("Rank | Epoch | Predicted | Actual")
    for rank in range(size):
        p = Process(target=init_process, args=(rank, size, run_lstm_ring, {
            'queue': queue,
            'epochs': 3,
            'model_filepath': os.path.join(model_path, 'model.pt')

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
    run_lstm()
