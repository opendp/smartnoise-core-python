import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.datasets import load_iris
from torch.utils.data import random_split, DataLoader, TensorDataset

import os
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.dataset import Dataset

from opendp.smartnoise.network.optimizer import PrivacyAccountant
from .pums_downloader import get_pums_data_path, download_pums_data, datasets


class PumsDataset(Dataset):
    """PUMS dataset."""

    def __init__(self, csv_path, predictors):
        """
        Args:
            csv_path (string): Path to the csv file.
            transform (callable, optional): Optional transform to be applied on each sample.
        """
        self.data = pd.read_csv(csv_path, usecols=predictors)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        return self.data.iloc[idx].to_numpy(dtype=float)


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


def test_ddp_fn(rank, world_size):
    setup(rank, world_size)

    raise NotImplementedError("switch from iris to pums data here")
    iris_sklearn = load_iris(as_frame=True)
    data = iris_sklearn['data']
    target = iris_sklearn['target']

    # predictors = ['petal length (cm)', 'petal width (cm)']
    predictors = data.columns.values
    input_columns = torch.from_numpy(data[predictors].to_numpy()).type(torch.float32)
    output_columns = torch.tensor(target)

    data = TensorDataset(input_columns, output_columns)

    rows = input_columns.shape[0]
    test_split = int(rows * .2)
    train_split = rows - test_split

    train_set, test_set = random_split(data, [train_split, test_split])

    train_loader = DataLoader(train_set, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=1)

    model = PumsModule(len(predictors), 3).to(rank)

    with PrivacyAccountant(model) as accountant:
        model = DDP(model, device_ids=[rank])

        optimizer = torch.optim.Adam(model.parameters(), learning_rate)
        print("Epoch | Accuracy | Loss")

        for epoch in range(epochs):
            for batch in train_loader:
                loss = model.loss(batch)
                loss.backward()
                accountant.privatize_grad(optimizer)

                optimizer.step()
                optimizer.zero_grad()
                # for name, param in model.named_parameters():
                #     print(name, param.grad1.size())
                # print("epoch activations: ", epoch)
                # print(activations)

            accuracy, loss = evaluate(model, test_loader)
            print(f"{epoch: 5d} | {accuracy.item():.2f}     | {loss.item():.2f}")

    cleanup()


def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'
    dist.init_process_group(backend='gloo', rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()


if __name__ == "__main__":
    world_size = 2
    hook_before = True
    mp.spawn(test_ddp_fn, args=(world_size, hook_before), nprocs=world_size, join=True)
