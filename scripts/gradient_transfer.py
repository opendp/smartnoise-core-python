import multiprocessing
import time

import torch

from multiprocessing import Pool
from torch.utils.data import DataLoader
from dp_gradient_select import DPGradientSelector


class GradientTransfer(object):

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    def __init__(self, dataloader, model, optimizer=None, learning_rate=0.01, epochs=100):
        """
        Train a model, replacing the gradient at each epoch with the DP Median
        of previously calculated gradients.
        :param model: Should inherit from nn.Module
        :param train_loader: Instance of DataLoader
        :param test_loader: Instance of DataLoader
        :param learning_rate: Used by optimizer
        """
        self.dataloader = dataloader
        self.train_loader = dataloader['tr_loader']
        self.test_loader = dataloader['cv_loader']
        self.model = model
        self.learning_rate = learning_rate
        self.optimizer = optimizer if optimizer else torch.optim.SGD(model.parameters(), learning_rate)
        self.epochs = epochs
        self.gradients = dict((name, []) for name, _ in model.named_parameters())

    def _get_columns_for_median(self, name):
        """
        Used to unwrap gradients into individual components.
        e.g. The ij-th component of each gradient in the list.
        :param name: Key for self.gradients, of the form layer_name.param_name
        :return: Iterator of gradient components
        """
        for x in zip(*map(torch.flatten, self.gradients[name])):
            yield [y.item() for y in x]

    def evaluate(self):
        """Compute average of scores on each batch (unweighted by batch size)"""
        return torch.mean(torch.tensor([self.model.score(batch) for batch in self.test_loader]), dim=0)

    def _init_gradient(self):
        self.gradients = dict((name, []) for name, _ in self.model.named_parameters())

    def _calculate_gradient(self, i, sample):
        # print(f"Gradient for sample {i}")
        x = sample[0].cuda()
        y = sample[1].cuda()
        gradients = dict((name, []) for name, _ in self.model.named_parameters())
        loss = self.model.loss((x, y, ))
        loss.backward()
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                grad = param.grad.detach().clone().to(self.device)
                gradients[name].append(grad)
        return gradients

    def _unpack_gradients(self, all_gradients):
        for grad in all_gradients:
            for k, tensor_list in grad.items():
                for tensor in tensor_list:
                    self.gradients[k].append(tensor)

    def train(self, batches=10, batch_size=10):
        """
        Run training with the gradient transfer process
        :return:
        """
        start = time.time()
        total_loss = 0.0
        global_index = 0

        for epoch in range(0, self.epochs):
            data_iter = iter(self.train_loader)
            for batch in range(batches):
                try:
                    train_data = [next(data_iter) for _ in range(batch_size)]
                except StopIteration:
                    continue
                self._init_gradient()
                with multiprocessing.get_context('spawn').Pool() as p:
                    all_gradients = p.starmap(self._calculate_gradient, [(i, x) for i, x in enumerate(train_data)])
                    
                    self._unpack_gradients(all_gradients)
                    for name, grad in self.gradients.items():
                        # Use gradient selector for DP Median selection
                        dp_gradient_selector = DPGradientSelector(self.gradients[name], epsilon=1.0)
                        gradient_result = dp_gradient_selector.another_select_gradient_tensor()
                        gradient = gradient_result['point'].cuda()
                        # print(gradient)
                        # medians = dp_gradient_selector.select_gradient_tensor()
                        # Names are of the form "linear1.weight"
                        layer, param = name.split('.')
                        getattr(getattr(self.model, layer), param).grad = gradient

                self.optimizer.step()
                self.optimizer.zero_grad()

                batch_loss = 0.0
                for i, sample in enumerate(train_data):
                    loss = self.model.loss((sample[0].cuda(), sample[1].cuda(), ))
                    batch_loss += loss.item()

                    global_index += 1
                total_loss += batch_loss
                print(f'Epoch {epoch} | Batch {batch} | Batch Loss {batch_loss / batch_size} | '
                      f'Average Loss {total_loss / (global_index + 1)} | '
                      f'{1000 * (time.time() - start) / (global_index + 1)} ms/batch',
                      flush=True)

                # accuracy = self.evaluate()
                # print(f"Epoch: {epoch: 5d} | Accuracy: {accuracy.item():.2f}")  # | Loss: {loss.item():.2f}")

    def train_plain(self, batches=10, batch_size=10):
        """
        For comparison, run without any gradient transfers
        :return:
        """
        start = time.time()
        total_loss = 0.0
        global_index = 0

        for epoch in range(0, self.epochs):
            data_iter = iter(self.train_loader)
            for batch in range(batches):
                try:
                    train_data = [next(data_iter) for _ in range(batch_size)]
                except StopIteration:
                    continue
                for sample in train_data:
                    x = sample[0].cuda()
                    y = sample[1].cuda()
                    loss = self.model.loss((x, y, ))
                    loss.backward()
                    self.optimizer.step()
                    self.optimizer.zero_grad()

                batch_loss = 0.0
                for i, sample in enumerate(train_data):
                    loss = self.model.loss((sample[0].cuda(), sample[1].cuda(), ))
                    batch_loss += loss.item()

                    global_index += 1

                total_loss += batch_loss
                print(f'Epoch {epoch} | Batch {batch} | Batch Loss {batch_loss / batch_size} | '
                      f'Average Loss {total_loss / (global_index + 1)} | '
                      f'{1000 * (time.time() - start) / (global_index + 1)} ms/batch',
                      flush=True)
