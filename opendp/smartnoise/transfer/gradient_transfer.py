import numpy as np
import torch

from torch.utils.data import DataLoader

from opendp.smartnoise.transfer.dp_gradient_selector import DPGradientSelector
from scripts.pums_downloader import datasets
from scripts.pums_sgd import PumsModule, load_pums, problem, evaluate


class GradientTransfer(object):

    def __init__(self, model, train_loader, test_loader, learning_rate=0.01):
        """
        Train a model, replacing the gradient at each epoch with the DP Median
        of previously calculated gradients.
        :param model: Should inherit from nn.Module
        :param train_loader: Instance of DataLoader
        :param test_loader: Instance of DataLoader
        :param learning_rate: Used by optimizer
        """
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.learning_rate = learning_rate
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

    def run(self, epoch_size=5):
        """
        Run training with the gradient transfer process
        :param epoch_size: Number of epochs to train each batch
        :return:
        """
        optimizer = torch.optim.SGD(model.parameters(), self.learning_rate)
        print("Epoch | Accuracy | Loss")
        for epoch in range(0, epoch_size):
            for batch in self.train_loader:
                loss = model.loss(batch)
                loss.backward()

                for name, param in model.named_parameters():
                    if param.requires_grad:
                        grad = param.grad.detach().clone()
                        self.gradients[name].append(grad)
                optimizer.step()
                optimizer.zero_grad()

            for name, grad in self.gradients.items():
                columns = self._get_columns_for_median(name)

                # TODO: Use gradient selector for DP Median selection
                # gradient_selector = DPGradientSelector(self.gradients[name])
                # dp_median = gradient_selector.select_gradient_tensor()
                # For now, just use this np.median call
                medians = torch.Tensor([np.median(col) for col in columns]).reshape(*self.gradients[name][0].shape)

                # Names are of the form "linear1.weight"
                layer, param = name.split('.')
                getattr(getattr(model, layer), param).grad = medians

            optimizer.step()
            optimizer.zero_grad()

            accuracy, loss = evaluate(model, test_loader)
            print(f"{epoch: 5d} | {accuracy.item():.2f}     | {loss.item():.2f}")

    def run_plain(self, epoch_size=5):
        """
        For comparison, run without any gradient transfers
        :return:
        """
        optimizer = torch.optim.Adam(model.parameters(), self.learning_rate)
        print("Epoch | Accuracy | Loss")
        for epoch in range(epoch_size):
            for batch in self.train_loader:
                loss = model.loss(batch)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
            accuracy, loss = evaluate(model, test_loader)
            print(f"{epoch: 5d} | {accuracy.item():.2f}     | {loss.item():.2f}")


if __name__ == '__main__':
    batch_size = 1000
    train_loader = DataLoader(load_pums(datasets[1]), batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(load_pums(datasets[1]), batch_size=batch_size)

    model = PumsModule(len(problem['predictors']), 2)
    engine = GradientTransfer(model, train_loader, test_loader)
    print("Running with gradient transfer: ")
    engine.run(epoch_size=5)
    print()

    gradients = engine.gradients['linear1.weight']
    gradient_selector = DPGradientSelector(gradients)

    print("Example mean: ")
    print(gradient_selector.tensor_mean())

    print("Sorted: ")
    print(gradient_selector.utility_function())

    print("DP Median Selected Gradient: ")
    selected = gradient_selector.select_gradient_tensor()
    print(selected)

    # print("Running plain: ")
    # engine.run_plain(epoch_size=20)