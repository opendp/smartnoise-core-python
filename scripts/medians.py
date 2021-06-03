import time

import pandas as pd
import numpy as np
import torch


class Median(object):

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    def __init__(self, tensor_list, starting_point=None):
        self.tensor_list = tensor_list
        x = torch.FloatTensor(starting_point).requires_grad_() \
            if starting_point is not None else torch.zeros(tensor_list[0].size())
        self.x = x.to(self.device).detach().requires_grad_()

    def oja(self, learning_rate=0.001, max_iters=10**5, tol=0.0001, verbose=False):
        """
        :param learning_rate:
        :param max_iters:
        :param tol:
        :param verbose:
        :return:
        """
        optimizer = torch.optim.Adam([self.x], lr=learning_rate)
        y = None
        for i in range(0, max_iters+1):
            optimizer.zero_grad()
            y_previous = y.clone() if y else 0.0
            y = sum([torch.det(torch.sub(self.x, z)) for z in self.tensor_list]).to(self.device)
            y = y.requires_grad_()

            # y = torch.sub(x, tensor_list[0]).requires_grad_()
            y.backward(retain_graph=True)
            optimizer.step()
            if verbose:
                if i % 1000 == 0:
                    print(i+1, self.x, y)

            if torch.abs(torch.sub(y, y_previous)) <= tol:
                if verbose:
                    print(i, self.x, y)
                return self.x
        return self.x

    def l1_median_tensor(self, learning_rate=0.001, max_iters=10**5, tol=0.0001, verbose=False):
        """
        :param learning_rate:
        :param max_iters:
        :param tol:
        :param verbose:
        :return:
        """
        optimizer = torch.optim.Adam([self.x], lr=learning_rate)
        y = None
        for i in range(0, max_iters+1):
            optimizer.zero_grad()
            y_previous = y.clone() if y else 0.0
            y = torch.norm(sum([torch.sub(self.x, z) for z in self.tensor_list]), p=1).to(self.device)
            y = y.requires_grad_()
            y.backward(retain_graph=True)
            optimizer.step()
            if verbose:
                if i % 1000 == 0:
                    print(i+1, self.x, y)

            if torch.abs(torch.sub(y, y_previous)) <= tol:
                if verbose:
                    print(i, self.x, y)
                return self.x
        return self.x


def _performance_test(method, verbose=False):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    performance = []
    tensor_widths = [2, 5, 10, 100]
    tensor_heights = [2, 5, 10, 100]
    n_iters = 1
    list_sizes = [2, 5, 10, 20, 50]  # , 100, 500, 1000, 2000, 5000, 10_000, 50_000, 100_000]
    for m, n in zip(tensor_widths, tensor_heights):
        times = []
        for list_size in list_sizes:
            for i in range(n_iters):
                tensor_list = [torch.rand(m, n, requires_grad=True).to(device) for x in list_sizes]
                median = Median(tensor_list)
                start = time.perf_counter()
                if method == 'l1':
                    median_tensor = median.l1_median_tensor(tol=0.01, verbose=verbose)
                elif method == 'oja':
                    if m == n:
                        median_tensor = median.oja(tol=0.01, verbose=verbose)
                    else:
                        continue
                end = time.perf_counter()
                times.append(end-start)
                # print('----------------------------')
                # print(f'M: {m} N: {n} List Size: {list_size} Iter: {i}')
                # print(f'Median: {median_tensor}')
                # print(f'Time: {end-start}')
            performance.append((n, list_size, np.mean(times), np.std(times)))
    print(pd.DataFrame(performance, columns=['N', 'List Size', 'Mean', 'Stdev']))
    # print('\n--------------------------------------------------------------------------------\n')
    # # Define a list of {0, 1, ...., 10} so that median is 5
    # print("Using user-defined starting point: ")
    # tensor_list = [torch.tensor([float(x), float(x)], requires_grad=True) for x in range(0, 11)]
    # median_tensor = l1_median_tensor(tensor_list, starting_point=[2.5, 2.5], verbose=True)
    # print(f'Median: {median_tensor}')


def test_l1_median():
    print('-------------------')
    print('| L1 Median Test  |')
    print('-------------------')
    _performance_test("l1")


def test_oja_median():
    print('-------------------')
    print('| Oja Median Test |')
    print('-------------------')
    _performance_test("oja")


def first_test_oja_median():
    print('-------------------')
    print('| Oja Median Test |')
    print('-------------------')
    size = (2, 2)
    tensor_list = [torch.zeros(size), torch.eye(*size), 2*torch.eye(*size), 3*torch.eye(*size), 3*torch.eye(*size)]
    median = Median(tensor_list)
    median = median.oja(verbose=False, tol=10**-7)
    print(f'\nOja Median:\n {median}')


if __name__ == '__main__':
    test_l1_median()
    test_oja_median()
