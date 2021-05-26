import numpy as np
import torch

from scipy.optimize import minimize


def l1_median(tensor_list, starting_point):
    """
    Given a list of tensors, find the L1 median,
    defined as the point that minimizes the sum of the distances
    :param tensor_list:
    :param starting_point:
    :return:
    """
    func = lambda x: np.linalg.norm(np.array(sum([x - tensor for tensor in tensor_list])))
    return minimize(func, starting_point)


def l1_median_tensor(tensor_list, starting_point=None, tol=0.0001):
    """
    :param tensor_list:
    :return:
    """
    x = torch.FloatTensor(starting_point).requires_grad_() \
        if starting_point else torch.zeros(tensor_list[0].size(), requires_grad=True)
    optimizer = torch.optim.Adam([x], lr=0.001)

    for i in range(0, 100000):
        optimizer.zero_grad()
        y = torch.norm(sum([torch.sub(x, z) for z in tensor_list]))
        y = y.requires_grad_()

        # y = torch.sub(x, tensor_list[0]).requires_grad_()
        y.backward(retain_graph=True)
        optimizer.step()
        if i % 1000 == 0:
            print(i+1, x, y)

        if y <= tol:
            print(i, x, y)
            return x


if __name__ == '__main__':
    # Define a list of {0, 1, ...., 10} so that median is 5
    print("Using default starting point: ")
    tensor_list = [torch.tensor([float(x), float(x)], requires_grad=True) for x in range(0, 11)]
    median_tensor = l1_median_tensor(tensor_list)
    print(f'Median: {median_tensor}')
    print('\n--------------------------------------------------------------------------------\n')
    print("Using user-defined starting point: ")
    tensor_list = [torch.tensor([float(x), float(x)], requires_grad=True) for x in range(0, 11)]
    median_tensor = l1_median_tensor(tensor_list, starting_point=[2.5, 2.5])
    print(f'Median: {median_tensor}')