import torch
import numpy as np


class DPGradientSelector(object):

    def __init__(self, tensor_list, epsilon):
        """
        Select gradient from a list using DP Median
        :param tensor_list: List of torch tensors representing gradients
        :param epsilon: privacy parameter
        """
        self.tensor_list = tensor_list
        self.epsilon = epsilon

    def tensor_mean(self):
        return torch.mean(torch.stack(self.tensor_list), dim=0)

    def utility_function(self, candidates=None):
        """
        Score a list of tensors based on distance to some central measure
        :return: tuple of (index, distance)
        """
        candidates = candidates if candidates else self.tensor_list
        mean = self.tensor_mean()
        # We need to compare candidate tensors to a central measure. For now,
        # let's use the norm of the tensor describing the distance.
        dists = [(i, torch.norm(torch.abs(x-mean))) for i, x in enumerate(candidates)]
        return sorted(dists, key=lambda x: x[1])

    def select_gradient_tensor(self, num_extra_samples=10):
        """
        Use DP Exponential mechanism to select a gradient
        :return: torch tensor
        """
        grad_mean = torch.FloatTensor(sum(self.tensor_list) / len(self.tensor_list))
        tensor_size = tuple(self.tensor_list[0].size())
        candidates = []  # self.tensor_list.copy()
        for i in range(num_extra_samples):
            x = np.random.normal(loc=0.0, scale=1.0, size=tensor_size).astype(np.float32)
            candidates.append(torch.FloatTensor(grad_mean + x))
        utilities = np.array([x[1].item() for x in self.utility_function(candidates)])
        sensitivity = 1.0
        pr = np.exp(self.epsilon * utilities / (2.0 * sensitivity))
        pr = pr / np.linalg.norm(pr, ord=1)
        index = np.random.choice([i for i in range(len(candidates))], p=pr)
        return candidates[index]
