
import itertools

import torch
import numpy as np

from medians import l1_median_tensor


class DPGradientSelector(object):

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

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
        median = l1_median_tensor(self.tensor_list, max_iters=10**4)
        # We need to compare candidate tensors to a central measure. For now,
        # let's use the norm of the tensor describing the distance.
        dists = [(i, torch.norm(torch.abs(x-median))) for i, x in enumerate(candidates)]
        return sorted(dists, key=lambda x: x[1])

    def select_gradient_tensor(self, num_extra_samples=10):
        """
        Use DP Exponential mechanism to select a gradient
        :return: torch tensor
        """
        grad_mean = sum([x.cpu() for x in self.tensor_list]) / len([x.cpu() for x in self.tensor_list])
        tensor_size = tuple(self.tensor_list[0].size())
        candidates = []  # self.tensor_list.copy()
        for i in range(num_extra_samples):
            x = np.random.normal(loc=0.0, scale=1.0, size=tensor_size).astype(np.float32)
            candidates.append(torch.FloatTensor(grad_mean + x).to(self.device))
        utilities = np.array([x[1].item() for x in self.utility_function(candidates)])
        sensitivity = 1.0
        pr = np.exp(self.epsilon * utilities / (2.0 * sensitivity))
        pr = pr / np.linalg.norm(pr, ord=1)
        index = np.random.choice([i for i in range(len(candidates))], p=pr)
        return candidates[index]

    def determine_distance_measures(self):
        """
        Given a set of gradients, compute distances between
        all pairs and estimate a starting cube size for sampling.
        :return:
        """
        pairs = itertools.product(self.tensor_list, self.tensor_list)
        distances = list(filter(lambda x: x > 0., [torch.linalg.norm(torch.abs(x-y)) for x, y in pairs]))
        return {
            'average': (sum(distances) / len(distances)).item(),
            'min': min(distances).item(),
            'max': max(distances).item()
        }

    def determine_cube_size(self, gradient, num_points=1):
        """
        Use N closest points to estimate best cube size for sampling
        :param gradient:
        :param num_points:
        :return:
        """
        n_closest_points = list(sorted(filter(lambda x: x > 0., [torch.linalg.norm(torch.abs(x.cpu()-gradient))
                                                                 for x in self.tensor_list])))[:num_points]
        if not n_closest_points:
            raise ValueError("No close points")
        return sum(n_closest_points) / len(n_closest_points)

    def another_select_gradient_tensor(self,
                                       cube_size=None,
                                       min_cube_size=None,
                                       iteration_limit=1000,
                                       use_mean_distance=False,
                                       verbose=False,
                                       debug=False):
        """
        Use exp mechanism to select one of the gradients. Use rejection sampling to
        select from space around that gradient (consider cube around it).
        Select from box, then compute distance to all other gradients and see if
        it is closest to desired gradient. If yes, then it is in the polytope.
        :param cube_size: Size of starting hypercube to sample from
        :param min_cube_size: Size of smallest hypercube allowed, after which search will stop
        :param iteration_limit: Number of attempts per hypercube size
        :param use_mean_distance: If cube_size not given, use mean distance between gradients as hypercube size
        :param verbose: Print results
        :param debug: Print updates for each iteration
        :return: dict, e.g. {'point': candidate_point, 'iterations': i, 'cube_size': cube_size}
        """
        tensor_size = tuple(self.tensor_list[0].cpu().size())
        utilities = np.array([x[1].item() for x in self.utility_function()])
        # TODO: Dividing for now, to avoid Nans in pr
        utilities = utilities / sum(utilities)
        # print(f"Utilities: {utilities}")
        sensitivity = 1.0
        pr = np.exp(self.epsilon * utilities / (2.0 * sensitivity))
        pr = pr / np.linalg.norm(pr, ord=1)
        try:
            index = np.random.choice([i for i in range(len(self.tensor_list))], p=pr)
        except ValueError as ex:
            index = np.random.choice([i for i in range(len(self.tensor_list))])
            return {'point':  self.tensor_list[index].cpu(), 'iterations': 0, 'cube_size': cube_size}

        gradient = self.tensor_list[index].cpu()
        if not cube_size:
            if use_mean_distance:
                cube_size = self.determine_distance_measures()['average']
            else:
                try:
                    cube_size = self.determine_cube_size(gradient=gradient)
                except ValueError:
                    cube_size = 1.0
        min_cube_size = min_cube_size if min_cube_size else cube_size / 100.

        while True:
            total_iterations = 0
            for i in range(iteration_limit):
                if debug:
                    print(f"Iter: {i}\tCube Size: {cube_size}")
                candidate_point = gradient + \
                    torch.Tensor(np.random.uniform(low=-cube_size/2.,
                                                   high=cube_size/2.,
                                                   size=tensor_size)).cpu()
                gradient_dist = torch.norm(torch.abs(gradient-candidate_point))
                other_dists = [torch.norm(torch.abs(x.cpu()-candidate_point)) for x in self.tensor_list]
                closer_gradients = list(filter(lambda x: x < gradient_dist, other_dists))
                if not closer_gradients:
                    if verbose:
                        print(f"Found candidate point after {total_iterations} iterations with cube size {cube_size}")
                    return {'point': gradient, 'iterations': i, 'cube_size': cube_size}
                total_iterations += 1
            if cube_size <= min_cube_size:
                break
            cube_size -= cube_size / 2.

        raise Exception("Could not find point in polytope")
