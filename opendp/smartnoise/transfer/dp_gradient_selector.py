import torch

import opendp.smartnoise.core as sn


class DPGradientSelector(object):

    def __init__(self, tensor_list):
        """
        Select gradient from a list using DP Median
        :param tensor_list: List of torch tensors representing gradients
        """
        self.tensor_list = tensor_list

    def tensor_mean(self):
        return torch.mean(torch.stack(self.tensor_list), dim=0)

    def utility_function(self):
        """
        Score a list of tensors based on distance to some central measure
        :return:
        """
        mean = self.tensor_mean()
        # We need to compare candidate tensors to a central measure. For now,
        # let's use the norm of the tensor describing the distance.
        dists = [(i, torch.norm(torch.abs(x-mean))) for i, x in enumerate(self.tensor_list)]
        return sorted(dists, key=lambda x: x[1])

    # Currently throwing the following error:
    # "Error: at node_id 5
    #  Caused by: node specification Impute(Impute):
    #  Caused by: lower is greater than upper"
    def select_gradient_tensor(self):
        """
        Use DP Exponential mechanism to select a gradient
        :return: torch tensor
        """
        utilities = [x[1].item() for x in self.utility_function()]
        candidates = [x.mean().item() for x in self.tensor_list]
        with sn.Analysis() as analysis:
            median_scores = sn.median(
                utilities,
                candidates=candidates,
                data_rows=len(utilities),
                data_lower=-100.,
                data_upper=10.)
            selected_gradient = sn.exponential_mechanism(median_scores,
                                                         candidates=candidates,
                                                         privacy_usage={"epsilon": 1.})
        analysis.release()
        return selected_gradient