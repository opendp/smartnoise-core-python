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
        candidates = [float(x) for x in range(0, len(self.tensor_list))]

        with sn.Analysis():
            temp = sn.impute(sn.resize(sn.to_float(sn.Dataset(value=utilities)),
                             number_columns=1,
                             number_rows=len(utilities),
                             lower=-5000.,
                             upper=5000.),
                             lower=-5000.,
                             upper=5000.)
            selected = int(sn.dp_median(temp, candidates=candidates, privacy_usage={"epsilon": 1.}).value)
            return self.tensor_list[selected]
