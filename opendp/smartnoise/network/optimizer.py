"""
Library for extracting interesting quantites from autograd, see README.md

Not thread-safe because of module-level variables

Notation:
o: number of output classes (exact Hessian), number of Hessian samples (sampled Hessian)
n: batch-size
do: output dimension (output channels for convolution)
di: input dimension (input channels for convolution)
Hi: per-example Hessian of matmul, shaped as matrix of [dim, dim], indices have been row-vectorized
Hi_bias: per-example Hessian of bias
Oh, Ow: output height, output width (convolution)
Kh, Kw: kernel height, kernel width (convolution)

Jb: batch output Jacobian of matmul, output sensitivity for example,class pair, [o, n, ....]
Jb_bias: as above, but for bias

A, activations: inputs into current layer
B, backprops: backprop values (aka Lop aka Jacobian-vector product) observed at current layer

"""
import math
from typing import List

import torch
import torch.nn as nn
from opendp.smartnoise.core.api import LibraryWrapper
core_library = LibraryWrapper()

_supported_layers = (nn.Linear, nn.Conv2d)  # Supported layer class types

from torch.nn.parallel import DistributedDataParallel as DDP


class PrivacyAccountant(object):
    def __init__(self, model: nn.Module, epoch_epsilon, epoch_delta):
        self.model = model
        self._hooks_enabled = False  # work-around for https://github.com/pytorch/pytorch/issues/25723
        self.epoch_epsilon = epoch_epsilon
        self.epoch_delta = epoch_delta

    def __enter__(self):
        """
        Adds hooks to model to save activations and backprop values.

        The hooks will
        1. save activations into param.activations during forward pass
        2. append backprops to params.backprops_list during backward pass.

        Use __exit__ to disable this.
        """

        assert not hasattr(self.model, 'autograd_hacks_hooks'), "Attempted to install hooks twice"

        if self._hooks_enabled:
            # hooks have already been added
            return

        self._hooks_enabled = True

        def capture_activations(layer: nn.Module, input: List[torch.Tensor], _output: torch.Tensor):
            """Save activations into layer.activations in forward pass"""
            if not self._hooks_enabled:
                return
            setattr(layer, "activations", input[0].detach())

        def capture_backprops(layer: nn.Module, _input, output):
            """Append backprop to layer.backprops_list in backward pass."""
            if not self._hooks_enabled:
                return
            if not hasattr(layer, 'backprops_list'):
                setattr(layer, 'backprops_list', [])
            layer.backprops_list.append(output[0].detach())

        self.model.autograd_hacks_hooks = []
        for layer in self.model.modules():
            if isinstance(layer, _supported_layers):
                self.model.autograd_hacks_hooks.extend([
                    layer.register_forward_hook(capture_activations),
                    layer.register_backward_hook(capture_backprops)
                ])

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Remove hooks added by __enter__
        """

        # assert self.model == 0, "not working, remove this after fix to https://github.com/pytorch/pytorch/issues/25723"

        if not hasattr(self.model, 'autograd_hacks_hooks'):
            print("Warning, asked to remove hooks, but no hooks found")
        else:
            for handle in self.model.autograd_hacks_hooks:
                handle.remove()
            del self.model.autograd_hacks_hooks

        self._hooks_enabled = False

    def privatize_grad(self, layer, clipping_norm, loss_type: str = 'mean') -> None:
        """
        Compute per-example gradients for a layer, privatize them, and save them under 'param.grad'.
        Must be called after loss.backprop()
        Must be called before optimizer.step()

        Args:
            model:
            loss_type: either "mean" or "sum" depending whether backpropped loss was averaged or summed over batch
        """

        assert loss_type in ('sum', 'mean')

        if not isinstance(layer, _supported_layers):
            return

        assert hasattr(layer, 'activations'), "No activations detected, run forward after add_hooks(model)"
        assert hasattr(layer, 'backprops_list'), "No backprops detected, run backward after add_hooks(model)"
        assert len(layer.backprops_list) == 1, "Multiple backprops detected"

        A = layer.activations
        B = layer.backprops_list[0]

        n = A.shape[0]

        reducer = torch.sum
        if loss_type == 'mean':
            # could be moved into clipping norm
            B *= n
            reducer = torch.mean

        sigma = 1 + math.sqrt(2 * math.log(1 / self.epoch_delta))
        if isinstance(layer, nn.Linear):
            # reconstruct
            grad_instance = torch.einsum('ni,nj->nij', B, A)
            # clip
            bound = torch.linalg.norm(grad_instance, ord=2, dim=1) / clipping_norm
            grad_instance /= torch.max(torch.ones_like(bound), bound)[:, None, :]
            # reduce
            grad = reducer(grad_instance, dim=0)
            # noise
            grad.apply_(lambda x: x + core_library.gaussian_noise(sigma))

            setattr(layer.weight, 'grad', grad)
            if layer.bias is not None:
                pass
                # setattr(layer.bias, 'grad', torch.sum(B))

        # elif isinstance(layer, nn.Conv2d):
        #     A = torch.nn.functional.unfold(A, layer.kernel_size)
        #     B = B.reshape(n, -1, A.shape[-1])
        #     grad1 = torch.einsum('ijk,ilk->ijl', B, A)
        #     shape = [n] + list(layer.weight.shape)
        #     setattr(layer.weight, 'grad1', grad1.reshape(shape))
        #     if layer.bias is not None:
        #         setattr(layer.bias, 'grad1', torch.sum(B, dim=2))

        del layer.backprops_list
