"""
Tools for building differentially private models.
Thanks to https://github.com/cybertronai/autograd-hacks for demonstrating gradient hacks.

A, activations: inputs into current layer
B, backprops: backprop values (aka Lop aka Jacobian-vector product) observed at current layer

"""
import math
from typing import List

import torch
import torch.nn as nn
from opendp.smartnoise.core.api import LibraryWrapper
core_library = LibraryWrapper()

_supported_modules = (nn.Linear, nn.Conv2d, nn.LSTM)  # Supported layer class types


class PrivacyAccountant(object):
    def __init__(self, model: nn.Module, step_epsilon, step_delta=0., hook=True):
        """
        Context manager for tracking privacy usage
        :param model: pyTorch model
        :param step_epsilon:
        :param step_delta:
        :param hook: whether to call hook() on __init__
        """
        self.model = model
        self._hooks_enabled = False  # work-around for https://github.com/pytorch/pytorch/issues/25723
        self.step_epsilon = step_epsilon
        self.step_delta = step_delta
        self._epochs = []
        self.steps = 0

        if hook:
            self.hook()

    def hook(self):
        """
        Adds hooks to model to save activations and backprop values.

        The hooks will
        1. save activations into param.activations during forward pass
        2. append backprops to params.backprops_list during backward pass.

        Use unhook to disable this.
        """

        if self._hooks_enabled:
            # hooks have already been added
            return self

        self._hooks_enabled = True

        def capture_activations(layer: nn.Module, input: List[torch.Tensor], _output: torch.Tensor):
            """Save activations into layer.activations in forward pass"""
            if not self._hooks_enabled:
                return
            setattr(layer, "activations", input[0].detach())

        def capture_backprops(layer: nn.Module, _input, output):
            """Save backprops into layer.backprops in backward pass"""
            if not self._hooks_enabled:
                return
            setattr(layer, 'backprops', output[0].detach())

        self.model.autograd_hacks_hooks = []
        for layer in self.model.modules():
            if isinstance(layer, _supported_modules):
                self.model.autograd_hacks_hooks.extend([
                    layer.register_forward_hook(capture_activations),
                    layer.register_backward_hook(capture_backprops)
                ])

        return self

    def unhook(self):
        """
        Remove and deactivate hooks added by .hook()
        """

        # This issue indicates that hooks are not actually removed if the forward pass is run
        # https://github.com/pytorch/pytorch/issues/25723
        # Based on testing, the hooks are actually removed
        # Since hooks are removed, there is not an accumulation of hooks if the context manager is used within a loop

        if not hasattr(self.model, 'autograd_hacks_hooks'):
            print("Warning, asked to remove hooks, but no hooks found")
        else:
            for handle in self.model.autograd_hacks_hooks:
                handle.remove()
            del self.model.autograd_hacks_hooks

        # The _hooks_enabled flag is a secondary fallback if hooks aren't removed
        self._hooks_enabled = False

    def __enter__(self):
        return self.hook()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.unhook()

    def _calculate_norm(self, params):
        param_sum_squares = []
        for param in params:
            if not isinstance(param, _supported_modules):
                continue

            assert hasattr(param, 'activations'), "No activations detected, run forward after .hook()"
            assert hasattr(param, 'backprops'), "No backprops detected, run backward after .hook()"

            A = param.activations
            B = param.backprops

            if isinstance(param, nn.Linear):
                param_sum_squares.append(torch.einsum('ni,ni,nj,nj->n', A, A, B, B))

                if param.bias is not None:
                    param_sum_squares.append((B * B).sum(dim=1))

        return torch.sqrt(torch.stack(param_sum_squares, dim=1).sum(dim=0))

    def _calculate_and_privatize(self, grad_instance, loss_type, actual_norm, clipping_norm):
        """

        :param grad_instance:
        :param loss_type:
        :param actual_norm:
        :param clipping_norm:
        :return:
        """
        # clip
        grad_instance /= torch.max(torch.ones_like(actual_norm), actual_norm / clipping_norm)[:, None, :]
        # reduce
        grad = {'sum': nn.sum, 'mean': nn.mean}[loss_type](grad_instance, dim=0)
        # noise
        sensitivity = clipping_norm / {'sum': 1., 'mean': grad_instance.shape[0]}[loss_type]
        if self.step_delta == 0.:
            grad.apply_(lambda x: core_library.snapping_mechanism(
                value=x,
                epsilon=self.step_epsilon,
                sensitivity=sensitivity,
                min=-clipping_norm,
                max=clipping_norm,
                enforce_constant_time=False))
        else:
            grad.apply_(lambda x: core_library.analytic_gaussian_mechanism(
                value=x,
                epsilon=self.step_epsilon,
                delta=self.step_delta,
                sensitivity=sensitivity,
                enforce_constant_time=False))

        return grad


    def privatize_grad(self, *, params=None, clipping_norm=1., loss_type: str = 'mean') -> None:
        """
        Compute per-example gradients for a layer, privatize them, and save them under 'param.grad'.
        Must be called after loss.backprop()
        Must be called before optimizer.step()

        :param params:
        :param clipping_norm:
        :param loss_type:  either "mean" or "sum" depending whether backpropped loss was averaged or summed over batch
        """
        assert loss_type in ('sum', 'mean')

        self.steps += 1

        params = params or self.model.named_parameters()
        for param_name, param in params or self.model.named_parameters():
            if not isinstance(param, _supported_modules):
                return

        params = params or self.model.modules()

        # compute L2 norm for flat clipping
        actual_norm = self._calculate_norm(params)

        # reconstruct instance-level gradients and reduce privately
        for param in params:
            if not isinstance(param, _supported_modules):
                continue

            n = param.activations.shape[0]

            if loss_type == 'mean':
                param.backprops *= n

            if isinstance(param, nn.Linear):
                # reconstruct
                grad_instance = torch.einsum('ni,nj->nij', param.backprops, param.activations)
                setattr(param.weight, 'grad', self._calculate_and_privatize(
                    grad_instance, loss_type, actual_norm, clipping_norm))
                if param.bias is not None:
                    setattr(param.bias, 'grad', self._calculate_and_privatize(
                        param.backprops, loss_type, actual_norm, clipping_norm))

            if isinstance(param, nn.LSTM):
                # print("Privatizing LSTM layer")
                # if not lstm_privatized:
                #     for hidden_layer, output_layer in zip(self.model.lstm_hidden, self.model.lstm_output):
                # self._calculate_and_privatize(hidden_layer, output_layer, reducer, sigma, param)
                # lstm_privatized = True
                pass

            if isinstance(param, nn.Conv2d):
                # A = torch.nn.functional.unfold(A, layer.kernel_size)
                # B = B.reshape(n, -1, A.shape[-1])
                # grad1 = torch.einsum('ijk,ilk->ijl', B, A)
                # shape = [n] + list(layer.weight.shape)
                # setattr(layer.weight, 'grad1', grad1.reshape(shape))
                # if layer.bias is not None:
                #     setattr(layer.bias, 'grad1', torch.sum(B, dim=2))
                pass

            del param.backprops_list
            del param.activations

    def make_private_optimizer(self, optimizer, *args, **kwargs):

        class _PrivacyOptimizer(optimizer):
            """Extend *Optimizer with custom step function."""

            def __init__(_self, *_args, **_kwargs):
                _self.accountant = self
                super().__init__(*_args, **_kwargs)

            def step(_self, *_args, **_kwargs):
                r"""Performs a single optimization step (parameter update)."""
                with _self.accountant:
                    _self.accountant.privatize_grad(*_args, **_kwargs)
                return super().step(*_args, **_kwargs)

        return _PrivacyOptimizer(*args, **kwargs)

    def increment_epoch(self):
        if self.steps:
            self._epochs.append(self.steps)
        self.steps = 0

    def compute_usage(self, suggested_delta=None):
        """
        Compute epsilon/delta privacy usage for all tracked epochs
        :param suggested_delta: delta to
        :return:
        """
        epsilon = 0
        delta = 0

        for batch_len in self._epochs:
            if suggested_delta is None:
                delta = 2 * math.exp(-batch_len / 16 * math.exp(-self.step_epsilon)) + 1E-8
            else:
                suggested_delta / len(self._epochs)

            batch_epsilon, batch_delta = core_library.shuffle_amplification(
                step_epsilon=self.step_epsilon,
                step_delta=self.step_delta,
                delta=delta,
                steps=batch_len)

            epsilon += batch_epsilon
            delta += batch_delta

        return epsilon, delta

#
# _supported_modules_2 = {
#     nn.Linear:
# }