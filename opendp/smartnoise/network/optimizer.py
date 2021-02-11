"""
Tools for building differentially private models.
Thanks to https://github.com/cybertronai/autograd-hacks for demonstrating gradient hacks.

A, activations: inputs into current module
B, backprops: backprop values (aka Jacobian-vector product) observed at current module

"""
import math
from typing import List

import torch
import torch.nn as nn
from opendp.smartnoise.core.api import LibraryWrapper
from opendp.smartnoise.network.attention import CatBias
from opendp.smartnoise.network.bahdanau import BahdanauAttentionScale

core_library = LibraryWrapper()

CHECK_CORRECTNESS = True
torch.set_printoptions(sci_mode=False)


class PrivacyAccountant(object):
    def __init__(self, model: nn.Module, step_epsilon, step_delta=0., hook=True, disable=False):
        """
        Utility for tracking privacy usage
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
        self._disable = disable

        if not disable and hook:
            self.hook()

    def hook(self):
        """
        Adds hooks to model to save activations and backprop values.

        The hooks will
        1. save activations into module.activations during forward pass
        2. save backprops into module.backprops during backward pass.

        Use unhook to disable this.
        """

        if self._hooks_enabled:
            # hooks have already been added
            return self

        self._hooks_enabled = True

        def capture_activations(module: nn.Module, input: List[torch.Tensor], _output: torch.Tensor):
            """Save activations into module.activations in forward pass"""
            if not self._hooks_enabled:
                return
            if not hasattr(module, 'activations'):
                module.activations = []
            # always take the first argument of the input
            # NOTE: clone is required to prevent in-place-overwrite of stored activations
            module.activations.append(input[0].detach().clone())

        def capture_backprops(module: nn.Module, _input, output: List[torch.Tensor]):
            """Save backprops into module.backprops in backward pass"""
            if not self._hooks_enabled:
                return
            if not hasattr(module, 'backprops'):
                module.backprops = []
            # always take the first output's backprop
            # NOTE: clone is required to prevent in-place-overwrite of stored backprops
            module.backprops.append(output[0].detach().clone())

        self.model.autograd_hooks = []
        for module in self.model.modules():
            if next(module.parameters(recurse=False), None) is not None:
                self.model.autograd_hooks.extend([
                    module.register_forward_hook(capture_activations),
                    module.register_backward_hook(capture_backprops)
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

        if not hasattr(self.model, 'autograd_hooks'):
            print("Warning, asked to remove hooks, but no hooks found")
        else:
            for handle in self.model.autograd_hooks:
                handle.remove()
            del self.model.autograd_hooks

        # The _hooks_enabled flag is a secondary fallback if hooks aren't removed
        self._hooks_enabled = False

    def __enter__(self):
        return self.hook()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.unhook()

    def privatize_grad(self, *, modules=None, clipping_norm=1., reduction: str = 'mean') -> None:
        """
        Compute per-example gradients for each parameter, privatize them, and save them under 'param.grad'.
        Must be called after loss.backprop()
        Must be called before optimizer.step()

        :param modules:
        :param clipping_norm:
        :param reduction:  either "mean" or "sum" depending whether backpropped loss was averaged or summed over batch
        """
        assert reduction in ('sum', 'mean')

        if self._disable:
            return

        self.steps += 1

        modules = [m for m in modules or self.model.modules() if self._has_params(m)]

        # check for existence of activations and backprops
        for module in modules:
            assert hasattr(module, 'activations'), "No activations detected, run forward after .hook()"
            assert hasattr(module, 'backprops'), "No backprops detected, run backward after .hook()"

            if not module.backprops or not module.activations:
                return

        # retrieve batch size
        if modules:
            batch_size = modules[0].activations[0].shape[0]
        else:
            return

        # preprocess activations and backprops
        for module in modules:
            # ignore leading activations from evaluations outside of the training loop
            module.activations = module.activations[-len(module.backprops):]
            # backprops are in reverse-order
            module.backprops = module.backprops[::-1]

            if reduction == 'mean':
                for backprop in module.backprops:
                    backprop *= batch_size

        # compute L2 norm for flat clipping
        actual_norm = self._calculate_norm(modules, batch_size)

        # reconstruct instance-level gradients and reduce privately
        for module in modules:
            # pair forward activations with reversed backpropagations
            for A, B in zip(module.activations, module.backprops):

                if isinstance(module, nn.Embedding):
                    A = A.unsqueeze(-1).expand(*A.shape, module.embedding_dim)
                    shape = batch_size, -1, module.embedding_dim

                    # massive... empty... tensor, because clip doesn't distribute
                    grad_instance = torch.zeros([batch_size, *module.weight.shape])
                    grad_instance.scatter_add_(1, A.reshape(*shape), B.reshape(*shape))

                    self._accumulate_grad(module.weight, grad_instance)

                    # # reconstructs exact grad
                    # grad = torch.zeros_like(module.weight.grad)
                    # grad.index_add_(0, A.reshape(-1), B.reshape(-1, module.embedding_dim))
                    # self._accumulate_grad(module.weight, grad)

                elif isinstance(module, nn.Linear):
                    grad_instance = torch.einsum('n...i,n...j->n...ij', B, A)
                    # if isinstance(module, TAG2Linear):
                    #     print(torch.sum(grad_instance, dim=0)[0])
                    self._accumulate_grad(module.weight, torch.einsum('n...ij->nij', grad_instance))
                    if module.bias is not None:
                        self._accumulate_grad(module.bias, torch.einsum('n...i->ni', B))

                elif isinstance(module, nn.Conv2d):
                    # TODO: testing
                    self._accumulate_grad(module.weight, module.weight.grad_instance)
                    if module.bias is not None:
                        self._accumulate_grad(module.bias, torch.sum(B.reshape(batch_size, -1, A.shape[-1]), dim=2))

                elif isinstance(module, CatBias):
                    # grab the last column of the backprop for the layer, which corresponds to the cat'ed column
                    self._accumulate_grad(module.bias, B[:, -1])

                elif isinstance(module, BahdanauAttentionScale):
                    v_grad_instance = torch.einsum('n...i->ni', B * A)

                    if module.normalize:
                        g_grad_instance = torch.einsum('n...->n', B * A * module.v) / torch.norm(module.v)
                        self._accumulate_grad(module.g, g_grad_instance.unsqueeze(-1))
                        v_grad_instance *= module.g / torch.norm(module.v)

                    self._accumulate_grad(module.v, v_grad_instance)

                else:
                    raise NotImplementedError(f"Gradient reconstruction is not implemented for {module}")

            for param in module.parameters(recurse=False):
                if CHECK_CORRECTNESS:
                    print('checking:', module, param.shape)
                    self._check_grad(param, reduction)

                param.grad = self._privatize_grad(param.grad_instance, reduction, actual_norm, clipping_norm)
                del param.grad_instance
            del module.activations
            del module.backprops

    @staticmethod
    def _calculate_norm(modules, batch_size):
        instance_sum_squares = []
        for module in modules:
            for A, B in zip(module.activations, module.backprops):
                assert A.shape[0] == batch_size

                if isinstance(module, nn.Embedding):
                    instance_sum_squares.append((B ** 2).reshape(batch_size, -1).sum(dim=1))

                elif isinstance(module, nn.Linear):
                    # sum((AB)^2) == sum(A^2 * B^2),
                    #   where * is the dot product along final axis,
                    #   and sum preserves the first axis
                    squares = torch.einsum('n...i,n...i,n...j,n...j->n...', A, A, B, B)
                    instance_sum_squares.append(squares.reshape(batch_size, -1).sum(dim=1))

                    if module.bias is not None:
                        instance_sum_squares.append((B ** 2).reshape(batch_size, -1).sum(dim=1))

                elif isinstance(module, CatBias):
                    instance_sum_squares.append((B[:, -1, None] ** 2).sum(dim=1))

                elif isinstance(module, nn.Conv2d):
                    # TODO: testing
                    batch_size = A.shape[0]
                    A = nn.functional.unfold(A, module.kernel_size)
                    B = B.reshape(batch_size, -1, A.shape[-1])
                    grad_instance = torch.einsum('ijk,ilk->ijl', B, A) \
                        .reshape([batch_size] + list(module.weight.shape))

                    setattr(module.weight, 'grad_instance', grad_instance)
                    instance_sum_squares.append((grad_instance ** 2).reshape(batch_size, -1).sum(dim=1))

                    if module.bias is not None:
                        instance_sum_squares.append((B ** 2).reshape(batch_size, -1).sum(dim=1))

                elif isinstance(module, BahdanauAttentionScale):
                    NotImplementedError("Bahdanau!")

                else:
                    raise NotImplementedError(f"Norm calculation is not implemented for {module}")

        return torch.sqrt(torch.stack(instance_sum_squares, dim=1).sum(dim=1))

    @staticmethod
    def _accumulate_grad(tensor, grad):
        if hasattr(tensor, 'grad_instance'):
            tensor.grad_instance += grad.detach()
        else:
            tensor.grad_instance = grad.detach()

    @staticmethod
    def _check_grad(param, reduction):
        grad = PrivacyAccountant._reduce_grad(param.grad_instance, reduction)
        if not torch.equal(torch.Tensor(list(param.grad.shape)), torch.Tensor(list(grad.shape))):
            raise ValueError(f"Non-private reconstructed gradient {grad.shape} differs from expected shape {param.grad.shape}")
        if not torch.allclose(param.grad, grad, atol=.01, equal_nan=True):
            print('          failed')
            print('          difference:')
            print(param.grad - grad)
            print('          expected:')
            print(param.grad)
            print('          reconstructed:')
            print(grad)
            raise ValueError("Non-private reconstructed gradient differs in value")

    def _privatize_grad(self, grad_instance, reduction, actual_norm, clipping_norm):
        """

        :param grad_instance:
        :param reduction:
        :param actual_norm:
        :param clipping_norm:
        :return:
        """

        # clip
        PrivacyAccountant._clip_grad_(grad_instance, actual_norm, clipping_norm)
        # reduce
        grad = PrivacyAccountant._reduce_grad(grad_instance, reduction)
        # noise
        self._noise_grad_(grad, clipping_norm, reduction, grad_instance.shape[0])

        return grad

    @staticmethod
    def _clip_grad_(grad_instance, actual_norm, clipping_norm):
        singletons = (1,) * (grad_instance.ndim - 1)
        grad_instance /= torch.max(torch.ones_like(actual_norm), actual_norm / clipping_norm) \
            .reshape(-1, *singletons) \
            .expand_as(grad_instance)

    @staticmethod
    def _reduce_grad(grad_instance, reduction):
        return {'sum': torch.sum, 'mean': torch.mean}[reduction](grad_instance, dim=0)

    def _noise_grad_(self, grad, clipping_norm, reduction, n):
        sensitivity = clipping_norm / {'sum': 1., 'mean': n}[reduction]
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

    @staticmethod
    def _has_params(module):
        return next(module.parameters(recurse=False), None) is not None

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
        self.increment_epoch()

        epsilon = 0
        delta = 0

        for batch_len in self._epochs:
            if suggested_delta is None:
                batch_delta = 2 * math.exp(-batch_len / 16 * math.exp(-self.step_epsilon)) + 1E-8
            else:
                batch_delta = suggested_delta / len(self._epochs)

            batch_epsilon, batch_delta = core_library.shuffle_amplification(
                step_epsilon=self.step_epsilon,
                step_delta=self.step_delta,
                delta=batch_delta,
                steps=batch_len)

            epsilon += batch_epsilon
            delta += batch_delta

        return epsilon, delta
