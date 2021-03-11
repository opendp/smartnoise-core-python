"""
Tools for building differentially private models.
Thanks to https://github.com/cybertronai/autograd-hacks for demonstrating gradient hacks.

A, activations: inputs into current module
B, backprops: backprop values (aka Jacobian-vector product) observed at current module

"""
import copy
import math
from typing import List

import torch
import torch.nn as nn
from torch.nn.parameter import Parameter

from opendp.smartnoise.core.api import LibraryWrapper
from opendp.smartnoise.network.layers.base import InstanceGrad

from opendp.smartnoise.network.layers.bahdanau import DPBahdanauAttention
from opendp.smartnoise.network.layers.lstm import DPLSTM, DPLSTMCell

core_library = LibraryWrapper()

CHECK_CORRECTNESS = False

REPLACEMENT_MODULES = {
    nn.LSTM: DPLSTM,
    nn.LSTMCell: DPLSTMCell,
    'BahdanauAttention': DPBahdanauAttention
}


class _SharedParameter(Parameter):
    @classmethod
    def mark(cls, parameter):
        assert isinstance(parameter, Parameter)
        parameter.__class__ = cls

    def unmark(self):
        self.__class__ = Parameter

    def __copy__(self):
        return self

    def __deepcopy__(self, memodict={}):
        return self


class PrivacyAccountant(object):
    def __init__(
            self,
            model: nn.Module,
            step_epsilon, step_delta=0., clipping_norm=1.,
            modules=None,
            hook=True,
            disable=False,
            reduction='mean',
            replacement_modules=None):
        """
        Utility for tracking privacy usage
        :param model: pyTorch model
        :param step_epsilon:
        :param step_delta:
        :param hook: whether to call hook() on __init__
        """

        # copy network architecture, but share parameters
        for param in model.parameters():
            _SharedParameter.mark(param)
        self.model = copy.deepcopy(model)
        for param in model.parameters():
            param.unmark()

        replacement_modules = {**REPLACEMENT_MODULES, **(replacement_modules or {})}

        # restructure the copied network architecture in-place, without breaking references to original parameters
        self._replace_modules(self.model, replacement_modules)

        self._hooks_enabled = False  # work-around for https://github.com/pytorch/pytorch/issues/25723
        self.step_epsilon = step_epsilon
        self.step_delta = step_delta
        self._epochs = []
        self.steps = 0
        self._disable = disable

        assert reduction in ('sum', 'mean')
        self.reduction = reduction

        self.modules = modules
        self.clipping_norm = clipping_norm

        if not disable and hook:
            self.hook()

    @staticmethod
    def _replace_modules(module, replacement_modules):
        """
        replaces modules with DP-capable versions of modules throughout a network
        """

        for attr_str in dir(module):
            target_attr = getattr(module, attr_str)
            # ignore anything that isn't a module
            if not issubclass(type(target_attr), nn.Module):
                continue

            replacement_module = replacement_modules.get(type(target_attr))
            if not replacement_module:
                replacement_module = replacement_modules.get(target_attr.__class__.__name__)
            if replacement_module:
                replacement_attr = replacement_module.replace(target_attr)
                setattr(module, attr_str, replacement_attr)

        # recurse down child modules
        for name, child_module in module.named_children():
            PrivacyAccountant._replace_modules(child_module, replacement_modules)

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

        modules = [m for m in self.modules or self.model.modules() if self._has_params(m)]

        def capture_activations(module: nn.Module, input: List[torch.Tensor], _output):
            """Save activations into module.activations in forward pass"""
            if not self._hooks_enabled:
                return
            if not hasattr(module, 'activations'):
                module.activations = []
            # NOTE: clone is required to prevent in-place-overwrite of stored activations
            module.activations.append(tuple(in_arg.detach().clone() for in_arg in input))

        def capture_backprops(module: nn.Module, _input: List[torch.Tensor], output: List[torch.Tensor]):
            """Save backprops into module.backprops in backward pass"""
            if not self._hooks_enabled:
                return
            if not hasattr(module, 'backprops'):
                module.backprops = []
            # NOTE: clone is required to prevent in-place-overwrite of stored backprops
            module.backprops.append(tuple(out_arg.detach().clone() for out_arg in output))

        def get_batch_size(module):
            # first activation, first arg, first axis shape
            return module.activations[0][0].shape[0]

        def make_privatization_hook(module, param, instance_grad_generator):
            def privatization_hook(grad):

                # ignore leading activations from evaluations outside of the training loop
                module.activations = module.activations[-len(module.backprops):]

                if self.reduction == 'mean':
                    for backprop in module.backprops:
                        backprop *= get_batch_size(module)

                # backprops are in reverse-order
                for A, B in zip(module.activations, module.backprops[::-1]):
                    for chunk in instance_grad_generator(A, B):
                        InstanceGrad._accumulate_instance_grad(param, chunk)

                if CHECK_CORRECTNESS:
                    print('checking:', module, param.shape)
                    self._check_grad(grad, param.grad_instance, self.reduction)

                actual_norm = torch.norm(
                    param.grad_instance.reshape(param.grad_instance.shape[0], -1) ** 2,
                    dim=1)

                private_grad = self._privatize_grad(
                    param.grad_instance, self.reduction,
                    actual_norm, self.clipping_norm)

                del param.grad_instance
                param.is_grad_dp = True
                # clear module hook data once all param grads are dp
                if all(hasattr(par, 'is_grad_dp') and par.is_grad_dp for par in module.parameters(recurse=False)):
                    del module.activations
                    del module.backprops
                    for par in module.parameters(recurse=False):
                        del par.is_grad_dp

                return private_grad
            return privatization_hook

        self.model.autograd_hooks = []
        for module in modules:
            # ignore the module if it has no parameters
            if next(module.parameters(recurse=False), None) is None:
                continue

            # register global hooks
            self.model.autograd_hooks.extend([
                module.register_forward_hook(capture_activations),
                module.register_backward_hook(capture_backprops)
            ])

            if isinstance(module, InstanceGrad):
                # Dict[Parameter, Callable[[A, B], G]], where A is activations, B is backprops, G is instance grad
                # A: tuple of activations, one for each argument
                # B: tuple of backprops, one for each output from the network. Implicitly upgraded to a singleton
                # G: instance gradient of shape- n x (*param.shape)
                instance_grads = module.get_instance_grad_functions()

            elif isinstance(module, nn.Embedding):
                def make_embedding_grad_generator(module):
                    def embedding_grad_generator(A, B):
                        # only take the first argument to embedding forward
                        A, B = A[0], B[0]
                        batch_size = A.shape[0]
                        A = A.unsqueeze(-1).expand(*A.shape, module.embedding_dim)
                        shape = batch_size, -1, module.embedding_dim

                        # massive... empty... tensor, because clip doesn't distribute
                        grad_instance = torch.zeros([batch_size, *module.weight.shape])
                        grad_instance.scatter_add_(1, A.reshape(*shape), B.reshape(*shape))
                        yield grad_instance
                    return embedding_grad_generator

                instance_grads = {module.weight: make_embedding_grad_generator(module)}

                # # reconstructs exact grad
                # grad = torch.zeros_like(module.weight.grad)
                # grad.index_add_(0, A.reshape(-1), B.reshape(-1, module.embedding_dim))
                # self._accumulate_grad(module.weight, grad)

            elif isinstance(module, nn.Linear):
                def make_weight_grad_generator(_module):
                    def weight_grad_generator(A, B):
                        # linear is unary
                        A, B = A[0], B[0]

                        if len(A.shape) > 2:
                            for A, B in zip(torch.chunk(A, chunks=10, dim=1), torch.chunk(B, chunks=10, dim=1)):
                                grad_instance = torch.einsum('n...i,n...j->n...ij', B, A)
                                yield torch.einsum('n...ij->nij', grad_instance)
                        else:
                            grad_instance = torch.einsum('n...i,n...j->n...ij', B, A)
                            yield torch.einsum('n...ij->nij', grad_instance)
                    return weight_grad_generator

                def make_bias_grad_generator(module):
                    def bias_grad_generator(A, B):
                        A, B = A[0], B[0]
                        if module.bias is None:
                            return

                        if len(A.shape) > 2:
                            for A, B in zip(torch.chunk(A, chunks=10, dim=1), torch.chunk(B, chunks=10, dim=1)):
                                yield torch.einsum('n...i->ni', B)
                        else:
                            yield torch.einsum('n...i->ni', B)
                    return bias_grad_generator

                instance_grads = {
                    module.weight: make_weight_grad_generator(module),
                    module.bias: make_bias_grad_generator(module),
                }

            else:
                raise NotImplementedError(f"Gradient reconstruction is not implemented for {module}")

            for param in instance_grads:
                privatization_hook = make_privatization_hook(module, param, instance_grads[param])
                self.model.autograd_hooks.append(param.register_hook(privatization_hook))

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
        self.hook()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.unhook()

    @staticmethod
    def _check_grad(grad, instance_grad, reduction):
        grad_2 = PrivacyAccountant._reduce_grad(instance_grad, reduction)
        if not torch.equal(torch.Tensor(list(grad.shape)), torch.Tensor(list(grad_2.shape))):
            raise ValueError(f"Non-private reconstructed gradient {grad_2.shape} differs from expected shape {grad.shape}")
        if not torch.allclose(grad, grad_2, atol=.01, equal_nan=True):
            print('          failed')
            print('          difference:')
            print(grad - grad_2)
            print('          expected:')
            print(grad)
            print('          reconstructed:')
            print(grad_2)
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
        grad = self._noise_grad(grad, clipping_norm, reduction, grad_instance.shape[0])

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

    def _noise_grad(self, grad, clipping_norm, reduction, n):
        sensitivity = clipping_norm / {'sum': 1., 'mean': n}[reduction]
        device = grad.device
        if device != 'cpu':
            grad = grad.to('cpu')

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
        if device != 'cpu':
            grad = grad.to(device)
        return grad

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
