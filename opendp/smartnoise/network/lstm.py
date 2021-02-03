from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch.nn.utils.rnn import PackedSequence


class LSTMCell(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, layer_norm: bool = False):
        super().__init__()

        self.hidden_lin = nn.Linear(hidden_size, 4 * hidden_size)
        self.input_lin = nn.Linear(input_size, 4 * hidden_size, bias=False)

        # TODO: check layer norm for DP-ness
        if layer_norm:
            self.layer_norm = nn.ModuleList([nn.LayerNorm(hidden_size) for _ in range(4)])
            self.layer_norm_c = nn.LayerNorm(hidden_size)
        else:
            self.layer_norm = nn.ModuleList([nn.Identity() for _ in range(4)])
            self.layer_norm_c = nn.Identity()

    def forward(self, x: torch.Tensor, h: torch.Tensor, c: torch.Tensor):
        ifgo = self.hidden_lin(h) + self.input_lin(x)
        ifgo = ifgo.chunk(4, dim=-1)
        i, f, g, o = [self.layer_norm[i](ifgo[i]) for i in range(4)]
        c_next = torch.sigmoid(f) * c + torch.sigmoid(i) * torch.tanh(g)
        h_next = torch.sigmoid(o) * torch.tanh(self.layer_norm_c(c_next))
        return h_next, c_next


class DPLSTM(nn.LSTM):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.cells = nn.ModuleList(
            [LSTMCell(self.input_size, self.hidden_size)] +
            [LSTMCell(self.hidden_size, self.hidden_size) for _ in range(self.num_layers - 1)])

    def forward(self, input, hx: Optional[Tuple[torch.Tensor, torch.Tensor]] = None):
        orig_input = input

        # xxx: isinstance check needs to be in conditional for TorchScript to compile
        if isinstance(orig_input, PackedSequence):
            input, batch_sizes, sorted_indices, unsorted_indices = input
            max_batch_size = batch_sizes[0]
            max_batch_size = int(max_batch_size)
        else:
            batch_sizes = None
            max_batch_size = input.size(0 if self.batch_first else 1)
            sorted_indices = None
            unsorted_indices = None

        if hx is None:
            num_directions = 2 if self.bidirectional else 1
            real_hidden_size = self.proj_size if self.proj_size > 0 else self.hidden_size
            h_zeros = torch.zeros(self.num_layers * num_directions,
                                  max_batch_size, real_hidden_size,
                                  dtype=input.dtype, device=input.device)
            c_zeros = torch.zeros(self.num_layers * num_directions,
                                  max_batch_size, self.hidden_size,
                                  dtype=input.dtype, device=input.device)
            hx = (h_zeros, c_zeros)
        else:
            # Each batch of the hidden state should match the input sequence that
            # the user believes he/she is passing in.
            hx = self.permute_hidden(hx, sorted_indices)

        h, c = list(torch.unbind(hx[0])), list(torch.unbind(hx[1]))

        self.check_forward_args(input, hx, batch_sizes)

        out = []
        for inp in input:
            for layer in range(self.num_layers):
                h[layer], c[layer] = self.cells[layer](inp, h[layer], c[layer])
                inp = h[layer]
            out.append(h[-1])

        out = torch.stack(out)

        # xxx: isinstance check needs to be in conditional for TorchScript to compile
        if isinstance(orig_input, PackedSequence):
            output_packed = PackedSequence(out, batch_sizes, sorted_indices, unsorted_indices)
            return output_packed, self.permute_hidden(hx, unsorted_indices)
        else:
            return out, self.permute_hidden(hx, unsorted_indices)
