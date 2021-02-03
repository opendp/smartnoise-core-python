import warnings
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.linear import _LinearWithBias
from torch.tensor import Tensor


# class PrepareForMultiHeadAttention(nn.Module):
#     """
#     This module does a linear transformation and splits the vector into given number of heads for multi-head attention.
#     This is used to transform key, query, and value vectors.
#     """
#
#     def __init__(self, d_model: int, heads: int, d_k: int, bias: bool):
#         super().__init__()
#         self.linear = nn.Linear(d_model, heads * d_k, bias=bias)
#         self.heads = heads
#
#         # Number of dimensions in vectors in each head
#         self.d_k = d_k
#
#     def __call__(self, x: torch.Tensor):
#         """
#         Input has shape [seq_len, batch_size, d_model] or [batch_size, d_model].
#         We apply the linear transformation of the last dimension and splits that into the heads
#         :param x:
#         :return:
#         """
#
#         head_shape = x.shape[:-1]
#
#         x = self.linear(x)
#
#         # Split last dimension into heads
#         x = x.view(*head_shape, self.heads, self.d_k)
#
#         # Output has shape [seq_len, batch_size, heads, d_k] or [batch_size, d_model]
#         return x


class DPMultiHeadAttention(nn.MultiheadAttention):
    """
    This computes scaled multi-headed attention for given query, key and value vectors.
    Attention(Q,K,V)=softmax_seq(Q @ (K^T) / sqrt(d_k)) @ V
    In simple terms, it finds keys that matches the query, and get the values of those keys.
    It uses dot-product of query and key as the indicator of how matching they are.
    Before taking the softmax the dot-products are scaled by 1 / sqrt(d_k).
    This is done to avoid large dot-product values causing softmax to give very small gradients when d_k is large.
    Softmax is calculated along the axis of of the sequence (or time).
    """

    def __init__(self,
                 embed_dim: int,
                 num_heads: int,
                 dropout: float = 0.0,
                 bias: bool = True,
                 add_bias_kv: bool = False,
                 add_zero_attn: bool = False,
                 kdim: bool = None,
                 vdim: bool = None):
        """

        :param embed_dim: number of features in the query, key and value vectors
        :param num_heads: number of heads
        :param dropout:
        :param bias:
        """
        super(nn.Module, self).__init__()

        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self._qkv_same_embed_dim = self.kdim == embed_dim and self.vdim == embed_dim

        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"

        if self._qkv_same_embed_dim is False:

            self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
            self.k_proj = nn.Linear(embed_dim, self.kdim, bias=bias)
            self.v_proj = nn.Linear(embed_dim, self.vdim, bias=bias)
            self.register_parameter('in_proj', None)
        else:
            self.in_proj = nn.Linear(3 * embed_dim, embed_dim, bias=bias)
            self.register_parameter('q_proj', None)
            self.register_parameter('k_proj', None)
            self.register_parameter('v_proj', None)

        self.out_proj = _LinearWithBias(embed_dim, embed_dim)

        if add_bias_kv:
            self.bias_k = Parameter(torch.empty(1, 1, embed_dim))
            self.bias_v = Parameter(torch.empty(1, 1, embed_dim))
        else:
            self.bias_k = self.bias_v = None

        self.add_zero_attn = add_zero_attn

        self._reset_parameters()

    #
    # @staticmethod
    # def get_scores(query: torch.Tensor, key: torch.Tensor):
    #     """
    #     Calculate scores between queries and keys
    #     this method can be overridden for other variations like relative attention.
    #     :param query:
    #     :param key:
    #     :return:
    #     """
    #     # Calculate Q @ K^T
    #     return torch.einsum('ibhd,jbhd->ijbh', query, key)  # S_ijbh = ∑ Q_ibhd * K_jbhd
    #
    # def __call__(self, *,
    #              query: torch.Tensor,
    #              key: torch.Tensor,
    #              value: torch.Tensor,
    #              mask: Optional[torch.Tensor] = None):
    #     """
    #     query, key and value are the tensors that store collection of query, key and value vectors.
    #     They have shape [seq_len, batch_size, d_model].
    #     mask has shape [seq_len, seq_len, batch_size]
    #     mask[i, j, b] indicates whether for batch b, query at position i has access to key-value at position j.
    #     :param query:
    #     :param key:
    #     :param value:
    #     :param mask:
    #     :return:
    #     """
    #     # query, key and value have shape [seq_len, batch_size, d_model]
    #     seq_len, batch_size, _ = query.shape
    #
    #     if mask is not None:
    #         # mask has shape [seq_len, seq_len, batch_size] where first dimension is the query dimension.
    #         # If the query dimension is equal to x it will be broadcasted
    #         assert mask.shape[0] == 1 or mask.shape[0] == mask.shape[1]
    #
    #         # Same mask applied to all heads.
    #         mask = mask.unsqueeze(-1)
    #
    #     # Prepare query, key and value for attention computation
    #     # These will then have shape [seq_len, batch_size, heads, d_k]
    #     query = self.query(query)
    #     key = self.key(key)
    #     value = self.value(value)
    #
    #     # Compute attention scores Q @ (K^T)
    #     # Results in a tensor of shape [seq_len, seq_len, batch_size, heads]
    #     scores = self.get_scores(query, key)
    #
    #     # Scale scores Q @ (K^T) / sqrt(d_k)
    #     scores *= self.scale
    #
    #     # Apply mask
    #     if mask is not None:
    #         scores = scores.masked_fill(mask == 0, -1e9)
    #
    #     # attention along the key sequence dimension
    #     attn = self.softmax(scores)  # softmax_seq(Q @ (K^T) / sqrt(d_k))
    #
    #     # Apply dropout
    #     attn = self.dropout(attn)
    #
    #     # Multiply by values
    #     x = torch.einsum("ijbh,jbhd->ibhd", attn, value)  # softmax_seq(Q @ K^T / sqrt(d_k)) @ V
    #
    #     # Concatenate multiple heads
    #     x = x.reshape(seq_len, batch_size, -1)
    #
    #     # Output layer
    #     return self.output(x)

    def forward(self, query, key, value, key_padding_mask=None,
                need_weights=True, attn_mask=None) -> Tuple[Tensor, Optional[Tensor]]:
        r"""
        Args:
            query, key, value: map a query and a set of key-value pairs to an output.
                See "Attention Is All You Need" for more details.
            embed_dim_to_check: total dimension of the model.
            num_heads: parallel attention heads.
            in_proj_weight, in_proj_bias: input projection weight and bias.
            bias_k, bias_v: bias of the key and value sequences to be added at dim=0.
            add_zero_attn: add a new batch of zeros to the key and
                           value sequences at dim=1.
            dropout_p: probability of an element to be zeroed.
            out_proj_weight, out_proj_bias: the output projection weight and bias.
            training: apply dropout if is ``True``.
            key_padding_mask: if provided, specified padding elements in the key will
                be ignored by the attention. This is an binary mask. When the value is True,
                the corresponding value on the attention layer will be filled with -inf.
            need_weights: output attn_output_weights.
            attn_mask: 2D or 3D mask that prevents attention to certain positions. A 2D mask will be broadcasted for all
                the batches while a 3D mask allows to specify a different mask for the entries of each batch.
            use_separate_proj_weight: the function accept the proj. weights for query, key,
                and value in different forms. If false, in_proj_weight will be used, which is
                a combination of q_proj_weight, k_proj_weight, v_proj_weight.
            q_proj_weight, k_proj_weight, v_proj_weight, in_proj_bias: input projection weight and bias.
            static_k, static_v: static key and value used for attention operators.


        Shape:
            Inputs:
            - query: :math:`(L, N, E)` where L is the target sequence length, N is the batch size, E is
              the embedding dimension.
            - key: :math:`(S, N, E)`, where S is the source sequence length, N is the batch size, E is
              the embedding dimension.
            - value: :math:`(S, N, E)` where S is the source sequence length, N is the batch size, E is
              the embedding dimension.
            - key_padding_mask: :math:`(N, S)` where N is the batch size, S is the source sequence length.
              If a ByteTensor is provided, the non-zero positions will be ignored while the zero positions
              will be unchanged. If a BoolTensor is provided, the positions with the
              value of ``True`` will be ignored while the position with the value of ``False`` will be unchanged.
            - attn_mask: 2D mask :math:`(L, S)` where L is the target sequence length, S is the source sequence length.
              3D mask :math:`(N*num_heads, L, S)` where N is the batch size, L is the target sequence length,
              S is the source sequence length. attn_mask ensures that position i is allowed to attend the unmasked
              positions. If a ByteTensor is provided, the non-zero positions are not allowed to attend
              while the zero positions will be unchanged. If a BoolTensor is provided, positions with ``True``
              are not allowed to attend while ``False`` values will be unchanged. If a FloatTensor
              is provided, it will be added to the attention weight.
            - static_k: :math:`(N*num_heads, S, E/num_heads)`, where S is the source sequence length,
              N is the batch size, E is the embedding dimension. E/num_heads is the head dimension.
            - static_v: :math:`(N*num_heads, S, E/num_heads)`, where S is the source sequence length,
              N is the batch size, E is the embedding dimension. E/num_heads is the head dimension.

            Outputs:
            - attn_output: :math:`(L, N, E)` where L is the target sequence length, N is the batch size,
              E is the embedding dimension.
            - attn_output_weights: :math:`(N, L, S)` where N is the batch size,
              L is the target sequence length, S is the source sequence length.
        """

        embed_dim_to_check = self.embed_dim
        num_heads = self.num_heads
        bias_k = self.bias_k
        bias_v = self.bias_v
        add_zero_attn = self.add_zero_attn
        dropout_p = self.dropout
        training = self.training
        key_padding_mask = self.key_padding_mask
        static_k = None
        static_v = None

        if not torch.jit.is_scripting():
            raise NotImplementedError()
            # tens_ops = (query, key, value, in_proj_weight, in_proj_bias, bias_k, bias_v,
            #             out_proj_weight, out_proj_bias)
            # if any([type(t) is not Tensor for t in tens_ops]) and has_torch_function(tens_ops):
            #     return handle_torch_function(
            #         multi_head_attention_forward, tens_ops, query, key, value,
            #         embed_dim_to_check, num_heads, in_proj_weight, in_proj_bias,
            #         bias_k, bias_v, add_zero_attn, dropout_p, out_proj_weight,
            #         out_proj_bias, training=training, key_padding_mask=key_padding_mask,
            #         need_weights=need_weights, attn_mask=attn_mask,
            #         use_separate_proj_weight=use_separate_proj_weight,
            #         q_proj_weight=q_proj_weight, k_proj_weight=k_proj_weight,
            #         v_proj_weight=v_proj_weight, static_k=static_k, static_v=static_v)
        tgt_len, bsz, embed_dim = query.size()
        assert embed_dim == embed_dim_to_check
        # allow MHA to have different sizes for the feature dimension
        assert key.size(0) == value.size(0) and key.size(1) == value.size(1)

        head_dim = embed_dim // num_heads
        assert head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"
        scaling = float(head_dim) ** -0.5

        if not self._qkv_same_embed_dim:
            if torch.equal(query, key) and torch.equal(key, value):
                # self-attention
                q, k, v = self.in_proj(query).chunk(3, dim=-1)

            elif torch.equal(key, value):
                raise NotImplementedError()
                # encoder-decoder attention
                # This is inline in_proj function with in_proj_weight and in_proj_bias
                # _b = in_proj_bias
                # _start = 0
                # _end = embed_dim
                # _w = in_proj_weight[_start:_end, :]
                # if _b is not None:
                #     _b = _b[_start:_end]
                # q = linear(query, _w, _b)
                #
                # if key is None:
                #     assert value is None
                #     k = None
                #     v = None
                # else:
                #
                #     # This is inline in_proj function with in_proj_weight and in_proj_bias
                #     _b = in_proj_bias
                #     _start = embed_dim
                #     _end = None
                #     _w = in_proj_weight[_start:, :]
                #     if _b is not None:
                #         _b = _b[_start:]
                #     k, v = linear(key, _w, _b).chunk(2, dim=-1)

            else:
                raise NotImplementedError()
                # # This is inline in_proj function with in_proj_weight and in_proj_bias
                # _b = in_proj_bias
                # _start = 0
                # _end = embed_dim
                # _w = in_proj_weight[_start:_end, :]
                # if _b is not None:
                #     _b = _b[_start:_end]
                # q = linear(query, _w, _b)
                #
                # # This is inline in_proj function with in_proj_weight and in_proj_bias
                # _b = in_proj_bias
                # _start = embed_dim
                # _end = embed_dim * 2
                # _w = in_proj_weight[_start:_end, :]
                # if _b is not None:
                #     _b = _b[_start:_end]
                # k = linear(key, _w, _b)
                #
                # # This is inline in_proj function with in_proj_weight and in_proj_bias
                # _b = in_proj_bias
                # _start = embed_dim * 2
                # _end = None
                # _w = in_proj_weight[_start:, :]
                # if _b is not None:
                #     _b = _b[_start:]
                # v = linear(value, _w, _b)
        else:
            # q_proj_weight_non_opt = torch.jit._unwrap_optional(q_proj_weight)
            # len1, len2 = q_proj_weight_non_opt.size()
            # assert len1 == embed_dim and len2 == query.size(-1)
            #
            # k_proj_weight_non_opt = torch.jit._unwrap_optional(k_proj_weight)
            # len1, len2 = k_proj_weight_non_opt.size()
            # assert len1 == embed_dim and len2 == key.size(-1)
            #
            # v_proj_weight_non_opt = torch.jit._unwrap_optional(v_proj_weight)
            # len1, len2 = v_proj_weight_non_opt.size()
            # assert len1 == embed_dim and len2 == value.size(-1)
            #
            # if in_proj_bias is not None:
            #
            #     q = linear(query, q_proj_weight_non_opt, in_proj_bias[0:embed_dim])
            #     k = linear(key, k_proj_weight_non_opt, in_proj_bias[embed_dim:(embed_dim * 2)])
            #     v = linear(value, v_proj_weight_non_opt, in_proj_bias[(embed_dim * 2):])
            # else:
            #     q = linear(query, q_proj_weight_non_opt, in_proj_bias)
            #     k = linear(key, k_proj_weight_non_opt, in_proj_bias)
            #     v = linear(value, v_proj_weight_non_opt, in_proj_bias)

            q = self.q_proj(query)
            k = self.q_proj(key)
            v = self.q_proj(value)
        q = q * scaling

        if attn_mask is not None:
            assert attn_mask.dtype == torch.float32 or attn_mask.dtype == torch.float64 or \
                   attn_mask.dtype == torch.float16 or attn_mask.dtype == torch.uint8 or attn_mask.dtype == torch.bool, \
                'Only float, byte, and bool types are supported for attn_mask, not {}'.format(attn_mask.dtype)
            if attn_mask.dtype == torch.uint8:
                warnings.warn(
                    "Byte tensor for attn_mask in nn.MultiheadAttention is deprecated. Use bool tensor instead.")
                attn_mask = attn_mask.to(torch.bool)

            if attn_mask.dim() == 2:
                attn_mask = attn_mask.unsqueeze(0)
                if list(attn_mask.size()) != [1, query.size(0), key.size(0)]:
                    raise RuntimeError('The size of the 2D attn_mask is not correct.')
            elif attn_mask.dim() == 3:
                if list(attn_mask.size()) != [bsz * num_heads, query.size(0), key.size(0)]:
                    raise RuntimeError('The size of the 3D attn_mask is not correct.')
            else:
                raise RuntimeError("attn_mask's dimension {} is not supported".format(attn_mask.dim()))
            # attn_mask's dim is 3 now.

        # convert ByteTensor key_padding_mask to bool
        if key_padding_mask is not None and key_padding_mask.dtype == torch.uint8:
            warnings.warn(
                "Byte tensor for key_padding_mask in nn.MultiheadAttention is deprecated. Use bool tensor instead.")
            key_padding_mask = key_padding_mask.to(torch.bool)

        if bias_k is not None and bias_v is not None:
            if static_k is None and static_v is None:
                k = torch.cat([k, bias_k.repeat(1, bsz, 1)])
                v = torch.cat([v, bias_v.repeat(1, bsz, 1)])
                if attn_mask is not None:
                    attn_mask = F.pad(attn_mask, (0, 1))
                if key_padding_mask is not None:
                    key_padding_mask = F.pad(key_padding_mask, (0, 1))
            else:
                assert static_k is None, "bias cannot be added to static key."
                assert static_v is None, "bias cannot be added to static value."
        else:
            assert bias_k is None
            assert bias_v is None

        q = q.contiguous().view(tgt_len, bsz * num_heads, head_dim).transpose(0, 1)
        if k is not None:
            k = k.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)
        if v is not None:
            v = v.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)

        if static_k is not None:
            assert static_k.size(0) == bsz * num_heads
            assert static_k.size(2) == head_dim
            k = static_k

        if static_v is not None:
            assert static_v.size(0) == bsz * num_heads
            assert static_v.size(2) == head_dim
            v = static_v

        src_len = k.size(1)

        if key_padding_mask is not None:
            assert key_padding_mask.size(0) == bsz
            assert key_padding_mask.size(1) == src_len

        if add_zero_attn:
            src_len += 1
            k = torch.cat([k, torch.zeros((k.size(0), 1) + k.size()[2:], dtype=k.dtype, device=k.device)], dim=1)
            v = torch.cat([v, torch.zeros((v.size(0), 1) + v.size()[2:], dtype=v.dtype, device=v.device)], dim=1)
            if attn_mask is not None:
                attn_mask = F.pad(attn_mask, (0, 1))
            if key_padding_mask is not None:
                key_padding_mask = F.pad(key_padding_mask, (0, 1))

        attn_output_weights = torch.bmm(q, k.transpose(1, 2))
        assert list(attn_output_weights.size()) == [bsz * num_heads, tgt_len, src_len]

        if attn_mask is not None:
            if attn_mask.dtype == torch.bool:
                attn_output_weights.masked_fill_(attn_mask, float('-inf'))
            else:
                attn_output_weights += attn_mask

        if key_padding_mask is not None:
            attn_output_weights = attn_output_weights.view(bsz, num_heads, tgt_len, src_len)
            attn_output_weights = attn_output_weights.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2),
                float('-inf'),
            )
            attn_output_weights = attn_output_weights.view(bsz * num_heads, tgt_len, src_len)

        attn_output_weights = F.softmax(
            attn_output_weights, dim=-1)
        attn_output_weights = F.dropout(attn_output_weights, p=dropout_p, training=training)

        attn_output = torch.bmm(attn_output_weights, v)
        assert list(attn_output.size()) == [bsz * num_heads, tgt_len, head_dim]
        attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
        attn_output = self.out_proj(attn_output)

        if need_weights:
            # average attention weights over heads
            attn_output_weights = attn_output_weights.view(bsz, num_heads, tgt_len, src_len)
            return attn_output, attn_output_weights.sum(dim=1) / num_heads
        else:
            return attn_output, None
