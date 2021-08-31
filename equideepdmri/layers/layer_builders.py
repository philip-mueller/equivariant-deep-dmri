from collections import OrderedDict
from functools import partial
from typing import Union, List

import torch
from torch import nn
from e3nn.non_linearities.rescaled_act import sigmoid
from e3nn.non_linearities.gated_block import GatedBlock
from e3nn.batchnorm import BatchNorm as E3NNBatchNorm

from equideepdmri.layers.EquivariantPQLayer import EquivariantPQLayer
from equideepdmri.layers.QLengthWeightedPool import QLengthWeightedAvgPool
from equideepdmri.utils.q_space import Q_SamplingSchema
from equideepdmri.utils.spherical_tensor import SphericalTensorType
from equideepdmri.layers.filter.utils import get_scalar_non_linearity


def build_pq_layer(type_in: Union[SphericalTensorType, List[int]],
                   type_out: Union[SphericalTensorType, List[int]],
                   p_kernel_size: int,
                   kernel: str,
                   q_sampling_schema_in: Union[Q_SamplingSchema, torch.Tensor, List, None],
                   q_sampling_schema_out: Union[Q_SamplingSchema, torch.Tensor, List, None],
                   non_linearity_config=None,
                   use_non_linearity=True,
                   batch_norm_config=None,
                   use_batch_norm=True,
                   transposed=False,
                   auto_recompute=True,
                   **kernel_kwargs) -> nn.Module:
    """
    Builds a pq-layer consisting of an EquivariantPQLayer followed by a nonlinearity (e.g. gated nonlinearity).

    :param type_in: The spherical tensor type of the input feature map.
        This defines how many channels of each tensor order the input feature map has.
        It can either be given as SphericalTensorType object or as List[int]] the element at index i of the list
        defines the number of order-i channels,
        e.g. the first element defines the number of order-0 (scalar) channels
        and the second the number of order-1 (vector) channels and so on.
        For all orders corresponding to out-of-range indices the number of channels is 0.
    :param type_out: The spherical tensor type of the output feature map (after non-linearity).
        This defines how many channels of each tensor order the output feature map has.
        It can either be given as SphericalTensorType object or as List[int]] the element at index i of the list
        defines the number of order-i channels,
        e.g. the first element defines the number of order-0 (scalar) channels
        and the second the number of order-1 (vector) channels and so on.
        For all orders corresponding to out-of-range indices the number of channels is 0.
    :param p_kernel_size: Size of the kernel in p-space.
        Note that the kernel always covers the whole q-space (as it is not translationally equivariant),
        so there is no q_kernel_size.
    :param kernel: Which filter basis to use in the EquivariantPQLayer layer.
        Valid options are:

        - "p_space": to use the p-space filter basis
          using only p-space coordinate offsets in the angular and radial part.
        - "q_space": to use the q-space filter basis
          using only q-space coordinate offsets in the angular part
          and q-space coordinates from input and output in the radial part.
        - "pq_diff": to use the pq-diff filter basis
          using difference between p- and q-space coordinate offsets in the angular part
          and p-space coordinate offsets, q-space coordinates from input and output in the radial part.
        - "pq_TP": to use the TP (tensor product) filter basis
          using the tensor product of the p- and q-space filters in the angular part
          and p-space coordinate offsets, q-space coordinates from input and output in the radial part.
        - "sum(<filters>)": where <filters> is a ";"-separated list (without spaces) of valid options for kernel_definition,
          e.g. "sum(pq_diff;p_space)" or "sum(pq_diff;q_space)". This uses the sum of the named basis filters.
        - "concat(<filters>)" where <filters> is a ";"-separated list (without spaces) of strings "<output_channels>:<filter_type>"
          where <output_channels> lists the channels of each order where the named filter is to be used
          (e.g. "[3, 4]" to use it for 3 scalar and 4 vector output channelw) and
          <filter_type> names a valid kernel_definition to use for these output channels.
          The number of all concatenated channels needs to math type_out.
          Example: "concat([3,4]:pq_diff,[5,2,1]:p_space)" which would require type_out = [8,6,1]

    :param q_sampling_schema_in: The q-sampling schema of input feature map.
        The q-sampling schema may either be given as a Q_SamplingSchema object,
        a Tensor of size (Q_in, 3) or a list of length Q_in (one element for each vector) of lists of size 3 of floats.
        Note that Q_in is not explicitly given but derived form the length of this parameter.
        If this is None (default) then the input does not have q-space but only p-space.
    :param q_sampling_schema_out: The q-sampling schema of output feature map.
        The q-sampling schema may either be given as a Q_SamplingSchema object,
        a Tensor of size (Q_out, 3) or a list of length Q_out (one element for each vector) of lists of size 3 of floats.
        Note that Q_out is not explicitly given but derived form the length of this parameter.
        If this is None (default) then the output does not have q-space but only p-space.
    :param non_linearity_config: Dict with the following optional keys:

        - tensor_non_lin: The nonlinearity to use for channels with l>0 (non-scalar channels).
          Default (and currently only option) is "gated".
        - scalar_non_lin: The nonlinearity to use for channles with l=0 (scalar channels).
          Valid options are "swish" and "relu".
          Default is "swish".

    :param use_non_linearity: Whether to use a nonlinearity.
    :param batch_norm_config: Dict with the following optional keys:

        - eps: avoid division by zero when we normalize by the variance
        - momentum: momentum of the running average
        - affine: do we have weight and bias parameters
        - reduce: method to contract over the spacial dimensions

    :param use_batch_norm: Whether to use a batch normalization
    :param transposed: Whether to perform a transposed convolution using the equivariant kernel
    :param auto_recompute: Whether to automatically recompute the kernel in each forward pass.
        By default it is recomputed each time.
        If this parameter is set to false, it is not recomputed and the method recompute() needs to be called
        explicitly after parameters of this nn.Module have been updated.
    :param kernel_selection_rule: Rule defining which angular filter orders (l_filter) to use
        for a paths form input orders l_in to output orders l_out.
        Defaults to using all possible filter orders,
        i.e. all l_filter with \|l_in - l_out\| <= l_filter <= l_in + l_out.
        Options are:

        - dict with key "lmax" and int value which additionally defines a maximum l_filter.
        - dict with int-pairs as keys and list of ints as values that defines
          for each pair of l_in and l_out the list of l_filter to use.
          E.g. {(0,0): [0], (1,1): [0,1], (0,1): [1]}

    :param p_radial_basis_type: The radial basis function type used for p-space.
        Valid options are "gaussian" (default), "cosine", "bessel".
        Note that this parameter is ignored if there is no basis filter using p-space.
    :param p_radial_basis_params: A (optional) dict of additional parameters for the radial basis function used for p-space.
        Valid keys in this dict are:

        - num_layers: Number of layers in the FC applied to the radial basis function.
          If num_layers = 0 (default) then no FC is applied to the radial basis function.
        - num_units: Number of units (neurons) in each of the layer in the FC applied to the radial basis function.
          No default, this parameter is required and must be >0 if num_layers > 0.
        - activation_function: activation function used in the FC applied to the radial basis function,
          valid are "relu" (default) or "swish"

        Note that this parameter is ignored if there is no basis filter using p-space.
    :param q_radial_basis_type: The radial basis function type used for q-space (q-in and q-out).
        Valid options are "gaussian" (default), "cosine", "bessel".
        Note that this parameter is ignored if there is no basis filter using q-space.
    :param q_out_radial_basis_type: The radial basis function type used for q-out (q-space of output feature map).
        See q_radial_basis_type but only for q-out.
        Defaults to q_radial_basis_type.
    :param q_in_radial_basis_type: The radial basis function type used for q-in (q-space of input feature map).
        See q_radial_basis_type but only for q-in.
        Defaults to q_radial_basis_type.
    :param q_radial_basis_params: A (optional) dict of additional parameters for the radial basis function used for q-space.
        Valid keys in this dict are:

        - num_layers: Number of layers in the FC applied to the radial basis function.
          If num_layers = 0 (default) then no FC is applied to the radial basis function.
        - num_units: Number of units (neurons) in each of the layer in the FC applied to the radial basis function.
          No default, this parameter is required and must be >0 if num_layers > 0.
        - activation_function: activation function used in the FC applied to the radial basis function,
          valid are "relu" (default) or "swish"

        Note that this parameter is ignored if there is no basis filter using q-space.
    :param q_out_radial_basis_params: A dict of additional parameters for the radial basis function used for q-out (q-space of output feature map).
        See q_radial_basis_params but only for q-out.
        Defaults to q_radial_basis_params.
    :param q_in_radial_basis_params: A dict of additional parameters for the radial basis function used for q-in (q-space of input feature map).
        See q_radial_basis_params but only for q-in.
        Defaults to q_radial_basis_params.
    :param sub_kernel_selection_rule:
        Rule defining for the TP filter which pairs of l_p and l_q to use for each l_filter.
        Defaults to "TP\pm 1".
        Options are:
        
        - dict with string keys: defines some constraints which combinations to use.
          The following constraint always holds:
          \|l_p - l_q\| <= l_filter <= l_p + l_q
          Additionally constraints can be defined by the following keys in the dict:

          - "l_diff_to_out_max": Maximum difference between l_p and l_filter as well as l_q and l_filter.
            Default to 1 (as in "TP\pm 1")
          - "l_max" (optional): Maximum value for l_p and l_q.
          - "l_in_diff_max" (optional): Maximum difference between l_p and l_q.

        - dict with ints as keys and list of int-pairs as values that defines
          for each l_filter the used pairs of l_p and l_q.
          E.g. {0: [(0, 0), (1, 1)], 1: [(0, 1), (1, 0), (1, 1)]}

        Note that this parameter is ignored if no TP-filter basis is used.
    For additional parameters see EquivariantPQLayer.
    """
    type_in = SphericalTensorType.from_multiplicities_or_type(type_in)
    type_out = SphericalTensorType.from_multiplicities_or_type(type_out)

    if batch_norm_config is None:
        batch_norm_config = {}
    if non_linearity_config is None:
        non_linearity_config = {}

    if use_non_linearity:
        type_non_lin_in, non_linearity = build_non_linearity(type_out, **non_linearity_config)
        conv = EquivariantPQLayer(type_in, type_non_lin_in,
                                  kernel_definition=kernel,
                                  p_kernel_size=p_kernel_size,
                                  q_sampling_schema_in=q_sampling_schema_in,
                                  q_sampling_schema_out=q_sampling_schema_out,
                                  transposed=transposed,
                                  auto_recompute_kernel=auto_recompute,
                                  **kernel_kwargs)
        if use_batch_norm:
            batch_norm = BatchNorm(type_non_lin_in.Rs, **batch_norm_config)
            return nn.Sequential(
                OrderedDict([('conv', conv), ('batch_norm', batch_norm), ('non_linearity', non_linearity)]))
        else:
            return nn.Sequential(OrderedDict([('conv', conv), ('non_linearity', non_linearity)]))
    else:
        conv = EquivariantPQLayer(type_in, type_out,
                                  kernel_definition=kernel,
                                  p_kernel_size=p_kernel_size,
                                  q_sampling_schema_in=q_sampling_schema_in,
                                  q_sampling_schema_out=q_sampling_schema_out,
                                  transposed=transposed,
                                  auto_recompute_kernel=auto_recompute,
                                  **kernel_kwargs)
        if use_batch_norm:
            batch_norm = BatchNorm(type_out.Rs, **batch_norm_config)
            return nn.Sequential(OrderedDict([('conv', conv), ('batch_norm', batch_norm)]))
        else:
            return conv


def build_p_layer(type_in: Union[SphericalTensorType, List[int]],
                  type_out: Union[SphericalTensorType, List[int]],
                  kernel_size: int,
                  non_linearity_config=None,
                  use_non_linearity=True,
                  batch_norm_config=None,
                  use_batch_norm=True,
                  transposed=False,
                  auto_recompute=True,
                  **kernel_kwargs):
    """
    Builds a p-layer consisting of an EquivariantPLayer followed by a nonlinearity (e.g. gated nonlinearity).

    :param type_in: The spherical tensor type of the input feature map.
        This defines how many channels of each tensor order the input feature map has.
        It can either be given as SphericalTensorType object or as List[int]] the element at index i of the list
        defines the number of order-i channels,
        e.g. the first element defines the number of order-0 (scalar) channels
        and the second the number of order-1 (vector) channels and so on.
        For all orders corresponding to out-of-range indices the number of channels is 0.
    :param type_out: The spherical tensor type of the output feature map (after non-linearity).
        This defines how many channels of each tensor order the output feature map has.
        It can either be given as SphericalTensorType object or as List[int]] the element at index i of the list
        defines the number of order-i channels,
        e.g. the first element defines the number of order-0 (scalar) channels
        and the second the number of order-1 (vector) channels and so on.
        For all orders corresponding to out-of-range indices the number of channels is 0.
    :param p_kernel_size: Size of the kernel in p-space.
        Note that the kernel always covers the whole q-space (as it is not translationally equivariant),
        so there is no q_kernel_size.
    :param non_linearity_config: Dict with the following optional keys:

        - tensor_non_lin: The nonlinearity to use for channels with l>0 (non-scalar channels).
          Default (and currently only option) is "gated".
        - scalar_non_lin: The nonlinearity to use for channles with l=0 (scalar channels).
          Valid options are "swish" and "relu".
          Default is "swish".
    
    :param use_non_linearity: Whether to use a nonlinearity.
    :param batch_norm_config: Dict with the following optional keys:

        - eps: avoid division by zero when we normalize by the variance
        - momentum: momentum of the running average
        - affine: do we have weight and bias parameters
        - reduce: method to contract over the spacial dimensions

    :param use_batch_norm: Whether to use a batch normalization
    :param transposed: Whether to perform a transposed convolution using the equivariant kernel
    :param auto_recompute: Whether to automatically recompute the kernel in each forward pass.
        By default it is recomputed each time.
        If this parameter is set to false, it is not recomputed and the method recompute() needs to be called
        explicitly after parameters of this nn.Module have been updated.
    :param kernel_selection_rule: Rule defining which angular filter orders (l_filter) to use
        for a paths form input orders l_in to output orders l_out.
        Defaults to using all possible filter orders,
        i.e. all l_filter with \|l_in - l_out\| <= l_filter <= l_in + l_out.
        Options are:

        - dict with key "lmax" and int value which additionally defines a maximum l_filter.
        - dict with int-pairs as keys and list of ints as values that defines
          for each pair of l_in and l_out the list of l_filter to use.
          E.g. {(0,0): [0], (1,1): [0,1], (0,1): [1]}

    :param p_radial_basis_type: The radial basis function type used for p-space.
        Valid options are "gaussian" (default), "cosine", "bessel".
        Note that this parameter is ignored if there is no basis filter using p-space.
    :param p_radial_basis_params: A (optional) dict of additional parameters for the radial basis function used for p-space.
        Valid keys in this dict are:

        - num_layers: Number of layers in the FC applied to the radial basis function.
          If num_layers = 0 (default) then no FC is applied to the radial basis function.
        - num_units: Number of units (neurons) in each of the layer in the FC applied to the radial basis function.
          No default, this parameter is required and must be >0 if num_layers > 0.
        - activation_function: activation function used in the FC applied to the radial basis function,
          valid are "relu" (default) or "swish"

        Note that this parameter is ignored if there is no basis filter using p-space.
    For additional parameters see EquivariantPLayer.
    """
    return build_pq_layer(type_in, type_out, kernel_size,
                          kernel='p_space',
                          q_sampling_schema_in=None, q_sampling_schema_out=None,
                          non_linearity_config=non_linearity_config,
                          use_non_linearity=use_non_linearity,
                          batch_norm_config=batch_norm_config,
                          use_batch_norm=use_batch_norm,
                          transposed=transposed,
                          auto_recompute=auto_recompute,
                          **kernel_kwargs)


def build_q_reduction_layer(type_in: Union[SphericalTensorType, List[int]], q_sampling_schema_in: Q_SamplingSchema,
                            reduction='length_weighted_average',
                            auto_recompute=True,
                            **kwargs):
    """
    Builds a q-reduction layer to globally reduce q-space leaving only p-space.

    :param type_in: The spherical tensor type of the input feature map.
        This defines how many channels of each tensor order the input feature map has.
        It can either be given as SphericalTensorType object or as List[int]] the element at index i of the list
        defines the number of order-i channels,
        e.g. the first element defines the number of order-0 (scalar) channels
        and the second the number of order-1 (vector) channels and so on.
        For all orders corresponding to out-of-range indices the number of channels is 0.
    :param q_sampling_schema_in: The q-sampling schema of input feature map.
        The q-sampling schema may either be given as a Q_SamplingSchema object,
        a Tensor of size (Q_in, 3) or a list of length Q_in (one element for each vector) of lists of size 3 of floats.
        Note that Q_in is not explicitly given but derived form the length of this parameter.
        If this is None (default) then the input does not have q-space but only p-space.
    :param reduction: The type of reduction to use. Valid options are:

        - length_weighted_average: To use QLengthWeightedAvgPool (global length-weighted avg-pooling over q-space)
          For additional parameters in param kwargs see QLengthWeightedAvgPool.
        - mean: To use global avg-pooling over q-space.
        - conv: To use an EquivariantPQLayer (and gated nonlinearity) without output q-space.
          For additional parameters in param kwargs see build_pq_layer
          (except the params type_out, q_sampling_schema_out).

    :param auto_recompute: Whether to automatically recompute the kernels in each forward pass.
    :return (reduction_layer, type_out):

        - reduction_layer: The created q-reduction layer (nn.Module)
        - type_out: The spherical tensor type of the output feature map.
    """
    type_in = SphericalTensorType.from_multiplicities_or_type(type_in)
    if reduction == 'length_weighted_average':
        return QLengthWeightedAvgPool(type_in, q_sampling_schema_in,
                                      auto_recompute=auto_recompute, **kwargs), type_in
    elif reduction == 'mean':
        return partial(torch.mean, dim=2), type_in
    elif reduction == 'conv':
        type_out = SphericalTensorType.from_multiplicities_or_type(kwargs.pop('type_out', type_in))
        return build_pq_layer(type_in, type_out,
                              q_sampling_schema_in=q_sampling_schema_in,
                              q_sampling_schema_out=None,
                              **kwargs), type_out
    else:
        raise ValueError(f'q-reduction "{reduction}" not supported.')


def build_non_linearity(type_out: SphericalTensorType, tensor_non_lin='gated', scalar_non_lin='swish') -> (
        SphericalTensorType, nn.Module):
    """
    Builds a nonlinearity for spherical tensor feature maps.
    Currently only the gated nonlinearity is supported.

    :param type_out: The spherical tensor type of the output feature map (after non-linearity).
        This defines how many channels of each tensor order the output feature map has.
        It can either be given as SphericalTensorType object or as List[int]] the element at index i of the list
        defines the number of order-i channels,
        e.g. the first element defines the number of order-0 (scalar) channels
        and the second the number of order-1 (vector) channels and so on.
        For all orders corresponding to out-of-range indices the number of channels is 0.
    :param tensor_non_lin: The nonlinearity to use for channels with l>0 (non-scalar channels).
        Default (and currently only option) is "gated".
    :param scalar_non_lin: The nonlinearity to use for channles with l=0 (scalar channels).
        Valid options are "swish" and "relu".
        Default is "swish".
    :return (type_in, nonlinearity):

        - type_in: The expected spherical tensor type of the input feature map.
        - nonlinearity: the nonlinearity (as nn.Module) which accepts the input feature map.
    """
    type_out = SphericalTensorType.from_multiplicities_or_type(type_out)
    if tensor_non_lin == 'gated':
        scalar_non_lin = get_scalar_non_linearity(scalar_non_lin)
        non_lin = GatedBlockNonLin(type_out.Rs, scalar_non_lin, sigmoid)
        return SphericalTensorType.from_Rs(non_lin.Rs_in), non_lin
    else:
        raise ValueError(f'Tensor Non-linearity "{tensor_non_lin}" not supported.')


class GatedBlockNonLin(GatedBlock):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x):
        x = super(GatedBlockNonLin, self).forward(x, dim=1)
        return x


class BatchNorm(E3NNBatchNorm):
    def __init__(self, rs, eps=1e-5, momentum=0.1, affine=True, reduce='mean', normalization='component'):
        """
        Adapted batch normalization from e3nn library:

        Batch normalization layer for orthonormal representations
        It normalizes by the norm of the representations.
        Not that the norm is invariant only for orthonormal representations.
        Irreducible representations `o3.irr_repr` are orthonormal.

        input shape : [batch, stacked orthonormal representations, q_dim, [spacial dimensions]]

        :param rs: list of tuple (multiplicity, dimension)
        :param eps: avoid division by zero when we normalize by the variance
        :param momentum: momentum of the running average
        :param affine: do we have weight and bias parameters
        :param reduce: method to contract over the spacial dimensions
        """
        rs = [(m, d + 1) for m, d in rs if m * (d + 1) > 0]
        super().__init__(rs, eps, momentum, affine, reduce, normalization)

    def forward(self, data):
        """
        :param data: [batch, stacked features, q_dim, x, y, z]
        """
        data = data.permute(0, 2, 3, 4, 5, 1)
        output = super().forward(data)
        return output.permute(0, 5, 1, 2, 3, 4).contiguous()
