from typing import List, Union, Optional

import torch
from torch import nn
import torch.nn.functional as F
import math

from equideepdmri.layers.Recomputeable import Recomputable
from equideepdmri.layers.filter.filter_kernel import KernelDefinitionInterface
from equideepdmri.layers.filter.kernel_builders import build_kernel
from equideepdmri.utils.q_space import Q_SamplingSchema
from equideepdmri.utils.spherical_tensor import SphericalTensorType


class EquivariantPQLayer(nn.Module, Recomputable):
    """
    Linear layer for data with p- and q-space
    that is equivariant to joint rotations in p- and q-space and to translations in p-space.

    Input: (N x dim_in x [Q_in] x P_z x P_y x P_x)
    Output: (N x dim_out x [Q_out] x P_z_out x P_y_out x P_x_out)
    """
    def __init__(self,
                 type_in: Union[SphericalTensorType, List[int]],
                 type_out: Union[SphericalTensorType, List[int]],
                 kernel_definition: Union[KernelDefinitionInterface, str, None],
                 p_kernel_size: int,
                 q_sampling_schema_in: Union[Q_SamplingSchema, torch.Tensor, List] = None,
                 q_sampling_schema_out: Union[Q_SamplingSchema, torch.Tensor, List] = None,
                 p_padding: Union[str, int] = 'same',
                 p_stride: int = 1,
                 p_dilation: int = 1,
                 groups=1,
                 auto_recompute_kernel=True,
                 normalize_q_sampling_schema_in=True,
                 normalize_q_sampling_schema_out=True,
                 scalar_bias=True,
                 **kernel_kwargs):
        """
        Linear layer for data with p- and q-space
        that is equivariant to joint rotations in p- and q-space and to translations in p-space.

        Input: (N x dim_in x [Q_in] x P_z x P_y x P_x)
        Output: (N x dim_out x [Q_out] x P_z_out x P_y_out x P_x_out)

        :param type_in: The spherical tensor type of the input feature map.
            This defines how many channels of each tensor order the input feature map has.
            It can either be given as SphericalTensorType object or as List[int]] the element at index i of the list
            defines the number of order-i channels,
            e.g. the first element defines the number of order-0 (scalar) channels
            and the second the number of order-1 (vector) channels and so on.
            For all orders corresponding to out-of-range indices the number of channels is 0.
        :param type_out: The spherical tensor type of the output feature map.
            This defines how many channels of each tensor order the output feature map has.
            It can either be given as SphericalTensorType object or as List[int]] the element at index i of the list
            defines the number of order-i channels,
            e.g. the first element defines the number of order-0 (scalar) channels
            and the second the number of order-1 (vector) channels and so on.
            For all orders corresponding to out-of-range indices the number of channels is 0.
        :param kernel_definition: Which filter basis to use in this layer.
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

        :param p_kernel_size: Size of the kernel in p-space.
            Note that the kernel always covers the whole q-space (as it is not translationally equivariant),
            so there is no q_kernel_size.
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
        :param p_padding: Padding used in p-space. Either int or the string "same" (default).
            If p_padding is "same", then the padding is determined automatically so that the input p-space
            size is the same as the output p-space size.
            Note that p_padding "same" cannot be used if p_stride != 1.
            Note that there is no q_padding
            as the kernel always covers the whole q-space (as it is not translationally equivariant).
        :param p_stride: Stride used in p-space. Note that there is no q_stride
            as the kernel always covers the whole q-space (as it is not translationally equivariant).
        :param p_dilation: Dilation used in p-space. Note that there is no q_dilation
            as the kernel always covers the whole q-space (as it is not translationally equivariant).
        :param groups: Controls the connections between inputs and outputs.
            This is the analog to the paraneter groups of the Conv3D layer from PyTorch.
            Note that the name groups is not related to groups in a mathematical sense (like the group of rotaions)
        :param auto_recompute_kernel:
            Whether to automatically recompute the kernel in each forward pass.
            By default it is recomputed each time.
            If this parameter is set to false, it is not recomputed and the method recompute() needs to be called
            explicitly after parameters of this nn.Module have been updated.
        :param normalize_q_sampling_schema_in: Whether to normalize the input q-sampling schema.
            If true, the vectors in the input sampling schema are normalized to lengths between 0 and 1.
            If the input sampling schema only contains a single 0-vector, no normalization is applied.
        :param normalize_q_sampling_schema_out: Whether to normalize the output q-sampling schema.
            If true, the vectors in the output sampling schema are normalized to lengths between 0 and 1.
            If the output sampling schema only contains a single 0-vector, no normalization is applied.
        :param scalar_bias: Whether to use a learned bias for each scalar (order 0) output channel.
        :param kernel_selection_rule: Rule defining which angular filter orders (l_filter) to use
            for a paths form input orders l_in to output orders l_out.
            Defaults to using all possible filter orders,
            i.e. all l_filter with  |l_in - l_out | <= l_filter <= l_in + l_out.
            Options are:

            - dict with key "lmax" and int value which additionally defines a maximum l_filter.
            - dict with int-pairs as keys and list of ints as values that defines
              for each pair of l_in and l_out the list of l_filter to use.
              E.g. {(0,0): [0], (1,1): [0,1], (0,1): [1]}

        :param use_linear_model_for_zero_length: Whether to use a linear point-wise model instead of the normal kernel
            for zero lengths in the radial filter basis (default is True).
            See zero_length_eps which defines which lengths are treated as zeros.
            This options is useful as for small lengths the angular basis might get very inaccurate.
        :param zero_length_eps: The epsilon value for treating lengths as zero (see use_linear_model_for_zero_length).
            A length is treated as zero if length < zero_length_eps.
            Only relevant if use_linear_model_for_zero_length is True.
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
        :param normalize_Q_before_diff: Defaults to True.
            Whether to noramlize the q-vectors of the input and output sampling schema
            before computing the offset that might be used the in the angular basis.
            Note that this parameter does not influence the normalization of q-sampling schemas for the radial basis.
            See normalize_q_sampling_schema_in and normalize_q_sampling_schema_out to always normalize these schemas
            (for the radial and the angular basis)
            Note that this parameter is ignored if there is no basis filter using q-space.
        :param sub_kernel_selection_rule:
            Rule defining for the TP filter which pairs of l_p and l_q to use for each l_filter.
            Defaults to "TP\pm 1".
            Options are:
            
            - dict with string keys: defines some constraints which combinations to use.
              The following constraint always holds:
              |l_p - l_q | <= l_filter <= l_p + l_q
              Additionally constraints can be defined by the following keys in the dict:

              - "l_diff_to_out_max": Maximum difference between l_p and l_filter as well as l_q and l_filter.
                Default to 1 (as in "TP\pm 1")
              - "l_max" (optional): Maximum value for l_p and l_q.
              - "l_in_diff_max" (optional): Maximum difference between l_p and l_q.

            - dict with ints as keys and list of int-pairs as values that defines
              for each l_filter the used pairs of l_p and l_q.
              E.g. {0: [(0, 0), (1, 1)], 1: [(0, 1), (1, 0), (1, 1)]}
                
            Note that this parameter is ignored if no TP-filter basis is used.
        :param normalization: The normalization used for the spherical harmonics and the tensor product.
            Valid values are "component" (default) or "norm".
        """
        super().__init__()

        # ----- out/in types -----
        self.type_out = SphericalTensorType.from_multiplicities_or_type(type_out)
        self.type_in = SphericalTensorType.from_multiplicities_or_type(type_in)

        # ----- compute Q/P vectors and initialize kernel -----

        self.has_Q_in = q_sampling_schema_in is not None
        if self.has_Q_in:
            self.q_sampling_schema_in = Q_SamplingSchema.from_q_sampling_schema(q_sampling_schema_in)
            if normalize_q_sampling_schema_in:
                self.q_sampling_schema_in = self.q_sampling_schema_in.normalized
        else:
            self.q_sampling_schema_in = None

        self.has_Q_out = q_sampling_schema_out is not None
        if self.has_Q_out:
            self.q_sampling_schema_out = Q_SamplingSchema.from_q_sampling_schema(q_sampling_schema_out)
            if normalize_q_sampling_schema_out:
                self.q_sampling_schema_out = self.q_sampling_schema_out.normalized
        else:
            self.q_sampling_schema_out = None

        self.p_kernel_size = p_kernel_size
        self.p_diff_vectors = self._compute_P_diff_vectors(p_kernel_size)

        if kernel_definition is None:
            # Default kernel uses p_space only
            kernel_definition = 'p_space'
        # build the kernel constructor based on the kernel_type (which might be a string)
        kernel_constructor = build_kernel(kernel_definition, has_Q_in=self.has_Q_in, has_Q_out=self.has_Q_out,
                                          **kernel_kwargs)
        self.kernel = kernel_constructor(self.type_out, self.type_in,
                                         self.q_sampling_schema_out,
                                         self.q_sampling_schema_in,
                                         self.p_diff_vectors,
                                         self.p_kernel_size)
        self.computed_kernel: Optional[torch.Tensor] = None

        # ----- conv parameters -----
        if isinstance(p_padding, int):
            self.p_padding = p_padding
        elif p_padding == 'same':
            assert p_stride == 1, f'"same" padding for P_padding is only possible if P_stride == 1, but P_stride was {p_stride}'
            self.p_padding = math.ceil((p_kernel_size - 1) / 2)
        else:
            raise ValueError(f'Unsupported value for P_padding: "{p_padding}". '
                             f'Supported values are "same" or any positive int')

        self.p_stride = p_stride
        self.p_dilation = p_dilation
        self.groups = groups

        # ----- other parameters -----
        self.auto_recompute_kernel = auto_recompute_kernel

        # ----- scalar bias -----
        scalar_bias_size = self.type_out.C_l(l=0)
        if scalar_bias and scalar_bias_size > 0:
            self.scalar_bias = torch.nn.Parameter(torch.randn(scalar_bias_size), requires_grad=True)
        else:
            self.scalar_bias = None

    @property
    def Q_in(self):
        return self.q_sampling_schema_in.Q if self.q_sampling_schema_in is not None else 1

    @property
    def Q_out(self):
        return self.q_sampling_schema_out.Q if self.q_sampling_schema_out is not None else 1

    @property
    def num_P_diff_vectors(self):
        return self.p_diff_vectors.size()[0]

    @staticmethod
    def _compute_P_diff_vectors(P_kernel_size: int):
        r = torch.linspace(-1, 1, P_kernel_size)
        # Note: permute is required because conv3d (where the kernel will be used)
        # expects the kernel dimension to be in the order z, y, x (depth, height, width)
        # so the order needs to be changed from x, y, z -> z, y, x
        # ((P_kernel_size * P_kernel_size * P_kernel_size) x 3)
        return torch.stack(torch.meshgrid(r, r, r), dim=-1).permute(2, 1, 0, 3).reshape(-1, 3)

    def recompute(self):
        self.computed_kernel = None
        self.computed_kernel = self.kernel()  # (Q_out x Q_in x num_P_diff_vectors x type_out.dim x type_in.dim)
        assert self.computed_kernel.size() == (self.Q_out, self.Q_in, self.num_P_diff_vectors, self.type_out.dim, self.type_in.dim), \
            f'Invalid size for computed kernel. ' \
            f'Expected size {(self.Q_out, self.Q_in, self.num_P_diff_vectors, self.type_out.dim, self.type_in.dim)} ' \
            f'=> (Q_out x Q_in x num_P_diff_vectors x type_out.dim x type_in.dim) ' \
            f'but size was {self.computed_kernel.size()}.'

        # ((type_out.dim * Q_out) x (type_in.dim * Q_in) x P_kernel_size x P_kernel_size x P_kernel_size)
        self.computed_kernel = self.computed_kernel.permute(3, 0, 4, 1, 2)\
            .reshape((self.type_out.dim * self.Q_out), (self.type_in.dim * self.Q_in),
                     self.p_kernel_size, self.p_kernel_size, self.p_kernel_size)

        # normalize for P and Q_in
        self.computed_kernel.mul_(1 / ((self.p_kernel_size ** 3 * self.Q_in) ** 0.5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies the layer to input feature map x.

        :param x: Input feature map. Dim (N x dim_in x [Q_in] x P_z x P_y x P_x) with
        
            - N: batch size
            - dim_in: size of the spherical tensor at each point of the input feature map.
            - Q_in: size of the input q-space sampling schema.
              Optional: if no q_sampling_schema_in is specified (not None), the input is assumed to have only p-space but no q-space.
            - P_z, P_y, P_x: p-space size.

        :return: The output feature map. Dim (N x dim_out x [Q_out] x P_z_out x P_y_out x P_x_out) with

            - N: batch size
            - dim_out: size of the spherical tensor at each point of the output feature map.
            - Q_out: size of the output q-space sampling schema.
              Optional: if no q_sampling_schema_out is specified (not None), the output has only p-space but no q-space.
            - P_z_out, P_y_out, P_x_out: p-space size of the output feature map.
              Depends on P_z, P_y, P_x, p_kernel_size, p_padding, p_stride, and p_dilation.
        """
        assert isinstance(x, torch.Tensor)
        if self.has_Q_in:
            assert x.ndim == 6
            N, dim_in, Q_in, *P_size_in = x.size()
        else:
            assert x.ndim == 5
            N, dim_in, *P_size_in = x.size()
            Q_in = 1
        assert dim_in == self.type_in.dim
        assert Q_in == self.Q_in

        if self.auto_recompute_kernel:
            self.recompute()

        assert self.computed_kernel is not None, \
            'If auto_recompute_kernel=False, then recompute_kernel() needs to be called ' \
            'before using this layer the first time.'

        # (N x (type_in.dim * Q_in) x P_size_z_in x P_size_y_in x P_size_x_in)
        x = x.view(-1, (self.type_in.dim * self.Q_in), *P_size_in)
        # (N x (type_out.dim * Q_out) x P_size_z_out x P_size_y_out x P_size_x_out)
        x = F.conv3d(x, self.computed_kernel,
                     padding=self.p_padding, stride=self.p_stride, dilation=self.p_dilation, groups=self.groups)

        if self.auto_recompute_kernel:
            self.computed_kernel = None  # free memory of computed kernel
        P_size_out = x.size()[-3:]
        # (N x type_out.dim x Q_out x P_size_z_out x P_size_y_out x P_x_size_out)
        x = x.view(N, self.type_out.dim, self.Q_out, *P_size_out)
        if self.scalar_bias is not None:
            x[:, self.type_out.slice_l(l=0), :, :, :, :] += self.scalar_bias.view(1, -1, 1, 1, 1, 1)

        if not self.has_Q_out:
            x = x.squeeze(2)  # squeeze the Q_dim
        return x

    def __repr__(self):
        if self.has_Q_in or self.has_Q_out:
            return f'<EquivariantPQLayer {self.type_in.multiplicities}->{self.type_out.multiplicities}>'
        else:
            return f'<EquivariantPLayer {self.type_in.multiplicities}->{self.type_out.multiplicities}>'


def EquivariantPLayer(*args, **kwargs):
    """
    Linear layer for data with p-space only
    that is equivariant to rotations and translations in p-space.

    Input: (N x dim_in x P_z x P_y x P_x)
    Output: (N x dim_out x P_z_out x P_y_out x P_x_out)

    :param type_in: The spherical tensor type of the input feature map.
        This defines how many channels of each tensor order the input feature map has.
        It can either be given as SphericalTensorType object or as List[int]] the element at index i of the list
        defines the number of order-i channels,
        e.g. the first element defines the number of order-0 (scalar) channels
        and the second the number of order-1 (vector) channels and so on.
        For all orders corresponding to out-of-range indices the number of channels is 0.
    :param type_out: The spherical tensor type of the output feature map.
        This defines how many channels of each tensor order the output feature map has.
        It can either be given as SphericalTensorType object or as List[int]] the element at index i of the list
        defines the number of order-i channels,
        e.g. the first element defines the number of order-0 (scalar) channels
        and the second the number of order-1 (vector) channels and so on.
        For all orders corresponding to out-of-range indices the number of channels is 0.
    :param p_kernel_size: Size of the kernel in p-space.
    :param p_padding: Padding used in p-space. Either int or the string "same" (default).
        If p_padding is "same", then the padding is determined automatically so that the input p-space
        size is the same as the output p-space size.
        Note that p_padding "same" cannot be used if p_stride != 1.
    :param p_stride: Stride used in p-space.
    :param p_dilation: Dilation used in p-space.
    :param groups: Controls the connections between inputs and outputs.
        This is the analog to the paraneter groups of the Conv3D layer from PyTorch.
        Note that the name groups is not related to groups in a mathematical sense (like the group of rotaions)
    :param auto_recompute_kernel:
        Whether to automatically recompute the kernel in each forward pass.
        By default it is recomputed each time.
        If this parameter is set to false, it is not recomputed and the method recompute() needs to be called
        explicitly after parameters of this nn.Module have been updated.
    :param scalar_bias: Whether to use a learned bias for each scalar (order 0) output channel.
    :param kernel_selection_rule: Rule defining which angular filter orders (l_filter) to use
        for a paths form input orders l_in to output orders l_out.
        Defaults to using all possible filter orders,
        i.e. all l_filter with  |l_in - l_out | <= l_filter <= l_in + l_out.
        Options are:

        - dict with key "lmax" and int value which additionally defines a maximum l_filter.
        - dict with int-pairs as keys and list of ints as values that defines
          for each pair of l_in and l_out the list of l_filter to use.
          E.g. {(0,0): [0], (1,1): [0,1], (0,1): [1]}

    :param use_linear_model_for_zero_length: Whether to use a linear point-wise model instead of the normal kernel
        for zero lengths in the radial filter basis (default is True).
        See zero_length_eps which defines which lengths are treated as zeros.
        This options is useful as for small lengths the angular basis might get very inaccurate.
    :param zero_length_eps: The epsilon value for treating lengths as zero (see use_linear_model_for_zero_length).
        A length is treated as zero if length < zero_length_eps.
        Only relevant if use_linear_model_for_zero_length is True.
    :param p_radial_basis_type: The radial basis function type used for p-space.
        Valid options are "gaussian" (default), "cosine", "bessel".
    :param p_radial_basis_params: A (optional) dict of additional parameters for the radial basis function used for p-space.
        Valid keys in this dict are:

        - num_layers: Number of layers in the FC applied to the radial basis function.
          If num_layers = 0 (default) then no FC is applied to the radial basis function.
        - num_units: Number of units (neurons) in each of the layer in the FC applied to the radial basis function.
          No default, this parameter is required and must be >0 if num_layers > 0.
        - activation_function: activation function used in the FC applied to the radial basis function,
          valid are "relu" (default) or "swish"

    :param normalization: The normalization used for the spherical harmonics and the tensor product.
        Valid values are "component" (default) or "norm".
    """
    return EquivariantPQLayer(*args, kernel_definition=None, q_sampling_schema_in=None, q_sampling_schema_out=None,
                              **kwargs)
