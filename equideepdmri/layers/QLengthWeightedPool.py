import torch
from torch import nn

from equideepdmri.layers.Recomputeable import Recomputable
from equideepdmri.layers.filter.radial_basis_functions import build_radial_basis_constructor
from equideepdmri.utils.q_space import Q_SamplingSchema
from equideepdmri.layers.filter.utils import compute_channel_mapping_matrix, SphericalTensorType


class QLengthWeightedAvgPool(nn.Module, Recomputable):
    """
    Pooling layer that globally pools over q-space (and thus removing it) using a weighted average.

    All q vectors of the same length are weighted equally. It is thus invariant to rotations.
    The weights for the q vectors are based on a radial basis and learned weights.
    They are not constrained so also negative weights can be learned and the sum of all weights is also not constrained.
    For each channel different weights can be learned.

    Input: (N x dim_in_out x Q_in x P_z x P_y x P_x)
    Output: (N x dim_in_out x P_z x P_y x P_x)
    """
    def __init__(self,
                 type_in_out: SphericalTensorType,
                 q_sampling_schema_in: Q_SamplingSchema,
                 q_radial_basis_type='gaussian',
                 q_radial_basis_params=None,
                 auto_recompute=True):
        """
        Pooling layer that globally pools over q-space (and thus removing it) using a weighted average.

        All q vectors of the same length are weighted equally. It is thus invariant to rotations.
        The weights for the q vectors are based on a radial basis and learned weights.
        They are not constrained so also negative weights can be learned and the sum of all weights is also not constrained.
        For each channel different weights can be learned.

        Input: (N x dim_in x Q_in x P_z x P_y x P_x)
        Output: (N x dim_out x P_z x P_y x P_x)

        :param type_in_out: The spherical tensor type of the input and output feature map (the type is the same).
            This defines how many channels of each tensor order the input/output feature map has.
            It can either be given as SphericalTensorType object or as List[int]] the element at index i of the list
            defines the number of order-i channels,
            e.g. the first element defines the number of order-0 (scalar) channels
            and the second the number of order-1 (vector) channels and so on.
            For all orders corresponding to out-of-range indices the number of channels is 0.
        :param q_sampling_schema_in: The q-sampling schema of input feature map.
            Note that there is no output q_sampling_schema as q-space is reduced.
            Note that Q_in is not explicitly given but derived form the length of this parameter.
        :param q_radial_basis_type: The radial basis function type used for weighting q-space lengths.
            Valid options are "gaussian" (default), "cosine", "bessel".
        :param q_radial_basis_params: A (optional) dict of additional parameters for the radial basis function used for q-space.
            Valid keys in this dict are:
            - num_layers: Number of layers in the FC applied to the radial basis function.
                If num_layers = 0 (default) then no FC is applied to the radial basis function.
            - num_units: Number of units (neurons) in each of the layer in the FC applied to the radial basis function.
                No default, this parameter is required and must be >0 if num_layers > 0.
            - activation_function: activation function used in the FC applied to the radial basis function,
                valid are "relu" (default) or "swish"
        :param auto_recompute:
            Whether to automatically recompute the kernel in each forward pass.
            By default it is recomputed each time.
            If this parameter is set to false, it is not recomputed and the method recompute() needs to be called
            explicitly after parameters of this nn.Module have been updated.
        """
        super().__init__()
        self.auto_recompute = auto_recompute

        q_sampling_schema_in = Q_SamplingSchema.from_q_sampling_schema(q_sampling_schema_in)
        self.q_sampling_schema_in = q_sampling_schema_in
        self.Q_in = q_sampling_schema_in.Q

        type_in_out = SphericalTensorType.from_multiplicities_or_type(type_in_out)
        self.type_in_out = type_in_out

        if q_radial_basis_params is None:
            q_radial_basis_params = {}
        radial_basis_constructor = build_radial_basis_constructor(q_radial_basis_type, **q_radial_basis_params)
        self.radial_basis = radial_basis_constructor(q_sampling_schema_in.radial_basis_size,
                                                     q_sampling_schema_in.max_length)

        # Compute the mapping matrix that maps all representation indices of each channel to the channel
        # this matrix can later be used to apply radial basis and weights
        # Dim (type_in_out.dim x type_in_out.C)
        self.register_buffer('channel_mapping_matrix', compute_channel_mapping_matrix(type_in_out))

        self.register_buffer('lengths_Q_in', q_sampling_schema_in.q_lengths)  # (Q_in)

        # the learned weights
        # (type_in_out.C x radial_basis.number_of_basis)
        self.weights = torch.nn.Parameter(torch.randn(type_in_out.C, self.radial_basis.basis_size), requires_grad=True)

        # the final weights applied to the input
        self.computed_weights = None

    def recompute(self):
        radial_basis_values = self.radial_basis(self.lengths_Q_in)  # (Q_in, radial_basis.number_of_basis)

        # (dim_in_out x Q_in)
        self.computed_weights = torch.einsum('dc,cr,qr->dq', self.channel_mapping_matrix,
                                             self.weights / ((self.radial_basis.basis_size * self.Q_in) ** 0.5),
                                             radial_basis_values)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies the layer to input feature map x.

		:param x: Dim (N x dim_in_out x Q_in x P_z x P_y x P_x)
            - N: batch size
            - dim_in_out: size of the spherical tensor at each point of the input/output feature map.
            - Q_in: size of the input q-space sampling schema.
            - P_z, P_y, P_x: p-space size.
        :return: Dim (N x dim_in_out x P_z x P_y x P_x)
             N: batch size
            - dim_in_out: size of the spherical tensor at each point of the input/output feature map.
            - P_z, P_y, P_x: p-space size (same as input).
        """
        if self.auto_recompute:
            self.recompute()
        assert self.computed_weights is not None, \
            'If auto_recompute=False, then recompute() needs to be called ' \
            'before using this layer the first time.'
        assert x.ndim == 6
        assert x.size(1) == self.type_in_out.dim, f'Expected as total of {self.type_in_out.dim} channels but was {x.size(1)}'
        assert x.size(2) == self.Q_in

        return torch.einsum('ndqzyx,dq->ndzyx', x, self.computed_weights)
