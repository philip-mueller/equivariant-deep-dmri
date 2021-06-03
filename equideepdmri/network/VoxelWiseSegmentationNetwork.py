from itertools import chain
from typing import List, Union

import torch
from torch import nn
from torch.utils.checkpoint import CheckpointFunction

from equideepdmri.layers.Recomputeable import recompute
from equideepdmri.layers.layer_builders import build_pq_layer, build_p_layer, build_q_reduction_layer
from equideepdmri.utils.q_space import Q_SamplingSchema
from equideepdmri.utils.spherical_tensor import SphericalTensorType


class VoxelWiseSegmentationNetwork(nn.Module):
    """
    Network for voxel-wise segmentation on dMRI data.
    It is equivariant to joint rotations in p- and q-space and to translations in p-space.
    The input are scans with p- and q-space, the outputs are scores for each position in p-space.

    First all q=0 channels are combined using the mean of them.
    Then it is processed by EquivariantPQLayer s (called pq-layers), then q-space
    is reduced (called q-reduction) leaving only p-space, then the data is processed
    by EquivariantPLayer s (called p-layers).

    Input: (N x Q_in x P_z x P_y x P_x)
    Output: (N x P_z x P_y x P_x)
    """
    def __init__(self,
                 q_sampling_schema_in: Q_SamplingSchema,
                 pq_channels: List[List[int]],
                 p_channels: List[List[int]],
                 kernel_sizes: Union[int, List[int]],
                 pq_kernel: dict,
                 p_kernel: dict = None,
                 non_linearity=None,
                 q_sampling_schemas=None,
                 q_reduction=None,
                 auto_recompute=True,
                 checkpointing=False):
        """
        Network for voxel-wise segmentation on dMRI data.
        It is equivariant to joint rotations in p- and q-space and to translations in p-space.
        The input are scans with p- and q-space, the outputs are scores for each position in p-space.

        First all q=0 channels are combined using the mean of them.
        Then it is processed by EquivariantPQLayer s (called pq-layers), then q-space
        is reduced (called q-reduction) leaving only p-space, then the data is processed
        by EquivariantPLayer s (called p-layers).

        :param q_sampling_schema_in: The q-sampling schema of the scans.
            All scans are expected to have the same q-sampling schema but if the schema slightly
            differs between samples, the mean sampling schema of all (training) samples can be used.
            Note that Q_in is not explicitly given but derived form the length of this parameter.
        :param pq_channels: The number of output channels (of each tensor order) for the pq-layers.
            The number of pq-layers is determined by the length of this list.
            For each pq-layer this list contains the spherical tensor type of the output feature map.
            (The spherical tensor type defines how many channels of each tensor order the output feature map has
            It is given as List[int], the element at index i of the list
            defines the number of order-i channels,
            e.g. the first element defines the number of order-0 (scalar) channels
            and the second the number of order-1 (vector) channels and so on.
            For all orders corresponding to out-of-range indices the number of channels is 0.)
        :param p_channels: The number of output channels (of each tensor order) for the p-layers.
            The number of p-layers is determined by the length of this list.
            For each p-layer this list contains the spherical tensor type of the output feature map.
            (The spherical tensor type defines how many channels of each tensor order the output feature map has
            It is given as List[int], the element at index i of the list
            defines the number of order-i channels,
            e.g. the first element defines the number of order-0 (scalar) channels
            and the second the number of order-1 (vector) channels and so on.
            For all orders corresponding to out-of-range indices the number of channels is 0.)
            If p_channels = None, then no p-layers are used.
        :param kernel_sizes: The kernel sizes (in p-space) of all pq- and p-layers.
            If this param is a single int, then the same kernel size is used for all layers.
            Else a list of integers is expected, containing one value (the kernel size)
            for each pq- and each p-layer.
        :param pq_kernel: Dict containing parameters for the kernel of the pq-layers
            (only an entry for the key "kernel" is required):

            - kernel (required): Which filter basis to use in this layer.
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

            - kernel_selection_rule: Rule defining which angular filter orders (l_filter) to use
              for a paths form input orders l_in to output orders l_out.
              Defaults to using all possible filter orders,
              i.e. all l_filter with  |l_in - l_out | <= l_filter <= l_in + l_out.
              Options are:

              - dict with key "lmax" and int value which additionally defines a maximum l_filter.
              - dict with int-pairs as keys and list of ints as values that defines
                for each pair of l_in and l_out the list of l_filter to use.
                E.g. {(0,0): [0], (1,1): [0,1], (0,1): [1]}

            - p_radial_basis_type: The radial basis function type used for p-space.
              Valid options are "gaussian" (default), "cosine", "bessel".
              Note that this parameter is ignored if there is no basis filter using p-space.
            - p_radial_basis_params: A (optional) dict of additional parameters for the radial basis function used for p-space.
              Valid keys in this dict are:

              - num_layers: Number of layers in the FC applied to the radial basis function.
                If num_layers = 0 (default) then no FC is applied to the radial basis function.
              - num_units: Number of units (neurons) in each of the layer in the FC applied to the radial basis function.
                No default, this parameter is required and must be >0 if num_layers > 0.
              - activation_function: activation function used in the FC applied to the radial basis function,
                valid are "relu" (default) or "swish"

              Note that this parameter is ignored if there is no basis filter using p-space.
            - q_radial_basis_type: The radial basis function type used for q-space (q-in and q-out).
              Valid options are "gaussian" (default), "cosine", "bessel".
              Note that this parameter is ignored if there is no basis filter using q-space.
            - q_out_radial_basis_type: The radial basis function type used for q-out (q-space of output feature map).
              See q_radial_basis_type but only for q-out.
              Defaults to q_radial_basis_type.
            - q_in_radial_basis_type: The radial basis function type used for q-in (q-space of input feature map).
              See q_radial_basis_type but only for q-in.
              Defaults to q_radial_basis_type.
            - q_radial_basis_params: A (optional) dict of additional parameters for the radial basis function used for q-space.
              Valid keys in this dict are:

              - num_layers: Number of layers in the FC applied to the radial basis function.
                If num_layers = 0 (default) then no FC is applied to the radial basis function.
              - num_units: Number of units (neurons) in each of the layer in the FC applied to the radial basis function.
                No default, this parameter is required and must be >0 if num_layers > 0.
              - activation_function: activation function used in the FC applied to the radial basis function,
                valid are "relu" (default) or "swish"

              Note that this parameter is ignored if there is no basis filter using q-space.
            - q_out_radial_basis_params: A dict of additional parameters for the radial basis function used for q-out (q-space of output feature map).
              See q_radial_basis_params but only for q-out.
              Defaults to q_radial_basis_params.
            - q_in_radial_basis_params: A dict of additional parameters for the radial basis function used for q-in (q-space of input feature map).
              See q_radial_basis_params but only for q-in.
              Defaults to q_radial_basis_params.
            - sub_kernel_selection_rule:
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

            For more params see layers.EquivariantPQLayer
        :param p_kernel: Dict containing parameters for the kernel of the pq-layers (optional):
            
            - kernel_selection_rule: Rule defining which angular filter orders (l_filter) to use
              for a paths form input orders l_in to output orders l_out.
              Defaults to using all possible filter orders,
              i.e. all l_filter with  |l_in - l_out | <= l_filter <= l_in + l_out.
              Options are:
            
              - dict with key "lmax" and int value which additionally defines a maximum l_filter.
              - dict with int-pairs as keys and list of ints as values that defines
                for each pair of l_in and l_out the list of l_filter to use.
                E.g. {(0,0): [0], (1,1): [0,1], (0,1): [1]}
            
            - p_radial_basis_type: The radial basis function type used for p-space.
              Valid options are "gaussian" (default), "cosine", "bessel".
            - p_radial_basis_params: A (optional) dict of additional parameters for the radial basis function used for p-space.
              Valid keys in this dict are:
            
              - num_layers: Number of layers in the FC applied to the radial basis function.
                If num_layers = 0 (default) then no FC is applied to the radial basis function.
              - num_units: Number of units (neurons) in each of the layer in the FC applied to the radial basis function.
                No default, this parameter is required and must be >0 if num_layers > 0.
              - activation_function: activation function used in the FC applied to the radial basis function,
                valid are "relu" (default) or "swish"
        
        :param non_linearity: Dict with the following optional keys:
        
            - tensor_non_lin: The nonlinearity to use for channels with l>0 (non-scalar channels).
              Default (and currently only option) is "gated".
            - scalar_non_lin: The nonlinearity to use for channles with l=0 (scalar channels).
              Valid options are "swish" and "relu".
              Default is "swish".
        
        :param q_sampling_schemas: List of output q_sampling schemes of the pq-layers
            or None if q_sampling_schema_in should be used in the output of all pq-layers.
            If not None, then the length must equal the length of pq_channels.
        :param q_reduction: Dict configuring the q-reduction layer with the following keys:
            
            - reduction: The type of reduction to use. Valid options are:
            
              - length_weighted_average: To use QLengthWeightedAvgPool (global length-weighted avg-pooling over q-space)
                For additional keys in the dict see QLengthWeightedAvgPool
                (except the params type_in, q_sampling_schema_in, auto_recompute).
              - mean: To use global avg-pooling over q-space.
              - conv: To use an EquivariantPQLayer (and gated nonlinearity) without output q-space.
                For additional keys in the dict see build_pq_layer
                (except the params type_in, type_out, q_sampling_schema_in, q_sampling_schema_out, auto_recompute).
        
        :param auto_recompute: Whether to automatically recompute the kernels in each forward pass.
            By default it is recomputed each time.
            If this parameter is set to False, it is not recomputed and the method recompute() needs to be called
            explicitly after parameters of this nn.Module have been updated.
        :param checkpointing: Whether to use use gradient checkpointing (default False).
            If True, checkpointing is applied after each layer,
            reducing the required memory but increasing computation time.
        """
        super().__init__()

        self.input_sampling_schema = q_sampling_schema_in

        q_sampling_schema_in = q_sampling_schema_in.sampling_schema_for_combine_b0_channels()
        if q_sampling_schemas is None:
            # default: same sampling schemas for all layers
            q_sampling_schemas = [q_sampling_schema_in for _ in range(len(pq_channels))]
        assert len(q_sampling_schemas) == len(pq_channels)

        if p_channels is None:
            p_channels = []
        if p_kernel is None:
            p_kernel = {}

        if isinstance(kernel_sizes, int):
            # default: same kernel sizes for all layers
            kernel_sizes = [kernel_sizes for _ in chain(pq_channels, p_channels)]
        assert len(kernel_sizes) == len(pq_channels) + len(p_channels)

        type_in = SphericalTensorType.from_multiplicities([1])  # input only has a single scalar channel
        self.type_in = type_in
        pq_layers = []
        for i, (type_out, q_sampling_schema_out, kernel_size) in \
                enumerate(zip(pq_channels, q_sampling_schemas, kernel_sizes[:len(pq_channels)])):
            type_out = SphericalTensorType.from_multiplicities_or_type(type_out)
            use_non_linearity = len(p_channels) > 0 or i < len(pq_channels) - 1  # last layer has no non-linearity
            pq_layers.append(build_pq_layer(type_in, type_out, kernel_size,
                                            q_sampling_schema_in=q_sampling_schema_in,
                                            q_sampling_schema_out=q_sampling_schema_out,
                                            non_linearity_config=non_linearity,
                                            use_non_linearity=use_non_linearity,
                                            auto_recompute=auto_recompute,
                                            **pq_kernel))

            type_in = type_out
            q_sampling_schema_in = q_sampling_schema_out

        self.pq_layers = nn.ModuleList(pq_layers)
        self.q_reduction_layer, type_in = build_q_reduction_layer(type_in, q_sampling_schema_in,
                                                                  auto_recompute=auto_recompute,
                                                                  **q_reduction)

        p_layers = []
        for i, (type_out, kernel_size) in enumerate(zip(p_channels, kernel_sizes[len(pq_channels):])):
            type_out = SphericalTensorType.from_multiplicities_or_type(type_out)
            use_non_linearity = i < len(p_channels) - 1  # last layer has no non-linearity
            p_layers.append(build_p_layer(type_in, type_out, kernel_size,
                                          non_linearity_config=non_linearity,
                                          use_non_linearity=use_non_linearity,
                                          auto_recompute=auto_recompute,
                                          **p_kernel))

            type_in = type_out
        self.type_out = type_in
        assert self.type_out == [1], f'Currently only output type [1] is supported, but type was {self.type_out}'
        self.p_layers = nn.ModuleList(p_layers)

        self.checkpointing = checkpointing

    def recompute(self):
        self.apply(recompute)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies the network to the input scan batch.

		:param x: Input feature map. Dim (N x Q_in x P_z x P_y x P_x) with
            - N: batch size
            - Q_in: size of the input q-space sampling schema.
            - P_z, P_y, P_x: p-space size.
        :return: The prediction scores (without sigmoid applied). Dim (N x P_z x P_y x P_x) with
            - N: batch size
            - P_z, P_y, P_x: p-space size (same as input).
        """
        x = x.unsqueeze(1)  # introduce the channel dimension
        # combine all b=0 channels using mean, other channels are untouched
        x = self.input_sampling_schema.combine_b0_channels(x)

        # pq-layers
        for i, pq_layer in enumerate(self.pq_layers):
            if self.checkpointing and (len(self.p_layers) > 0 or i < len(self.pq_layers) - 1):
                x = checkpoint(pq_layer, x)
            else:
                x = pq_layer(x)

        # q-reduction layer
        x = self.q_reduction_layer(x)

        # p-layers
        for i, p_layer in enumerate(self.p_layers):
            if self.checkpointing and i < len(self.p_layers) - 1:
                x = checkpoint(p_layer, x)
            else:
                x = p_layer(x)

        return x.squeeze(1)  # remove channel dimension


def checkpoint(function, *args, **kwargs):
    preserve = kwargs.pop('preserve_rng_state', True)
    if kwargs:
        raise ValueError("Unexpected keyword arguments: " + ",".join(arg for arg in kwargs))

    if all(not arg.requires_grad for arg in args):
        dummy_tensor = torch.ones(1, requires_grad=True)
        args = tuple([dummy_tensor] + list(args))
        old_fn = function
        function = lambda dummy, *f_args: old_fn(*f_args)

    return CheckpointFunction.apply(function, preserve, *args)
