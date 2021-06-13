from typing import List, Tuple

import torch
from torch import nn

from equideepdmri.layers.filter.filter_kernel import KernelDefinitionInterface
from equideepdmri.utils.q_space import Q_SamplingSchema
from equideepdmri.utils.spherical_tensor import SphericalTensorType


class SumKernel(nn.Module):
    def __init__(self,
                 type_out: SphericalTensorType,
                 type_in: SphericalTensorType,
                 Q_sampling_schema_out: Q_SamplingSchema,
                 Q_sampling_schema_in: Q_SamplingSchema,
                 P_diff_vectors: torch.Tensor,
                 P_kernel_size: int,
                 kernel_definitions: List[KernelDefinitionInterface]):
        super().__init__()

        self.kernels = nn.ModuleList([kernel_constructor(type_out, type_in,
                                                         Q_sampling_schema_out, Q_sampling_schema_in,
                                                         P_diff_vectors, P_kernel_size)
                                      for kernel_constructor in kernel_definitions])

    def forward(self) -> torch.Tensor:
        """
        :return: kernel (Q_out x Q_in x num_P_diff_vectors x type_out.dim x type_in.dim)
        """
        # (N_kernels x Q_out x Q_in x num_P_diff_vectors x type_out.dim x type_in.dim)
        kernel_tensors = torch.stack([kernel() for kernel in self.kernels], dim=0)

        return kernel_tensors.sum(dim=0) / len(self.kernels)


class ConcatKernel(nn.Module):
    def __init__(self,
                 type_out: SphericalTensorType,
                 type_in: SphericalTensorType,
                 Q_sampling_schema_out: Q_SamplingSchema,
                 Q_sampling_schema_in: Q_SamplingSchema,
                 P_diff_vectors: torch.Tensor,
                 P_kernel_size: int,
                 kernel_definitions: List[Tuple[SphericalTensorType, KernelDefinitionInterface]]):
        """
        :param type_out:
        :param type_in:
        :param Q_sampling_schema_out:
        :param Q_sampling_schema_in:
        :param P_diff_vectors:
        :param P_kernel_size:
        :param kernel_constructors: list of pairs (kernel_type_out, kernel_constructor) each representing a kernel to be concatenated.
            Note that all kernel_type_out concatenated need to be the same as type_out.
        """
        super().__init__()

        result_type, self.concat_indices = SphericalTensorType.concat_tensor_types(*[kernel_type_out
                                                                                     for kernel_type_out, _
                                                                                     in kernel_definitions])

        assert result_type == type_out, f'The kernel output types ' \
            f'{[kernel_type_out for kernel_type_out, _ in kernel_definitions]} ' \
            f'cannot be concatenated to the type {type_out}'

        self.kernels = nn.ModuleList([kernel_definition(kernel_type_out, type_in,
                                                        Q_sampling_schema_out, Q_sampling_schema_in,
                                                        P_diff_vectors, P_kernel_size)
                                      for kernel_type_out, kernel_definition in kernel_definitions])

        self.Q_out = Q_sampling_schema_out.Q
        self.Q_in = Q_sampling_schema_in.Q
        self.P_diff_vectors = P_diff_vectors
        self.num_P_diff_vectors, _ = P_diff_vectors.size()

        self.type_out = type_out
        self.type_in = type_in

    def forward(self) -> torch.Tensor:
        """
        :return: kernel (Q_out x Q_in x num_P_diff_vectors x type_out.dim x type_in.dim)
        """
        result_kernel = self.P_diff_vectors.new_zeros((self.Q_out, self.Q_in, self.num_P_diff_vectors, self.type_out.dim, self.type_in.dim))

        kernel_tensors = [kernel() for kernel in self.kernels]
        for kernel_indices, kernel_tensor in zip(self.concat_indices, kernel_tensors):
            result_kernel[:, :, :, kernel_indices, :] = kernel_tensor

        return result_kernel


