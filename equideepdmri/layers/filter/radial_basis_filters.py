from functools import partial
from typing import Callable, Iterable, Optional

import torch
from torch import nn
import numpy as np

from equideepdmri.layers.filter.radial_basis_functions import RadialBasisConstructor
from equideepdmri.utils.q_space import Q_SamplingSchema


class RadialKernelBasis(nn.Module):
    def __init__(self, scalar_basis_size: int, kernel_type_name: str):
        super().__init__()

        self.scalar_basis_size = scalar_basis_size
        self.kernel_type_name = kernel_type_name

    def forward(self) -> torch.Tensor:
        """
        :return: Dim (Q_out x Q_in x num_P_diff_vectors x scalar_basis_size)
        """
        raise NotImplementedError

    def __repr__(self):
        return f'<ScalarKernel {self.kernel_type_name} of size {self.scalar_basis_size}>'


RadialKernelBasisConstructor = Callable[[Optional[Q_SamplingSchema], Optional[Q_SamplingSchema], torch.Tensor, int], RadialKernelBasis]


class Constant_RadialKernelBasis(RadialKernelBasis):
    def __init__(self,
                 Q_sampling_schema_out: Optional[Q_SamplingSchema],
                 Q_sampling_schema_in: Optional[Q_SamplingSchema],
                 P_diff_vectors: torch.Tensor,
                 P_kernel_size: int):
        super().__init__(scalar_basis_size=1, kernel_type_name='1')
        Q_out = Q_sampling_schema_out.Q if Q_sampling_schema_out is not None else 1
        Q_in = Q_sampling_schema_in.Q if Q_sampling_schema_in is not None else 1
        num_P_diff_vectors = P_diff_vectors.size()[0]

        self.register_buffer('constant_tensor', P_diff_vectors.new_ones((Q_out, Q_in, num_P_diff_vectors, 1)))

    def forward(self) -> torch.Tensor:
        """
        :return: (Q_out x Q_in x num_P_diff_vectors x 1)
        """
        return self.constant_tensor


Constant_ScalarKernelConstructor: RadialKernelBasisConstructor = Constant_RadialKernelBasis


class LengthQIn_RadialKernelBasis(RadialKernelBasis):
    def __init__(self,
                 Q_sampling_schema_out: Optional[Q_SamplingSchema],
                 Q_sampling_schema_in: Optional[Q_SamplingSchema],
                 P_diff_vectors: torch.Tensor,
                 P_kernel_size: int,
                 radial_basis_constructor: RadialBasisConstructor):
        assert Q_sampling_schema_in is not None
        radial_basis = radial_basis_constructor(Q_sampling_schema_in.radial_basis_size, Q_sampling_schema_in.max_length)
        # Note that radial_basis.basis_size may not be equal to Q_sampling_schema_in.radial_basis_size
        # This is the case if a RadialBasis with FC layers is used, then radial_basis.basis_size
        # is the number of units in the last layer
        super().__init__(scalar_basis_size=radial_basis.basis_size,
                         kernel_type_name=f'{radial_basis.radial_basis_type_name}(|q_in|)')

        self.radial_basis = radial_basis
        self.Q_out = Q_sampling_schema_out.Q if Q_sampling_schema_out is not None else 1
        self.Q_in = Q_sampling_schema_in.Q
        self.num_P_diff_vectors = P_diff_vectors.size()[0]
        self.register_buffer('lengths_Q_in', Q_sampling_schema_in.q_lengths)  # (Q_in)

    def forward(self) -> torch.Tensor:
        """
        :return: (Q_out x Q_in x num_P_diff_vectors x radial_basis.number_of_basis)
        """
        scalar_values = self.radial_basis(self.lengths_Q_in)  # (Q_in, radial_basis.number_of_basis)
        return scalar_values.unsqueeze(0).unsqueeze(2).expand(self.Q_out, self.Q_in, self.num_P_diff_vectors, -1)


def LengthQIn_ScalarKernelConstructor(radial_basis_constructor: RadialBasisConstructor) -> RadialKernelBasisConstructor:
    return partial(LengthQIn_RadialKernelBasis, radial_basis_constructor=radial_basis_constructor)


class LengthQOut_RadialKernelBasis(RadialKernelBasis):
    def __init__(self,
                 Q_sampling_schema_out: Optional[Q_SamplingSchema],
                 Q_sampling_schema_in: Optional[Q_SamplingSchema],
                 P_diff_vectors: torch.Tensor,
                 P_kernel_size: int,
                 radial_basis_constructor: RadialBasisConstructor):

        assert Q_sampling_schema_out is not None
        radial_basis = radial_basis_constructor(Q_sampling_schema_out.radial_basis_size, Q_sampling_schema_out.max_length)
        # Note that radial_basis.basis_size may not be equal to Q_sampling_schema_out.radial_basis_size
        # This is the case if a RadialBasis with FC layers is used, then radial_basis.basis_size
        # is the number of units in the last layer
        super().__init__(scalar_basis_size=radial_basis.basis_size,
                         kernel_type_name=f'{radial_basis.radial_basis_type_name}(|q_out|)')

        self.radial_basis = radial_basis
        self.Q_out = Q_sampling_schema_out.Q
        self.Q_in = Q_sampling_schema_in.Q if Q_sampling_schema_in is not None else 1
        self.num_P_diff_vectors = P_diff_vectors.size()[0]
        self.register_buffer('lengths_Q_out', Q_sampling_schema_out.q_lengths)  # (Q_out)

    def forward(self) -> torch.Tensor:
        """
        :return: (Q_out x Q_in x num_P_diff_vectors x radial_basis.number_of_basis)
        """
        scalar_values = self.radial_basis(self.lengths_Q_out)  # (Q_out, radial_basis.number_of_basis)
        return scalar_values.unsqueeze(1).unsqueeze(2).expand(self.Q_out, self.Q_in, self.num_P_diff_vectors, -1)


def LengthQOut_ScalarKernelConstructor(radial_basis_constructor: RadialBasisConstructor) -> RadialKernelBasisConstructor:
    return partial(LengthQOut_RadialKernelBasis, radial_basis_constructor=radial_basis_constructor)

class LengthPDiff_RadialKernelBasis(RadialKernelBasis):
    def __init__(self,
                 Q_sampling_schema_out: Optional[Q_SamplingSchema],
                 Q_sampling_schema_in: Optional[Q_SamplingSchema],
                 P_diff_vectors: torch.Tensor,
                 P_kernel_size: int,
                 radial_basis_constructor: RadialBasisConstructor):
        basis_size = (P_kernel_size + 1) // 2
        # max_radius = 1.0 for P kernel because of the way the kernel is sampled (values from -1.0 to 1.0)
        max_radius = 1.0
        radial_basis = radial_basis_constructor(basis_size, max_radius)
        # Note that radial_basis.basis_size may not be equal to basis_size
        # This is the case if a RadialBasis with FC layers is used, then radial_basis.basis_size
        # is the number of units in the last layer
        super().__init__(scalar_basis_size=radial_basis.basis_size,
                         kernel_type_name=f'{radial_basis.radial_basis_type_name}(|p|)')

        self.radial_basis = radial_basis
        self.Q_out = Q_sampling_schema_out.Q if Q_sampling_schema_out is not None else 1
        self.Q_in = Q_sampling_schema_in.Q if Q_sampling_schema_in is not None else 1
        self.num_P_diff_vectors = P_diff_vectors.size()[0]
        self.register_buffer('lengths_P_diff', torch.norm(P_diff_vectors, p=2, dim=1))  # (num_P_diff_vectors)

    def forward(self) -> torch.Tensor:
        """
        :return: (Q_out x Q_in x num_P_diff_vectors x radial_basis.number_of_basis)
        """
        scalar_values = self.radial_basis(self.lengths_P_diff)  # (num_P_diff_vectors, radial_basis.number_of_basis)
        return scalar_values.unsqueeze(0).unsqueeze(1).expand(self.Q_out, self.Q_in, self.num_P_diff_vectors, -1)


def LengthPDiff_ScalarKernelConstructor(radial_basis_constructor: RadialBasisConstructor) -> RadialKernelBasisConstructor:
    return partial(LengthPDiff_RadialKernelBasis, radial_basis_constructor=radial_basis_constructor)


_scalar_kernel_combination_einsum_formulas = {
    2: 'qrpi,qrpj->qrpij',
    3: 'qrpi,qrpj,qrpk->qrpijk',
    4: 'qrpi,qrpj,qrpk,qrpl->qrpijkl',
    5: 'qrpi,qrpj,qrpk,qrpl,qrpm->qrpijklm'
}

class CombinedRadialKernelBasis(RadialKernelBasis):
    def __init__(self,
                 Q_sampling_schema_out: Optional[Q_SamplingSchema],
                 Q_sampling_schema_in: Optional[Q_SamplingSchema],
                 P_diff_vectors: torch.Tensor,
                 P_kernel_size: int,
                 scalar_kernel_contructors: Iterable[RadialKernelBasisConstructor]):
        scalar_kernels = [scalar_kernel_contructor(Q_sampling_schema_out, Q_sampling_schema_in,
                                                   P_diff_vectors, P_kernel_size)
                          for scalar_kernel_contructor in scalar_kernel_contructors]
        num_scalar_kernels = len(scalar_kernels)
        assert num_scalar_kernels in _scalar_kernel_combination_einsum_formulas, \
            f'Trying to combine {num_scalar_kernels} scalar kernels ' \
            f'but currently only one of the following number of kernels can be combined: ' \
            f'{_scalar_kernel_combination_einsum_formulas.keys()}'
        scalar_basis_size = np.prod([scalar_kernel.scalar_basis_size for scalar_kernel in scalar_kernels])
        kernel_type_name = '(' + ' * '.join(scalar_kernel.kernel_type_name for scalar_kernel in scalar_kernels) + ')'
        super().__init__(scalar_basis_size=scalar_basis_size, kernel_type_name=kernel_type_name)

        self.scalar_kernels = nn.ModuleList(scalar_kernels)
        self.einsum_formula = _scalar_kernel_combination_einsum_formulas[num_scalar_kernels]
        self.Q_out = Q_sampling_schema_out.Q if Q_sampling_schema_out is not None else 1
        self.Q_in = Q_sampling_schema_in.Q if Q_sampling_schema_in is not None else 1
        self.num_P_diff_vectors = P_diff_vectors.size()[0]

    def forward(self) -> torch.Tensor:
        """
        Dim (Q_out x Q_in x num_P_diff_vectors x scalar_basis_size)
        
        :return:
        """
        scalar_kernel_tensors = [scalar_kernel() for scalar_kernel in self.scalar_kernels]

        return torch.einsum(self.einsum_formula, *scalar_kernel_tensors)\
            .view(self.Q_out, self.Q_in, self.num_P_diff_vectors, -1)


def CombinedScalarKernelConstructor(*scalar_kernel_contructors: Iterable[RadialKernelBasisConstructor]) \
        -> RadialKernelBasisConstructor:
    return partial(CombinedRadialKernelBasis, scalar_kernel_contructors=scalar_kernel_contructors)
