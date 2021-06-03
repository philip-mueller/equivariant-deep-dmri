from typing import Callable, Optional, List, Tuple

import torch
from e3nn import o3
from e3nn.linear import KernelLinear
from e3nn.rs import simplify, dim, sort
from e3nn.util.sparse import register_sparse_buffer, get_sparse_buffer
from torch import nn
from torch_sparse import SparseTensor

from equideepdmri.layers.filter.angular_basis_filters import AngularKernelBasisConstructor
from equideepdmri.layers.filter.radial_basis_filters import RadialKernelBasisConstructor
from equideepdmri.layers.filter.utils import SelectionRuleInterface, selection_rule
from equideepdmri.utils.q_space import Q_SamplingSchema
from equideepdmri.utils.spherical_tensor import SphericalTensorType


class Kernel(nn.Module):
    def __init__(self,
                 type_out: SphericalTensorType,
                 type_in: SphericalTensorType,
                 Q_sampling_schema_out: Optional[Q_SamplingSchema],
                 Q_sampling_schema_in: Optional[Q_SamplingSchema],
                 P_diff_vectors: torch.Tensor,
                 P_kernel_size: int,
                 angular_kernel_constructor: AngularKernelBasisConstructor,
                 scalar_kernel_constructor: RadialKernelBasisConstructor,
                 zero_length_eps: float = 1e-6,
                 selection_rule: SelectionRuleInterface = selection_rule(),
                 use_linear_model_for_zero_length=True,
                 normalization='component'):
        super().__init__()

        # these are later required for recomputing the scalar kernels
        self.Q_sampling_schema_out = Q_sampling_schema_out
        self.Q_sampling_schema_in = Q_sampling_schema_in
        self.P_diff_vectors = P_diff_vectors
        self.P_kernel_size = P_kernel_size

        self.type_out = type_out
        self.type_in = type_in

        # The TP_filter_type is the filter type required to combine type_in with filter to type_out in a tensor product
        TP_filter_type = find_filter_type(type_in, type_out, selection_rule)

        # ---------- compute angular kernels for each l of filter type ----------
        # now the angular kernels are computed for the TP_filter type
        # - angular_kernel.kernel Dim (num_non_zero_QP x angular_kernel.kernel.type.dim)
        # - angular_kernel.kernel_mask: Dim (Q_out x Q_in x num_P_diff_vectors)
        angular_kernel = angular_kernel_constructor(TP_filter_type.ls, Q_sampling_schema_out, Q_sampling_schema_in,
                                                    P_diff_vectors, zero_length_eps)
        angular_kernel.check_validity(self.Q_out, self.Q_in, self.num_P_diff_vectors)
        assert not angular_kernel.kernel.value.requires_grad, 'Angular kernel cannot be trainable (may not contain any learnable parameters)'
        self.register_buffer('angular_kernel_tensor', angular_kernel.kernel.value)

        self.angular_kernel_type_name = angular_kernel.kernel_type_name
        filter_representations = []
        filter_multiplicities = []
        for l, (C_l_TP, C_l_angular) in enumerate(zip(TP_filter_type.multiplicities,
                                                      angular_kernel.kernel.type.multiplicities)):
            C_l = C_l_TP * C_l_angular
            filter_multiplicities.append(C_l)
            if C_l == 0:
                continue
            filter_representations.append((l, C_l, C_l_TP, C_l_angular))

        # type_filter may be equal to TP_filter_type
        # but it may also differ if the angular part of the filter has multiple channels
        self.type_filter = SphericalTensorType.from_multiplicities(filter_multiplicities)
        self.filter_representations = filter_representations

        # sparse tensor of dim ((type_out.dim * type_in.dim) x type_filter.dim)
        TP_mixing_matrix = compute_TP_mixing_matrix_for_filter(type_in, type_out,
                                                               angular_kernel.kernel.type, selection_rule,
                                                               normalization=normalization)
        register_sparse_buffer(self, 'TP_mixing_matrix', TP_mixing_matrix)


        # ---------- treat small (0) length vectors ----------
        if not angular_kernel.kernel_mask.all():
            assert not angular_kernel.kernel_mask.requires_grad, \
                'kernel_mask of angular kernel cannot be trainable, but requires_grad was True'
            self.register_buffer('kernel_mask', angular_kernel.kernel_mask)

            if use_linear_model_for_zero_length:
                # linear channel mixing (for l_in == l_out)  for the PQ-combinations were angular kernel is zero
                self.linear_kernel = KernelLinear(self.type_in.Rs, self.type_out.Rs)
            else:
                self.linear_kernel = None
        else:
            self.kernel_mask = None
            self.linear_kernel = None

        # ---------- scalar kernel ----------
        scalar_kernel = scalar_kernel_constructor(self.Q_sampling_schema_out, self.Q_sampling_schema_in,
                                                  self.P_diff_vectors, self.P_kernel_size)
        self.scalar_kernel_type_name = scalar_kernel.kernel_type_name
        self.scalar_basis_size = scalar_kernel.scalar_basis_size

        if len(list(scalar_kernel.parameters())) == 0:
            # => scalar kernel has no learnable parameters => it can already be precomputed

            scalar_kernel_tensor = scalar_kernel()

            assert not scalar_kernel_tensor.requires_grad, \
                'Scalar kernel has no registered parameters, so it should not be trainable but requires_grad was True'

            self.scalar_kernel = None
            self.register_buffer('scalar_kernel_tensor', scalar_kernel_tensor)

        else:

            # => scalar kernel has learnable parameters => register it as module and combine it later
            self.scalar_kernel = scalar_kernel
            self.scalar_kernel_tensor = None

        # ---------- weights ----------
        self.weights = torch.nn.Parameter(torch.randn(scalar_kernel.scalar_basis_size, self.type_filter.C),
                                          requires_grad=True)

    def _compute_kernel(self, scalar_kernel_tensor: torch.Tensor) -> torch.Tensor:
        """
        :param scalar_kernel_tensor: (Q_out x Q_in x num_P_diff_vectors x scalar_kernel.scalar_basis_size {s})
        :return: (num_non_zero_QP {n} x type_out.dim {o} x type_in.dim {i})
        """
        if self.kernel_mask is None:
            # (num_non_zero_QP x scalar_kernel.scalar_basis_size)
            scalar_kernel_tensor = scalar_kernel_tensor.view(-1, self.scalar_basis_size)
        else:
            # (num_non_zero_QP x scalar_kernel.scalar_basis_size)
            scalar_kernel_tensor = scalar_kernel_tensor[self.kernel_mask]

        # apply weights
        # (num_non_zero_QP {n} x type_filter.C {c})
        scalar_kernel_tensor = torch.einsum('ns,sc->nc',
                                            scalar_kernel_tensor,
                                            self.weights / (self.scalar_basis_size ** 0.5))
        # combine scalar with angular kernel
        # (num_non_zero_QP {n} x type_filter.dim)
        kernel = mul_scalar_angular_kernel(self.filter_representations, self.type_filter.dim,
                                           scalar_kernel_tensor, self.angular_kernel_tensor)
        # apply TP
        # ((type_out.dim * type_in.dim) x type_filter.dim)
        TP_mixing_matrix = get_sparse_buffer(self, "TP_mixing_matrix")
        kernel = TP_mixing_matrix @ kernel.T  # (type_out.dim * type_in.dim) x num_non_zero_QP)
        return kernel.T.reshape(-1, self.type_out.dim, self.type_in.dim)

    def forward(self) -> torch.Tensor:
        """
        :return: kernel (Q_out x Q_in x num_P_diff_vectors x type_out.dim x type_in.dim)
        """
        # ----- build the kernel by combining with scalar kernel
        if self.scalar_kernel_tensor is None:
            scalar_kernel_tensor = self.scalar_kernel()
        else:
            # scalar kernel had no leanrable parameters, TP_with_kernel could already be precomputed
            scalar_kernel_tensor = self.scalar_kernel_tensor
        TP_with_kernel = self._compute_kernel(scalar_kernel_tensor)

        if self.kernel_mask is None:
            # (Q_out x Q_in x num_P_diff_vectors x type_out.dim x type_in.dim)
            final_kernel = TP_with_kernel.view(self.Q_out, self.Q_in, self.num_P_diff_vectors, self.type_out.dim, self.type_in.dim)
        else:
            # (Q_out x Q_in x num_P_diff_vectors x type_out.dim x type_in.dim)
            final_kernel = TP_with_kernel.new_zeros(self.Q_out, self.Q_in, self.num_P_diff_vectors, self.type_out.dim, self.type_in.dim)
            final_kernel[self.kernel_mask] = TP_with_kernel

            if self.linear_kernel is not None:
                # note: self.linear_kernel() has Dim (self.type_out.dim, self.type_in.dim)
                # unsqueeze(0) so that the new dim is guaranteed to be broadcasted with the num_zero_QP from indexing by mask
                final_kernel[~self.kernel_mask] = self.linear_kernel().unsqueeze(0)

        return final_kernel

    @property
    def Q_in(self):
        return self.Q_sampling_schema_in.Q if self.Q_sampling_schema_in is not None else 1

    @property
    def Q_out(self):
        return self.Q_sampling_schema_out.Q if self.Q_sampling_schema_out is not None else 1

    @property
    def num_P_diff_vectors(self):
        return self.P_diff_vectors.size()[0]

    def __repr__(self):
        return f'<Kernel_PQ {self.scalar_kernel_type_name} * {self.angular_kernel_type_name} ' \
            f'of type {self.type_in.multiplicities} -> {self.type_out.multiplicities} ' \
            f'with basis size {self.type_filter.multiplicities} * {self.scalar_basis_size}>'


KernelDefinitionInterface = Callable[[SphericalTensorType, SphericalTensorType, Q_SamplingSchema, Q_SamplingSchema, torch.Tensor, int], nn.Module]


@torch.jit.script
def mul_scalar_angular_kernel(filter_representations: List[Tuple[int, int, int, int]],
                              filter_dim: int,
                              scalar_kernel_tensor: torch.Tensor, angular_kernel_tensor: torch.Tensor) -> torch.Tensor:
    """
    :param scalar_kernel_tensor: (num_non_zero_QP {n} x angular_filter_type.C {c})
        scalar kernel already combined with the weights
    :param angular_kernel_tensor: (num_non_zero_QP {n} x angular_kernel.kernel.type.dim)
    :return:(num_non_zero_QP {n} x filter_dim)
    """
    a = 0
    s = 0
    i = 0

    num_non_zero_QP = scalar_kernel_tensor.size(0)
    out = scalar_kernel_tensor.new_empty(num_non_zero_QP, filter_dim)

    for l, C_l, C_l_TP, C_l_angular in filter_representations:
        dim_angular = C_l_angular * (2 * l + 1)
        dim = C_l * (2 * l + 1)

        x = scalar_kernel_tensor[:, s: s+C_l].view(num_non_zero_QP, C_l_angular, C_l_TP, 1) \
            * angular_kernel_tensor[:, a: a + dim_angular].view(num_non_zero_QP, C_l_angular, 1, 2 * l + 1)
        x = x.view(num_non_zero_QP, dim)
        out[:, i: i + dim] = x

        i += dim
        s += C_l
        a += dim_angular

    assert s == scalar_kernel_tensor.size(-1)
    assert a == angular_kernel_tensor.size(-1)
    assert i == out.size(-1)

    return out


def find_filter_type(type_in: SphericalTensorType, type_out: SphericalTensorType,
                     selection_rule: SelectionRuleInterface) -> SphericalTensorType:
    Rs_filter = []
    for C_out, l_out in type_out.Rs:
        for C_in, l_in in type_in.Rs:
            for l_f in selection_rule(l_in, l_out):
                Rs_filter.append((C_in * C_out, l_f))
    return SphericalTensorType.from_Rs(Rs_filter)


def compute_TP_mixing_matrix_for_filter(type_in: SphericalTensorType, type_out: SphericalTensorType,
                                        type_angular_filter: SphericalTensorType,
                                        selection_rule: SelectionRuleInterface, normalization: str) -> SparseTensor:
    """
    Based on e3nn -> rs._tensor_product_in_out() but adapted to account for type_angular_filter

    :param type_in:
    :param type_out:
    :param type_angular_filter:
    :param normalization:
    :return: ((type_out.dim * type_in.dim) x type_filter.dim)
    """
    assert normalization in ['norm', 'component'], "normalization needs to be 'norm' or 'component'"

    Rs_in = simplify(type_in.Rs)
    Rs_out = simplify(type_out.Rs)

    Rs_f = []

    for C_out, l_out, _ in Rs_out:
        for C_in, l_in, _ in Rs_in:
            for l_f in selection_rule(l_in, l_out):
                C_f_angular = type_angular_filter.C_l(l_f)
                C_f = C_f_angular * C_out * C_in
                Rs_f.append((C_f, l_f, 0))

    Rs_f = simplify(Rs_f)

    dim_f_total = dim(Rs_f)
    row = []
    col = []
    val = []

    index_f = 0

    index_out = 0
    for C_out, l_out, _ in Rs_out:
        dim_out = C_out * (2 * l_out + 1)

        n_path = 0
        for C_in, l_in, _ in Rs_in:
            for l_f in selection_rule(l_in, l_out):
                n_path += C_in * type_angular_filter.C_l(l_f)

        index_in = 0
        for C_in, l_in, _ in Rs_in:
            dim_in = C_in * (2 * l_in + 1)
            for l_f in selection_rule(l_in, l_out):
                # compute part of mixing matrix for all paths from l_in using l_f to l_out
                #
                # how many different angular filters are used for each filter channel
                # in simple cases this is just 1
                # only if the angular filter is built as TP of multiple filter this may be >1
                C_f_angular = type_angular_filter.C_l(l_f)

                # filter channels for current l_f (C_f_angular channels for each path form in to out for current l)
                C_f = C_f_angular * C_out * C_in
                dim_f = C_f * (2 * l_f + 1)  # filter dim of current l_f
                C = o3.wigner_3j(l_out, l_in, l_f, cached=True)  # ((2*l_out + 1) x (2*l_in + 1) x (2*l_f + 1))
                if normalization == 'component':
                    C *= (2 * l_out + 1) ** 0.5
                if normalization == 'norm':
                    C *= (2 * l_in + 1) ** 0.5 * (2 * l_f + 1) ** 0.5

                # (C_out, C_in, C_f)
                I = torch.eye(C_out * C_in).reshape(C_out, C_in, C_out * C_in).repeat(1, 1, C_f_angular)
                I /= (n_path * C_f_angular) ** 0.5
                m = torch.einsum("wuv,kij->wkuivj", I, C).reshape(dim_out, dim_in, dim_f)

                i_out, i_1, i_2 = m.nonzero().T
                i_out += index_out
                i_1 += index_in
                i_2 += index_f
                row.append(i_out)
                col.append(i_1 * dim_f_total + i_2)
                val.append(m[m != 0])

                index_f += dim_f
            index_in += dim_in
        index_out += dim_out
    wigner_3j_tensor = SparseTensor(
        row=torch.cat(row) if row else torch.zeros(0, dtype=torch.long),
        col=torch.cat(col) if col else torch.zeros(0, dtype=torch.long),
        value=torch.cat(val) if val else torch.zeros(0),
        sparse_sizes=(dim(Rs_out), dim(Rs_in) * dim(Rs_f)))

    # sort to normalized Rs
    Rs_f, perm_mat = sort(Rs_f)
    Rs_f = simplify(Rs_f)
    wigner_3j_tensor = wigner_3j_tensor.sparse_reshape(-1, dim(Rs_f))
    wigner_3j_tensor = wigner_3j_tensor @ perm_mat.t()
    wigner_3j_tensor = wigner_3j_tensor.sparse_reshape(-1, dim(Rs_in) * dim(Rs_f))

    wigner_3j_tensor = wigner_3j_tensor.sparse_reshape(dim(Rs_out) * dim(Rs_in), dim(Rs_f))
    return wigner_3j_tensor
