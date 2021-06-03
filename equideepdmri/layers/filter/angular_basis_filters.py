from functools import partial
from typing import List, Callable, Optional

import torch
import torch.nn.functional as F

from equideepdmri.utils.spherical_tensor import SphericalTensor, SphericalTensorType
from equideepdmri.utils.q_space import Q_SamplingSchema
from equideepdmri.layers.filter.utils import normalized_sh, tensor_product_out, selection_rule_out, SelectionRuleOutInterface


class AngularKernelBasis:
    """
    Angular part of the filter basis.
    """
    kernel: SphericalTensor  # Dim (num_non_zero_QP x kernel.type.dim)

    # Mask that is True for indices where the kernel is non_zero
    # Note: len(kernel_mask.nonzero()) == num_non_zero_QP
    kernel_mask: torch.BoolTensor  # Dim (Q_out x Q_in x num_P_diff_vectors)

    # type of the kernel, for debugging only
    kernel_type_name: str

    def __init__(self, kernel: SphericalTensor, kernel_mask: torch.BoolTensor, kernel_type_name: str):
        """
        :param kernel: The kernel itself. Dim (num_non_zero_QP x kernel.type.dim)
        :param kernel_mask: Mask that is True for indices where the kernel is non_zero
            (len(kernel_mask.nonzero()) == num_non_zero_QP).
            Dim (Q_out x Q_in x num_P_diff_vectors)
        :param kernel_type_name: type of the kernel, for debugging only
        """
        assert isinstance(kernel, SphericalTensor)
        self.kernel = kernel
        self.kernel_mask = kernel_mask
        self.kernel_type_name = kernel_type_name

    def check_validity(self, Q_out: int, Q_in: int, num_P_diff_vectors: int, kernel_name: str = 'kernel'):
        """
        Checks that the dimensions of the angular kernel are valid for given values.

		:param Q_out: q-sampling schema size of output feature map.
        :param Q_in: q-sampling schema size of input feature map.
        :param num_P_diff_vectors: number of different p-differences.
        :param kernel_name: Name of the kernel (for debug).
        """
        assert self.kernel.value.size()[0] == len(self.kernel_mask.nonzero()), \
            f'kernel data and kernel_mask of "{kernel_name}" do not fit together.' \
            f'kernel data contains {self.kernel.value.size()[0]} elements but' \
            f'kernel_mask has {len(self.kernel_mask.nonzero())} True elements.'
        assert self.kernel_mask.size() == (Q_out, Q_in, num_P_diff_vectors), \
            f'Expected kernel_mask of "{kernel_name}" to have size {(Q_out, Q_in, num_P_diff_vectors)} ' \
            f'but size was {self.kernel_mask.size()}'

    def __repr__(self):
        return f'<AngularKernel {self.kernel_type_name} of type {self.kernel.type.multiplicities}>'



AngularKernelBasisConstructor = Callable[[List[int], Optional[Q_SamplingSchema], Optional[Q_SamplingSchema], torch.Tensor, float], AngularKernelBasis]


def TP_AngularKernel(ls: List[int],
                     Q_sampling_schema_out: Optional[Q_SamplingSchema],
                     Q_sampling_schema_in: Optional[Q_SamplingSchema],
                     P_diff_vectors: torch.Tensor,
                     length_eps: float,
                     kernel_fn_1: AngularKernelBasisConstructor, kernel_fn_2: AngularKernelBasisConstructor,
                     selection_rule: SelectionRuleOutInterface = selection_rule_out(),
                     normalization='component') -> AngularKernelBasis:
        """
        Function interface for creation of angular kernels.

        :param l_s: Filter orders l for which to create the kernel.
        :param Q_sampling_schema_out: q-sampling schema of output feature map.
        :param Q_sampling_schema_in: q-sampling schema size of input feature map
        :param P_diff_vectors: p-difference vectors. Dim (num_P_diff_vectors x 3)
        :param length_eps: epsilon for considering vector lengths as non-zero if length > length_eps.
        :return:
        """
        Q_out = Q_sampling_schema_out.Q if Q_sampling_schema_out is not None else 1
        Q_in = Q_sampling_schema_in.Q if Q_sampling_schema_in is not None else 1
        num_P_diff_vectors, _ = P_diff_vectors.size()

        # TP (SparseTensor): Dim (kernel_type.dim x (type_kernel_1.dim * type_kernel_2.dim))
        (ls_kernel_1, ls_kernel_2, kernel_type), TP = tensor_product_out(ls, selection_rule, normalization=normalization)
        assert kernel_type.ls == ls, f'The selection_rule did not return pairs for all of the following ls: {ls}'
        type_kernel_1 = SphericalTensorType.from_ls(ls_kernel_1)
        type_kernel_2 = SphericalTensorType.from_ls(ls_kernel_2)
        # (kernel_type.dim x type_kernel_1.dim x type_kernel_2.dim)
        TP = TP.to_dense().reshape(kernel_type.dim, type_kernel_1.dim, type_kernel_2.dim)

        # (num_Q_diff_vectors x num_P_diff_vectors x M_all_f_1)
        kernel_1 = kernel_fn_1(ls_kernel_1, Q_sampling_schema_out, Q_sampling_schema_in, P_diff_vectors, length_eps)
        kernel_1.check_validity(Q_out, Q_in, num_P_diff_vectors, 'kernel_1')
        assert kernel_1.kernel.type == type_kernel_1, \
            f'Currently only kernels with at most 1 channel per order l can combined using TP:' \
            f'Expected kernel_1 to be of type {type_kernel_1} but type was {kernel_1.kernel.type}'

        # (num_Q_diff_vectors x num_P_diff_vectors x M_all_f_2)
        kernel_2 = kernel_fn_2(ls_kernel_2, Q_sampling_schema_out, Q_sampling_schema_in, P_diff_vectors, length_eps)
        kernel_2.check_validity(Q_out, Q_in, num_P_diff_vectors, 'kernel_2')
        assert kernel_2.kernel.type == type_kernel_2, \
            f'Currently only kernels with at most 1 channel per order l can combined using TP, ' \
            f'Expected kernel_2 to be of type {type_kernel_2} but type was {kernel_2.kernel.type}'

        # (num_Q_diff_vectors x num_P_diff_vectors)
        kernel_mask = kernel_1.kernel_mask & kernel_2.kernel_mask

        # --- only select the elements of kernel 1 and 2 that are also in the final kernel mask ---
        # (num_non_zero_QP_pairs x type_kernel_1.dim)
        kernel_data_1 = kernel_1.kernel.value[kernel_mask[kernel_1.kernel_mask]]
        # (num_non_zero_QP_pairs x type_kernel_2.dim)
        kernel_data_2 = kernel_2.kernel.value[kernel_mask[kernel_2.kernel_mask]]


        # (num_non_zero_QP_pairs {x} x kernel_type.dim {k})
        # {i}: type_kernel_1.dim, {j}: type_kernel_2.dim
        kernel_data = torch.einsum('kij,xi,xj->xk', TP, kernel_data_1, kernel_data_2)

        return AngularKernelBasis(kernel=SphericalTensor(kernel_data, type=kernel_type), kernel_mask=kernel_mask,
                                  kernel_type_name=f'({kernel_1.kernel_type_name} x {kernel_2.kernel_type_name})')


def TP_AngularKernelConstructor(kernel_fn_1: AngularKernelBasisConstructor, kernel_fn_2: AngularKernelBasisConstructor,
                                selection_rule: SelectionRuleOutInterface = selection_rule_out(),
                                normalization='component') -> AngularKernelBasisConstructor:
    return partial(TP_AngularKernel, kernel_fn_1=kernel_fn_1, kernel_fn_2=kernel_fn_2,
                   selection_rule=selection_rule,
                   normalization=normalization)


def SH_P_AngularKernel(ls: List[int],
                       Q_sampling_schema_out: Optional[Q_SamplingSchema],
                       Q_sampling_schema_in: Optional[Q_SamplingSchema],
                       P_diff_vectors: torch.Tensor,
                       length_eps: float,
                       normalization='component') -> AngularKernelBasis:
    """
    Angular Kernel based on.

	:param ls: List of ranks for which to create the angular kernel
        in increasing order
    :param P_diff_vectors: Dim (num_P_diff_vectors, 3)
    :param Q_diff_vectors: Dim (num_Q_diff_vectors, 3)
    :return Dim (num_Q_diff_vectors x num_P_diff_vectors x M_all)
        where Y.type.dim = sum((2*l + 1) for l in ls)
    """
    Q_out = Q_sampling_schema_out.Q if Q_sampling_schema_out is not None else 1
    Q_in = Q_sampling_schema_in.Q if Q_sampling_schema_in is not None else 1
    num_P_diff_vectors, _ = P_diff_vectors.size()

    P_diff_lengths = P_diff_vectors.norm(p=2, dim=1)  # (num_P_diff_vectors)
    kernel_mask = P_diff_lengths > length_eps  # (num_P_diff_vectors)

    # Note that normalized_sh (or also sh) is invariant to length of P_diff_vectors
    Y = normalized_sh(ls, P_diff_vectors[kernel_mask], normalization=normalization)  # (num_nonzero_P_diff_vectors x Y.type.dim)

    # ((num_Q_diff_vectors * num_nonzero_P_diff_vectors) x Y.type.dim)
    Y.value = Y.value.unsqueeze(0).expand(Q_out * Q_in, -1, Y.type.dim).reshape(-1, Y.type.dim)
    # (Q_out x Q_in x num_P_diff_vectors)
    kernel_mask = kernel_mask.unsqueeze(0).unsqueeze(1).expand(Q_out, Q_in, num_P_diff_vectors)

    return AngularKernelBasis(kernel=Y, kernel_mask=kernel_mask, kernel_type_name='Y(p)')


def SH_P_AngularKernelConstructor(normalization='component') -> AngularKernelBasisConstructor:
    return partial(SH_P_AngularKernel, normalization=normalization)


def SH_Q_AngularKernel(ls: List[int],
                       Q_sampling_schema_out: Optional[Q_SamplingSchema],
                       Q_sampling_schema_in: Optional[Q_SamplingSchema],
                       P_diff_vectors: torch.Tensor,
                       length_eps: float,
                       normalize_Q_before_diff=True,
                       normalization='component') -> AngularKernelBasis:
    """
    Angular Kernel based on.

	:param ls: List of ranks for which to create the angular kernel
        in increasing order
    :param P_diff_vectors: Dim (num_P_diff_vectors, 3)
    :param Q_diff_vectors: Dim (num_Q_diff_vectors, 3)
    :return Dim (num_Q_diff_vectors x num_P_diff_vectors x M_all)
        where M_all = sum((2*l + 1) for l)
    """
    Q_diff_vectors = compute_Q_diff_vectors(Q_sampling_schema_out, Q_sampling_schema_in,
                                            normalize_before_diff=normalize_Q_before_diff)
    num_Q_diff_vectors, _ = Q_diff_vectors.size()
    Q_out = Q_sampling_schema_out.Q if Q_sampling_schema_out is not None else 1
    Q_in = Q_sampling_schema_in.Q if Q_sampling_schema_in is not None else 1
    num_P_diff_vectors, _ = P_diff_vectors.size()

    Q_diff_lengths = Q_diff_vectors.norm(p=2, dim=1)  # (num_Q_diff_vectors)
    kernel_mask = Q_diff_lengths > length_eps  # (num_Q_diff_vectors)

    # Note that normalized_sh (or also sh) is invariant to length of Q_diff_vectors
    Y = normalized_sh(ls, Q_diff_vectors[kernel_mask], normalization=normalization)  # (num_nonzero_Q_diff_vectors x Y.type.dim)

    # ((num_nonzero_Q_diff_vectors * num_P_diff_vectors) x Y.type.dim)
    Y.value = Y.value.unsqueeze(1).expand(-1, num_P_diff_vectors, Y.type.dim).reshape(-1, Y.type.dim)
    # (Q_out x Q_in x num_P_diff_vectors)
    kernel_mask = kernel_mask.unsqueeze(1).expand(-1, num_P_diff_vectors)\
        .view(Q_out, Q_in, num_P_diff_vectors)

    return AngularKernelBasis(kernel=Y, kernel_mask=kernel_mask, kernel_type_name='Y(q)')


def SH_Q_AngularKernelConstructor(normalize_Q_before_diff=True, normalization='component') -> AngularKernelBasisConstructor:
    return partial(SH_Q_AngularKernel, normalize_Q_before_diff=normalize_Q_before_diff, normalization=normalization)


def SH_PQDiff_AngularKernel(ls: List[int],
                            Q_sampling_schema_out: Optional[Q_SamplingSchema],
                            Q_sampling_schema_in: Optional[Q_SamplingSchema],
                            P_diff_vectors: torch.Tensor,
                            length_eps: float,
                            normalize_Q_before_diff=True,
                            normalization='component') -> AngularKernelBasis:
    """
    Angular Kernel based on.

	:param ls: List of ranks for which to create the angular kernel
        in increasing order
    :param P_diff_vectors: Dim (num_P_diff_vectors, 3)
    :param Q_diff_vectors: Dim (num_Q_diff_vectors, 3)
    :return Dim (num_Q_diff_vectors x num_P_diff_vectors x M_all)
        where M_all = sum((2*l + 1) for l)
    """
    Q_diff_vectors = compute_Q_diff_vectors(Q_sampling_schema_out, Q_sampling_schema_in,
                                            normalize_before_diff=normalize_Q_before_diff)
    num_Q_diff_vectors, _ = Q_diff_vectors.size()
    Q_out = Q_sampling_schema_out.Q if Q_sampling_schema_out is not None else 1
    Q_in = Q_sampling_schema_in.Q if Q_sampling_schema_in is not None else 1
    num_P_diff_vectors, _ = P_diff_vectors.size()

    # (num_Q_diff_vectors x num_P_diff_vectors x 3)
    PQ_differences = P_diff_vectors.unsqueeze(0) - Q_diff_vectors.unsqueeze(1)

    PQ_diff_lengths = PQ_differences.norm(p=2, dim=2)  # (num_Q_diff_vectors x num_P_diff_vectors)
    kernel_mask = PQ_diff_lengths > length_eps  # (num_Q_diff_vectors x num_P_diff_vectors)

    # Note that normalized_sh (or also sh) is invariant to length of PQ_differences
    Y = normalized_sh(ls, PQ_differences[kernel_mask], normalization=normalization)  # (num_nonzero_PQ_differences x Y.type.dim)

    # (Q_out x Q_in x num_P_diff_vectors)
    kernel_mask = kernel_mask.view(Q_out, Q_in, num_P_diff_vectors)

    return AngularKernelBasis(kernel=Y, kernel_mask=kernel_mask, kernel_type_name='Y(p-q)')


def SH_PQDiff_AngularKernelConstructor(normalize_Q_before_diff=True, normalization='component') -> AngularKernelBasisConstructor:
    return partial(SH_PQDiff_AngularKernel, normalize_Q_before_diff=normalize_Q_before_diff,
                   normalization=normalization)


def compute_Q_diff_vectors(q_schema_out: Q_SamplingSchema, q_schema_in: Q_SamplingSchema,
                           normalize_before_diff=True) -> torch.Tensor:
    if q_schema_out is not None and q_schema_in is not None:
        q_vectors_out = q_schema_out.q_vectors
        q_vectors_in = q_schema_in.q_vectors
        if normalize_before_diff:
            q_vectors_out = F.normalize(q_vectors_out, p=2, dim=1)
            q_vectors_in = F.normalize(q_vectors_in, p=2, dim=1)

        # ((Q_out * Q_in) x 3)
        return (q_vectors_out.unsqueeze(1) - q_vectors_in.unsqueeze(0)).view(-1, 3)

    elif q_schema_in is not None:
        # ((1 * Q_in) x 3)
        q_vectors_in = q_schema_in.q_vectors
        if normalize_before_diff:
            q_vectors_in = F.normalize(q_vectors_in, p=2, dim=1)
        return q_vectors_in

    elif q_schema_out is not None:
        # ((Q_out * 1) x 3)
        q_vectors_out = q_schema_out.q_vectors
        if normalize_before_diff:
            q_vectors_out = F.normalize(q_vectors_out, p=2, dim=1)
        return q_vectors_out
    else:
        raise ValueError('Neither input nor output contains q-space dimension: '
                         'No kernels working on q-space can be used in that case!')
