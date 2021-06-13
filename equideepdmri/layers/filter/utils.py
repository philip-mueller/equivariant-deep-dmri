import math
from functools import partial

import torch
from e3nn.non_linearities.rescaled_act import swish, relu
from collections import defaultdict
from typing import List, Tuple, Callable, Dict
from e3nn import rs
from e3nn.rsh import spherical_harmonics_xyz
from torch_sparse import SparseTensor

from equideepdmri.utils.spherical_tensor import SphericalTensorType, SphericalTensor


def get_scalar_non_linearity(name: str):
    return _scalar_non_linearities[name]


_scalar_non_linearities = {
    'swish': swish,
    'relu': relu
}


def eye_3d(dim: int) -> torch.Tensor:
    return torch.eye(dim).unsqueeze(0) * torch.eye(dim).unsqueeze(1)


def tensor_product_in_out(type_in_1: SphericalTensorType, type_out: SphericalTensorType,
                          selection_rule: Callable[[int, int], List[int]], normalization='component') \
        -> Tuple[SphericalTensorType, torch.Tensor]:
    """
    Note: based on o3.selection_rule_in_out_sh

    :param type_in_1: spherical tensor type of first input to TP
    :param type_out: spherical tensor type of output from TP
    :param selection_rule: selection rule for possible spherical tensor types of second input to TP
        selection_rule(l_1, l_out) -> l_2s
    :param normalization: how to normalize the TP
    :return: (type_in_2, C')
        - type_in_2: spherical tensor type of second input to TP
        - C': tensor describing the tensor product. Dim (type_out.dim x type_in_1.dim x type_in_2.dim)
    """
    selection_rule_adapted = lambda l_1, p_1, l_out, p_out: selection_rule(l_1, l_out)

    # (type_out.dim x type_in_1.dim x type_in_2.dim)
    Rs_in_2, C_ = rs.tensor_product(type_in_1.Rs, selection_rule_adapted, type_out.Rs, normalization, sorted=True)
    type_in_2 = SphericalTensorType.from_Rs(Rs_in_2)

    C_ = SparseTensor.to_dense(C_).reshape(type_out.dim, type_in_1.dim, type_in_2.dim)

    return type_in_2, C_


def tensor_product_out(ls_out: List[int], selection_rule: Callable[[int], List[Tuple[int, int]]], normalization='component') \
        -> Tuple[Tuple[List[int], List[int], SphericalTensorType], torch.Tensor]:
    """
    Creates the tensor C' (based on GC coeffs) that describes the tensorproduct between spherical tensors.
    The used input types are not given, only the wanted output orders and a selection rule for possible
    inputs order combinations for each output order.
    Then according to these values possible tensor products are found and C' is built for that purpose.
    The required input orders and the produced output type are also given.
    Note that the inputs tensors are assumed to have 1 channel for each required input order.
    The output type may have multiple channel for each output order, that is then based in different combinations
    of input orders.

    :param ls_out:
    :param selection_rule:
    :return:
    """

    l_s_in_1 = set()
    l_s_in_2 = set()
    possible_ls_out = defaultdict(set)
    for l_out in iter(ls_out):
        possible_ls_in = selection_rule(l_out)
        for l_in_1, l_in_2 in iter(possible_ls_in):
            l_s_in_1.add(l_in_1)
            l_s_in_2.add(l_in_2)
            possible_ls_out[(l_in_1, l_in_2)].add(l_out)

    possible_ls_out = {l_in_pair: list(sorted(l_s_out)) for l_in_pair, l_s_out in possible_ls_out.items()}
    selection_rule_out = lambda l_1, p_1, l_2, p_2: possible_ls_out.get((l_1, l_2), [])

    type_in_1 = SphericalTensorType.from_ls(l_s_in_1)
    type_in_2 = SphericalTensorType.from_ls(l_s_in_2)

    # (type_out.dim x type_in_1.dim x type_in_2.dim)
    Rs_out, TP = rs.tensor_product(type_in_1.Rs,
                                   type_in_2.Rs,
                                   selection_rule_out,
                                   normalization=normalization)

    return (type_in_1.ls, type_in_2.ls, SphericalTensorType.from_Rs(Rs_out)), TP


def compute_channel_mapping_matrix(tensor_type: SphericalTensorType) -> torch.Tensor:
    """
    Note: inspired by rs.map_mul_to_Rs.

	:param tensor_type:
    :return: Dim (tensor_type.dim x tensor_type.C)
    """
    # (tensor_type.dim x tensor_type.C)
    mapping_matrix = torch.zeros(tensor_type.dim, tensor_type.C)
    channel_index = 0
    for l, C_l in enumerate(tensor_type.multiplicities):
        for c_l in range(C_l):
            mapping_matrix[tensor_type.slice_c_l(l, c_l), channel_index] = 1.0
            channel_index += 1
    return mapping_matrix


def compute_angular_mapping_tensor(original_filter_type: SphericalTensorType, angular_filter_type:  SphericalTensorType) \
        -> Tuple[SphericalTensorType, torch.Tensor]:
    """
    We assume that angular parts of the filter do not contain learnable parameters.
    Because of this for each order l the same angular filter can be used for all filter channels of order l.
    But there may be multiple different angular filters for each order l (when two angular filter were combined).

    This function applies these projections to the original_C'.
    Note that the channels of the filter type change if the angular filter contains multiple channels for some l.
    In that case the new C_l' = C_l * C_l_angular
    updated_C_: Dim (type_out.dim x type_in.dim x angular_filter_type.dim x updated_filter_type.dim)

    If all C_l_angular = 1, then projection_tensor_iij = rs.map_irrep_to_Rs.ij

    Note: inspired by rs.map_irrep_to_Rs

    :param original_C_: Dim (type_out.dim x type_in.dim x type_original_filter.dim)
    :param original_filter_type:
    :param angular_filter_type:
    :return:
    - updated_filter_type
    - projection_tensor: Dim (updated_filter_type.dim, original_filter_type.dim, angular_filter_type.dim)
    """
    assert original_filter_type.l_max == angular_filter_type.l_max
    udpated_filter_multiplicities = [C_l * C_l_angular
                                     for C_l, C_l_angular
                                     in zip(original_filter_type.multiplicities, angular_filter_type.multiplicities)]

    updated_filter_type = SphericalTensorType.from_multiplicities(udpated_filter_multiplicities)

    # (updated_filter_type.dim x original_filter_type.dim x angular_filter_type.dim)
    mapping_tensor = torch.zeros(updated_filter_type.dim, original_filter_type.dim, angular_filter_type.dim)
    for l, (C_l, C_l_angular) in enumerate(zip(original_filter_type.multiplicities,
                                               angular_filter_type.multiplicities)):
        for c_l in range(C_l):
            for c_l_angular in range(C_l_angular):
                updated_c_l = (c_l * C_l_angular) + c_l_angular

                # map the corresponding slices, note that different m are not mixed
                # => use eye_3d tensor of dim ((2l+1) x (2l+1) x (2l+1))
                mapping_tensor[updated_filter_type.slice_c_l(l, updated_c_l),
                               original_filter_type.slice_c_l(l, c_l),
                               angular_filter_type.slice_c_l(l, c_l_angular)] = eye_3d(2*l + 1) / (C_l_angular ** 0.5)

    return updated_filter_type, mapping_tensor


sh = spherical_harmonics_xyz


def normalize_sh(Y: SphericalTensor, normalization: str = 'norm'):
    """
    see kernel_mod.py line 20.

	:param Y: spherical tensor. Dim (num_vectors, M_all)
    :return:
    """
    # Normalize the spherical harmonics
    if normalization == 'component':
        diag = torch.ones(Y.type.dim)
    if normalization == 'norm':
        diag = torch.cat([torch.ones(2 * l + 1) / math.sqrt(2 * l + 1) for l in Y.type.repeated_ls])
    norm_Y = math.sqrt(4 * math.pi) * torch.diag(diag)  # [M_all, M_all]

    return Y.apply_operator(norm_Y)


def normalized_sh(ls: List[int], vectors: torch.Tensor, normalization: str = 'component') -> SphericalTensor:
    """
    :param ls:
    :param vectors: (num_vectors, 3)
    :param normalization:
    :return: Dim (num_vectors, M_all)
    """
    tensor_type = SphericalTensorType.from_ls(ls)
    num_vectors, _ = vectors.size()

    if num_vectors == 0:
        return SphericalTensor(vectors.new_empty(0, tensor_type.dim), type=tensor_type)
    else:
        sh_data = sh(tensor_type.ls, vectors)  # (num_vectors, M_all)
        return normalize_sh(SphericalTensor(sh_data, type=tensor_type), normalization=normalization)


SelectionRuleInterface = Callable[[int, int], List[int]]


def predefined_selection_rule(l_by_l_pair: Dict[Tuple[int, int], List[int]]) -> SelectionRuleInterface:
    return lambda l_1, l_2: l_by_l_pair[(l_1, l_2)]


def selection_rule(lmax=None, lfilter=None) -> SelectionRuleInterface:
    return partial(selection_rule_fn, lmax=lmax, lfilter=lfilter)


def selection_rule_fn(l1: int, l2: int, lmax=None, lfilter=None) -> List[int]:
    """
    selection rule according to the triangle inequality:
    \|l1 - l2\| <= l  <= l1 + l2.

    This inequality is euqivalent to the following inequality system:
    (1) l1 + l2 >= l
    (2) l2 + l >= l1
    (3) l + l1 >= l2

    l1 and l2 can be the input orders and l is the output order
    or alternatively l1 and l2 can be the output order and one input order and l is the second input order.

    If lmax is given, then also: l <= lmax
    If lfilter is given, then also: lfilter(l) has to be True

    Note that as

    :return: list of l for which the conditions of the triangle inequality, lmax and lfilter hold
    """
    if lmax is None:
        l_max = l1 + l2
    else:
        l_max = min(lmax, l1 + l2)
    ls = list(range(abs(l1 - l2), l_max + 1))
    if lfilter is not None:
        ls = list(filter(lfilter, ls))
    return ls


SelectionRuleOutInterface = Callable[[int], List[Tuple[int, int]]]


def predefined_selection_rule_out(l_in_pairs_by_l_out: Dict[int, List[Tuple[int, int]]]) -> SelectionRuleOutInterface:
    return lambda l_out: l_in_pairs_by_l_out[l_out]


def selection_rule_out(l_diff_to_out_max: int = 1, l_max: int = None, l_in_diff_max: int = None,
                          l_in_filter: Callable[[int], bool] = None, l_pair_filter: Callable[[int, int], bool] = None) \
        -> SelectionRuleOutInterface:
    return partial(selection_rule_out_fn, l_diff_to_out_max=l_diff_to_out_max, l_max=l_max,
                   l_in_diff_max=l_in_diff_max, l_in_filter=l_in_filter, l_pair_filter=l_pair_filter)


def selection_rule_out_fn(l_out: int, l_diff_to_out_max: int = 1, l_max: int = None, l_in_diff_max: int = None,
                          l_in_filter: Callable[[int], bool] = None, l_pair_filter: Callable[[int, int], bool] = None) -> List[Tuple[int, int]]:
    """

    Examples with default values:
    - l_out = 0 => [(0, 0), (1, 1)]
    - l_out = 1 => [(0, 1), (1, 2), (1, 0), (2, 1), (1, 1), (2, 2)]

    :param l_out:
    :param l_max: maximum values for l_1 and l_2 such that l_1 <= l_max and l_2 <= l_max
        default: l_out + l_diff_to_out_max
    :param l_diff_to_out_max: maximum difference between l_1 or l_2 to l_out such that:
        \|l_out - l_1\| <= l_diff_to_out_max and \|l_out - l_2\| <= l_diff_to_out_max
        default: 1
    :param l_in_diff_max: maximum difference between l_1 and l_2 such that: \|l_1 - l_2\| <= l_in_diff_max
        default: l_out
    :param l_in_filter:
    :param l_pair_filter:
    :return:
    """
    if l_in_diff_max is None:
        l_in_diff_max = l_out
    else:
        l_in_diff_max = min(l_in_diff_max, l_out)

    if l_max is None:
        l_max = l_out + l_diff_to_out_max
    else:
        l_max = min(l_max, l_out + l_diff_to_out_max)

    l_min = max(0, l_out - l_diff_to_out_max)
    l_min = max(l_min, l_out - l_in_diff_max)

    # all pairs with l_1 < l_2
    l_in_pairs = [(l_1, l_2)
                  for l_1 in range(l_min, l_max) for l_2 in range(l_1 + 1, min(l_1 + l_in_diff_max, l_max) + 1)
                  if l_1 + l_2 >= l_out]

    # all pairs with l_2 < l_1
    l_in_pairs += [(l_2, l_1) for l_1, l_2 in l_in_pairs]

    # all pairs with l_1 = l_2
    l_in_pairs += [(l, l) for l in range(max(l_min, (l_out+1) // 2), l_max + 1)]

    # --- filter ---
    if l_in_filter is not None:
        l_in_pairs = [(l_1, l_2) for l_1, l_2 in l_in_pairs if l_in_filter(l_1) and l_in_filter(l_2)]
    if l_pair_filter is not None:
        l_in_pairs = [(l_1, l_2) for l_1, l_2 in l_in_pairs if l_pair_filter(l_1, l_2)]

    return l_in_pairs
