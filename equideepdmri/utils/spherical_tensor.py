from typing import List, Tuple, Iterable, Union

import torch
from e3nn import rs


class SphericalTensorType:
    """
    The type of a (multi-channel) spherical tensor, i.e. the number of channels of each tensor order.
    """
    multiplicities: Tuple[int]
    dim: int
    Rs: List[Tuple[int, int]]

    def __init__(self, multiplicities: Iterable[int]):
        """
        :param multiplicities The element at index i of the list
            defines the number of order-i channels,
            e.g. the first element defines the number of order-0 (scalar) channels
            and the second the number of order-1 (vector) channels and so on.
            For all orders corresponding to out-of-range indices the number of channels is 0.
        """
        self.multiplicities = tuple(multiplicities)
        assert len(self.multiplicities) > 0

        self.order_start_indices = []
        accumulated_index = 0
        for l, C in enumerate(self.multiplicities):
            self.order_start_indices.append(accumulated_index)
            accumulated_index += C * (2*l + 1)

        self.dim = accumulated_index

        self.Rs = [(C_l, l) for l, C_l in enumerate(self.multiplicities) if C_l > 0]

    @property
    def repeated_ls(self):
        repeated_ls = []
        for C_l, l in self.Rs:
            repeated_ls.extend([l] * C_l)
        return repeated_ls

    @property
    def ls(self):
        return [l for l, C_l in enumerate(self.multiplicities) if C_l > 0]

    @property
    def l_max(self) -> int:
        return len(self.multiplicities) - 1

    @property
    def C(self) -> int:
        """
        Number of channels for all l (sum of all C_l).
        :return:
        """
        return sum(C_l for l, C_l in enumerate(self.multiplicities))

    def C_l(self, l) -> int:
        """
        Number of channels for order l.
        :param l:
        :return:
        """
        return self.multiplicities[l] if l <= self.l_max else 0

    def slice_l(self, l: int) -> slice:
        """
        Slice of an spherical tensor representing the data of all channels of a given order l.
        :param l:
        :return: Slice of length C_l*(2l+1)
        """
        start_index = self.order_start_indices[l]
        data_length = self.multiplicities[l] * (2*l + 1)
        return slice(start_index, start_index + data_length)

    def slice_c_l(self, l: int, c_l: int) -> slice:
        """
        Slice of an spherical tensor representing the data of channel c_l of given order l.
        :param l:
        :param c_l: 0-based channel index (within the given order l)
        :return: Slice of length (2l+1)
        """
        start_index = self.order_start_indices[l] + c_l * (2*l + 1)
        return slice(start_index, start_index + (2*l + 1))

    def __eq__(self, other):
        if isinstance(other, SphericalTensorType):
            return other.multiplicities == self.multiplicities
        elif isinstance(other, list) or isinstance(other, tuple):
            return tuple(other) == self.multiplicities
        return False

    def __repr__(self):
        return f'<SphericalTensorType {self.multiplicities}>'

    @staticmethod
    def from_Rs(Rs: List[Tuple[int, int]]) -> 'SphericalTensorType':
        Rs = rs.convention(Rs)
        max_l = max(l for _, l, _ in Rs)
        multiplicities = [0 for _ in range(max_l + 1)]
        for C_l, l, _ in iter(Rs):
            multiplicities[l] += C_l
        return SphericalTensorType(multiplicities)

    @staticmethod
    def from_ls(ls: Iterable[int]) -> 'SphericalTensorType':
        ls = set(ls)
        max_l = max(ls)
        return SphericalTensorType(1 if l in ls else 0 for l in range(max_l + 1))

    @staticmethod
    def from_multiplicities(multiplicities: List[int]) -> 'SphericalTensorType':
        return SphericalTensorType(multiplicities)

    @staticmethod
    def from_multiplicities_or_type(tensor_type: Union['SphericalTensorType', List[int]]) -> 'SphericalTensorType':
        if isinstance(tensor_type, SphericalTensorType):
            return tensor_type
        else:
            return SphericalTensorType.from_multiplicities(tensor_type)

    @staticmethod
    def concat_tensor_types(*tensor_types: List['SphericalTensorType']) -> ('SphericalTensorType', List[List[int]]):
        l_max = max(tensor_type.l_max for tensor_type in tensor_types)

        result_multiplicities = [0] * (l_max+1)
        concat_indices = [[] for _ in tensor_types]

        current_index = 0
        for l in range(l_max + 1):
            for i, tensor_type in enumerate(tensor_types):
                C_l = tensor_type.C_l(l)
                result_multiplicities[l] += C_l
                concat_indices[i].extend(list(range(current_index, current_index + C_l * (2*l + 1))))
                current_index += C_l * (2*l + 1)

        return SphericalTensorType.from_multiplicities(result_multiplicities), concat_indices


class SphericalTensor:
    """
    Representing an array of spherical tensors of a given type
    """
    _value: torch.Tensor
    _type: SphericalTensorType
    representation_dim: int

    def __init__(self, data: torch.Tensor, type: SphericalTensorType, representation_dim=-1):
        """

        :param data: Dim (... x type.dim)
        :param type:
        :param representation_dim: index of the dimension that contains the spherical tensor representation
        """
        self._type = type
        if representation_dim < 0:
            representation_dim = data.ndim + representation_dim
        self.representation_dim = representation_dim
        self.value = data

    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, value):
        assert value.size()[self.representation_dim] == self.type.dim, \
            f'Expecting dimension {self.representation_dim} of data to be the representation dim ({self.type.dim}) ' \
                f'but dim was {value.size()[self.representation_dim]}. ' \
                f'Note: full size was {value.size()} and type was {self.type}'

        self._value = value

    @property
    def type(self):
        return self._type

    def apply_operator(self, operator_matrix: torch.Tensor, batch_wise=False) -> 'SphericalTensor':
        """

        :param operator_matrix: ([N_op x ]type.dim x type.dim)
        :param batch_wise
            If True the first dimension of this is interpreted as batch dimension (has to be same as N_op) and the operator
            is applied batch-wise.
            If False and the operator has batch-dim, then the N_op operators are applied to the same data of this
            resulting in a added dimension N_op.
            This parameter is ignored if operator_matrix has no batch-dim.
        :return: spherical tensor of dim ([N_op x] ... x type.dim x ...)
        """
        assert operator_matrix.ndim in [2, 3]
        assert operator_matrix.size()[-2:] == (self.type.dim, self.type.dim)

        einsum_indices_1, einsum_indices_2 = self._get_einsum_indices()

        if operator_matrix.ndim == 2:
            data = torch.einsum(f'mk,{einsum_indices_1}k{einsum_indices_2}->{einsum_indices_1}m{einsum_indices_2}',
                                [operator_matrix, self.value])
            representation_dim = self.representation_dim
        else:
            if batch_wise:
                assert self.value.size()[0] == operator_matrix.size()[0]
                assert self.representation_dim > 0
                data = torch.einsum(
                    f'nmk,n{einsum_indices_1[:-1]}k{einsum_indices_2}->n{einsum_indices_1[:-1]}m{einsum_indices_2}',
                    [operator_matrix, self.value])
                representation_dim = self.representation_dim
            else:
                data = torch.einsum(f'nmk,{einsum_indices_1}k{einsum_indices_2}->n{einsum_indices_1}m{einsum_indices_2}',
                                    [operator_matrix, self.value])
                representation_dim = self.representation_dim + 1
        return SphericalTensor(data, type=self.type, representation_dim=representation_dim)

    def _get_einsum_indices(self):
        einsum_indices = 'abcdefghij'
        einsum_indices_before = ''.join(einsum_indices[0:self.representation_dim])
        einsum_indices_after = ''.join(einsum_indices[self.representation_dim:self._value.ndim - 1])

        return einsum_indices_before, einsum_indices_after

    @staticmethod
    def concat(spherical_tensors: Iterable['SphericalTensor'],
               tensor_types: Iterable[SphericalTensorType] = None,
               representation_dim=1) -> Union['SphericalTensor', Tuple[torch.Tensor, SphericalTensorType]]:
        spherical_tensors = list(spherical_tensors)
        if tensor_types is not None:
            tensor_types = list(tensor_types)
            tensors = spherical_tensors
            return_spherical_tensor = False
        else:
            tensor_types = []
            tensors = []
            representation_dim = None
            for spherical_tensor in spherical_tensors:
                assert isinstance(spherical_tensor, SphericalTensor)
                assert representation_dim is None or representation_dim == spherical_tensor.representation_dim
                representation_dim = spherical_tensor.representation_dim
                tensor_types.append(spherical_tensor.type)
                tensors.append(spherical_tensor)
            return_spherical_tensor = True
        assert len(tensors) > 0
        result_type, concat_indices = SphericalTensorType.concat_tensor_types(*tensor_types)

        tensor_values = [tensor.value.transpose(1, representation_dim) for tensor in tensors]

        tensor_size = list(tensor_values[0].size())
        tensor_size[1] = result_type.dim
        result_tensor = tensor_values[0].new_zeros(tensor_size)

        for indices, tensor in zip(concat_indices, tensor_values):
            result_tensor[:, indices, ...] = tensor

        result_tensor = result_tensor.transpose(1, representation_dim)

        if return_spherical_tensor:
            return SphericalTensor(result_tensor, result_type)
        else:
            return result_tensor, result_type
