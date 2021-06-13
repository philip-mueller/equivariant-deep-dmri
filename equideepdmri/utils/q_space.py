import gc
from typing import Union, List, Optional, Iterable

import torch


class Q_SamplingSchema:
    def __init__(self, q_vectors: Union[torch.Tensor, List], b0_mask: torch.Tensor = None, b0_eps=0.0,
                 normalize=False,
                 radial_basis_size: Optional[int] = None, radial_basis_eps: float = 1.0e-3):
        """
        :param q_vectors: List of q-vectors (each 3d) or tensor of size (Q x 3).
        :param b0_mask: Optional tensor of size Q which is True for all q-vectors that represent b=0.
        :param b0_eps: eps value such that all q-vectors with length <b0_eps are treated as b=0.
            Only used if b0_mask = None.
        :param normalize: Whether to normalize the q-length to max=1
        :param radial_basis_size: Size of the radial basis for this sampling scheme, default is number of different q-lengths
            (max Q) or 0 if Q=1.
        :param radial_basis_eps: Minimum difference between q-length to consider them as different.
        """
        if not isinstance(q_vectors, torch.Tensor):
            q_vectors = torch.tensor(q_vectors)
        self.q_vectors = q_vectors.float()

        assert self.q_vectors.ndim == 2 and q_vectors.size()[1] == 3, \
            f'q_vectors is required to have size (Q, 3), but size was {q_vectors.size()}'

        if normalize:
            if self.q_vectors.size()[0] == 1 and self.max_length < 1e-7:
                self.q_vectors = self.q_vectors
            else:
                self.q_vectors = self.q_vectors / self.max_length
            self._normalized_schema = self
        else:
            self._normalized_schema = None

        if radial_basis_size is not None:
            self.radial_basis_size = radial_basis_size
        else:
            self.radial_basis_size = len(torch.unique(torch.floor(self.q_lengths / radial_basis_eps)))
            if self.radial_basis_size == 1:
                # if there is only one radius => no radial basis is required
                self.radial_basis_size = 0

        if b0_mask is not None:
            self.b0_mask = b0_mask
        else:
            self.b0_mask = self.q_lengths <= b0_eps

    @staticmethod
    def from_q_sampling_schema(q_sampling_schema: Union['Q_SamplingSchema', torch.Tensor, List], **kwargs) -> 'Q_SamplingSchema':
        if isinstance(q_sampling_schema, Q_SamplingSchema):
            return q_sampling_schema
        else:
            return Q_SamplingSchema(q_sampling_schema, **kwargs)

    @staticmethod
    def from_b(bvals: torch.Tensor, bvecs: torch.Tensor, b0_eps=0.0, normalize=True):
        """
        Note that (even if normalize is False), then the lengths of the computed q vectors are only
        proportional to the true q vector lengths.
        :param bvals: Q
        :param bvecs: Q * 3
        :param b0_eps:
        :return:
        """
        q_lengths = torch.sqrt(bvals)
        q_vectors = q_lengths.unsqueeze(1) * bvecs
        b0s_mask = bvals <= b0_eps
        return Q_SamplingSchema(q_vectors=q_vectors, b0_mask=b0s_mask, normalize=normalize)

    @property
    def Q(self) -> int:
        return self.q_vectors.size()[0]

    @property
    def normalized(self) -> 'Q_SamplingSchema':
        if self._normalized_schema is None:
            self._normalized_schema = Q_SamplingSchema(self.q_vectors, normalize=True)
        return self._normalized_schema

    @property
    def q_lengths(self) -> torch.Tensor:
        """
        :return: Dim (Q)
        """
        return torch.norm(self.q_vectors, p=2, dim=1)

    @property
    def max_length(self) -> torch.Tensor:
        return self.q_lengths.max()

    @property
    def num_b0_channels(self) -> int:
        return len(self.b0_mask.nonzero())

    def __eq__(self, other):
        if isinstance(other, Q_SamplingSchema):
            return self.q_vectors.allclose(other.q_vectors)
        elif isinstance(other, torch.Tensor):
            return self.q_vectors.allclose(other)
        else:
            return False

    def __repr__(self):
        return f'<Q_SamplingSchema {self.q_vectors}>'

    @staticmethod
    def normalize_multiple_sampling_schemas(sampling_schemas: Iterable['Q_SamplingSchema']) -> List['Q_SamplingSchema']:
        sampling_schemas = list(sampling_schemas)
        max_norm = max(torch.norm(schema.q_vectors, dim=1).max() for schema in sampling_schemas)
        return [Q_SamplingSchema(schema.q_vectors / max_norm) for schema in sampling_schemas]

    def extract_b0_channels(self, tensor_field: torch.Tensor) -> torch.Tensor:
        """
        Extracts only the Q channels with b=0 (based on self) of the given tensor field.
        
        :param tensor_field: (N x dim_in x Q x P_z x P_y x P_x)
        :return: (N x dim_in x num_b0_channels x P_z x P_y x P_x)
        """
        assert tensor_field.ndim == 6
        assert tensor_field.size()[2] == self.Q

        return tensor_field[:, :, self.b0_mask, :, :, :]

    def sampling_schema_for_combine_b0_channels(self, **kwargs) -> 'Q_SamplingSchema':
        """
        Computes the resulting q_sampling schema when combine_b0_channels is done.
        :return:
        """
        if self.num_b0_channels == 0:
            return self

        # (Q_non_b0 x 3)
        non_b0_q_vectors = self.q_vectors[self.b0_mask.logical_not(), :]
        # (Q_result x 3)
        result_q_vectors = torch.cat((self.q_vectors.new_zeros(1, 3), non_b0_q_vectors), dim=0)

        # first channel will be b=0 channel
        result_b0_mask = self.b0_mask.new_zeros(len(result_q_vectors))
        result_b0_mask[0] = True

        return Q_SamplingSchema(result_q_vectors, b0_mask=result_b0_mask, **kwargs)

    def combine_b0_channels(self, tensor_field: torch.Tensor, combination_fn=torch.mean) \
            -> torch.Tensor:
        """
        Combines all b=0 channels to a single b=0 channel, leaves the other channels unchanged.
        The combined b=0 channel will be the first channel.
        
        :param tensor_field: (N x dim_in x Q x P_z x P_y x P_x)
        :param combination_fn: Function to combine the different b=0 channels. Default: mean
        :param kwargs:
        :return: (N x dim_in x (Q-num_b0_channels+1) x P_z x P_y x P_x)
        """
        assert tensor_field.ndim == 6
        assert tensor_field.size()[2] == self.Q

        if self.num_b0_channels == 0:
            return tensor_field

        # (N x dim_in x 1 x P_z x P_y x P_x)
        combined_b0_channel = combination_fn(self.extract_b0_channels(tensor_field), dim=2).unsqueeze(2)

        # (N x dim_in x Q_non_b0 x P_z x P_y x P_x)
        non_b0_channels = tensor_field[:, :, self.b0_mask.logical_not(), :, :, :]
        # (N x dim_in x Q_result x P_z x P_y x P_x)
        result_field = torch.cat((combined_b0_channel, non_b0_channels), dim=2)

        del combined_b0_channel
        del non_b0_channels
        gc.collect()

        return result_field
