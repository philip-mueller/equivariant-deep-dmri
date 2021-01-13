import math
from functools import partial
from typing import Callable, Union

import torch
from torch import nn

from equideepdmri.layers.filter.utils import get_scalar_non_linearity

"""
RadialBasisConstructor(basis_size: int, max_radius: float) -> RadialBasis
"""
RadialBasisConstructor = Callable[[int, Union[float, torch.Tensor]], 'RadialBasis']


class RadialBasis(nn.Module):
    """
    Radial basis which may be used in the radial basis filters.
    """
    def __init__(self, basis_size: int, radial_basis_type_name: str):
        """

        :param basis_size: Size of the radial basis (relevant for the output dim when applying this radial basis to radii.
        :param radial_basis_type_name: Name of this type of radial basis, used for debugging only.
        """
        super().__init__()
        self.basis_size = basis_size
        self.radial_basis_type_name = radial_basis_type_name

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param x: Batch of radii on which the radial basis should be applied. Dim (radii_batch_size)
        :return Dim (radii_batch_size x basis_size)
        """
        raise NotImplementedError


class FC(nn.Module):
    """
    Fully Connected Network, where each layer consists of a linear transformation (i.e. fully connected layer)
    and a nonlinearity.

    Note: based on e3nn.radial.FC but with the following adaption:
      - removed additional linear output layer as this output layer is implemented in filter_kernel.Kernel.
    """
    def __init__(self, dim_in: int, num_layers: int = 0, num_units: int = 0, activation_function=None):
        """

        :param dim_in: Number of input neurons.
            Note that dim_out = dim_in for num_layers = 0.
        :param num_layers: Number of layers (hidden layers + output layer). If num_layers == 0 then the FC is the identity,
            i.e. the output is the same as the input.
        :param num_units: Number of units (neurons) in the hidden layers and the output layer.
            Must be >0 if num_layers > 0.
            Note that dim_out = num_units for num_layers > 0.
        :param activation_function: Activation function applied after the linear transformation in each layer.
            Not applied if num_layer == 0.
            Must not be None if num_layers > 0.
        """
        super().__init__()
        assert dim_in > 0

        weights = []

        if num_layers == 0:
            self.weights = []
            self.dim_out = dim_in
        else:
            assert num_layers > 0
            assert num_units > 0 and activation_function is not None, \
                'num_hidden_units > 0 and activation_function need to be given if num_layers > 0.'

            prev_num_units = dim_in
            for _ in range(num_layers):
                weights.append(nn.Parameter(torch.randn(num_units, prev_num_units), requires_grad=True))
                prev_num_units = num_units
            self.weights = torch.nn.ParameterList(weights)
            self.dim_out = num_units

        self.activation_function = activation_function

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies the fully connected network to x.
        :param x: Input. Dim (radii_batch_size x dim_in)
        :return: Dim (radii_batch_size x dim_out)
            Note that dim_out = dim_in for num_layers = 0 and dim_out = num_units for num_layers > 0.
        """
        for i, W in enumerate(self.weights):
            num_units_in = x.size(1)

            if i == 0:
                # note: normalization assumes that the sum of the inputs is 1
                x = self.activation_function(x @ W.t())
            else:
                x = self.activation_function(x @ (W.t() / num_units_in ** 0.5))

        return x


class FiniteElement_RadialBasis(RadialBasis):
    def __init__(self, reference_points: torch.Tensor,
                 radial_basis_fn: Callable[[torch.Tensor], torch.Tensor],
                 radial_basis_type_name: str,
                 num_layers: int = 0, num_units: int = 0, activation_function=None):
        """
        Radial Basis based on some radial basis function applied to multiple reference points
        optionally followed by a fully connected neural network (FC).

        If num_layers == 0 then basis_size = len(reference_points) else basis_size = num_units.

        Note: based on e3nn.radial.FiniteElementFCModel and e3nn.radial.FiniteElementModel
        :param reference_points: the reference points (list of scalars) of Dim (num_reference_points).
            The radial_basis_fn is applied to each of these scalars.
        :param radial_basis_fn: radial basis function applied to each of the reference points.
            batch-wise scalar function: (batch_size) -> (batch_size)
        :param radial_basis_type_name: Name of this type of radial basis, used for debugging only.
        :param num_layers: Number of layers (hidden layers + output layer) of the FC.
            If num_layers == 0 then the FC is not used.
        :param num_units: Number of units (neurons) in the hidden layers and the output layer of the FC.
            Must be >0 if num_layers > 0.
            Note that basis_size = num_units for num_layers > 0.
        :param activation_function: Activation function applied after the linear transformation in each layer of the FC.
            Not applied if num_layer == 0.
            Must not be None if num_layers > 0.
        """
        assert len(reference_points.size()) == 1

        model = FC(len(reference_points),
                   num_layers=num_layers, num_units=num_units, activation_function=activation_function)
        super().__init__(basis_size=model.dim_out, radial_basis_type_name=radial_basis_type_name)

        self.model = model
        self.register_buffer('reference_points', reference_points)
        self.radial_basis_fn = radial_basis_fn

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param x: Dim (radii_batch_size)
        :return: tensor Dim (radii_batch_size x model.dim_out)
            where model.dim_out = len(reference_points) if num_layers == 0 else model.dim_out = num_units.
        """
        differences = x.unsqueeze(1) - self.reference_points.unsqueeze(0)  # (radii_batch_size x num_reference_points)
        radii_batch_size, num_reference_points = differences.size()
        # (radii_batch_size x num_reference_points)
        x = self.radial_basis_fn(differences.view(-1)).view(radii_batch_size, num_reference_points)
        return self.model(x)


def Cosine_RadialBasis(basis_size: int, max_radius: float,
                       num_layers: int = 0, num_units: int = 0, activation_function='relu'):
    """
    Radial basis based on the cosine radial basis function applied to multiple reference points
    optionally followed by a fully connected neural network (FC),
    where the cosine radial basis function is defined as:
    \begin{cases}
        cos^2 (\gamma (d_{ab} - \mu_{k}) \frac{\pi}{2}) & 1 \geq \gamma (d_{ab} - \mu_{k}) \geq -1 \\
        0 & otherwise
    \end{cases},
    and the references points \mu_{k}, indexed by k, are sampled regularly between 0 and max_radius (inclusive)

    Note: based on e3nn.radial.CosineBasisModel

    :param basis_size
    :param max_radius

    """
    reference_points = torch.linspace(0, max_radius, steps=basis_size)
    step = reference_points[1] - reference_points[0]

    basis = partial(cosine_basis_fn, step=step)

    return FiniteElement_RadialBasis(reference_points, radial_basis_fn=basis,
                                     radial_basis_type_name='φ_cos',
                                     num_layers=num_layers, num_units=num_units,
                                     activation_function=activation_function)


def cosine_basis_fn(x, step):
    """
    Note: based on e3nn.radial.CosineBasisModel.basis

    :param x:
    :param step:
    :return:
    """
    return x.div(step).add(1).relu().sub(2).neg().relu().add(1).mul(math.pi / 2).cos().pow(2)


def Cosine_RadialBasisConstructor(num_layers: int = 0, num_units: int = 0, activation_function='relu') \
        -> RadialBasisConstructor:
    activation_function = get_scalar_non_linearity(activation_function)
    return partial(Cosine_RadialBasis,
                   num_layers=num_layers, num_units=num_units, activation_function=activation_function)


def Gaussian_RadialBasis(basis_size: int, max_radius: float, min_radius=0.,
                         num_layers: int = 0, num_units: int = 0, activation_function='relu'):
    """
    Note: based on e3nn.radial.GaussianRadialModel
    :param basis_size:
    :param max_radius:
    :param min_radius:
    :param num_layers:
    :param num_units:
    :param activation_function:
    :return:
    """
    activation_function = get_scalar_non_linearity(activation_function)
    """exp(-x^2 /spacing)"""
    spacing = (max_radius - min_radius) / (basis_size - 1)
    reference_points = torch.linspace(min_radius, max_radius, basis_size)
    sigma = 0.8 * spacing

    basis = partial(gaussian_basis_fn, sigma=sigma)

    return FiniteElement_RadialBasis(reference_points, radial_basis_fn=basis,
                                     radial_basis_type_name='φ_gauss',
                                     num_layers=num_layers, num_units=num_units,
                                     activation_function=activation_function)


def gaussian_basis_fn(x, sigma):
    """
    Note: based on e3nn.radial.GaussianRadialModel.basis
    :param x:
    :param sigma:
    :return:
    """
    return x.div(sigma).pow(2).neg().exp().div(1.423085244900308)


def Gaussian_RadialBasisConstructor(num_layers: int = 0, num_units: int = 0, activation_function='relu') \
        -> RadialBasisConstructor:
    return partial(Gaussian_RadialBasis,
                   num_layers=num_layers, num_units=num_units, activation_function=activation_function)


class Bessel_RadialBasis(RadialBasis):
    """The tex for the math reads
    \begin{cases}
        \sqrt{\frac{2}{c}} \frac{sin(\frac{n \pi}{c} d)}{d} & 0 \leq x \leq max_radius \\
        0 & otherwise
    \end{cases}
    c = max_radius (cutoff), n = basis_size, d = distance (in R_{+})"""
    def __init__(self, basis_size: int, max_radius: float, epsilon=1e-8,
                 num_layers: int = 0, num_units: int = 0, activation_function=None):
        self.model: FC = FC(basis_size,
                            num_layers=num_layers, num_units=num_units, activation_function=activation_function)
        super().__init__(basis_size=self.model.dim_out, radial_basis_type_name='φ_bessel',)

        n = torch.linspace(1, basis_size, basis_size).unsqueeze(0)  # (1 x basis_size)
        n_scaled = n * math.pi / max_radius  # (1 x basis_size)
        self.register_buffer('n_scaled', n_scaled)
        self.factor = math.sqrt(2/max_radius)
        self.max_radius = max_radius
        self.epsilon = epsilon

    def basis(self, x: torch.Tensor) -> torch.Tensor:
        """

        :param x: Dim (radii_batch_size)
        :return: Dim (radii_batch_size x basis_size)
        """
        assert x.ndim == 1
        x_within_cutoff = ((x >= 0.0) * (x <= self.max_radius)) * x  # (radii_batch_size)
        x_within_cutoff = x_within_cutoff.unsqueeze(-1)  # (radii_batch_size x 1)
        return self.factor * torch.sin(self.n_scaled * x_within_cutoff) / (x_within_cutoff + self.epsilon)

    def forward(self, x):
        """

        :param x: Dim (radii_batch_size)
        :return: Dim (radii_batch_size x model.dim_out)
        """
        x = self.basis(x)  # (radii_batch_size x basis_size)
        return self.model(x)


def Bessel_RadialBasisConstructor(epsilon=1e-8, num_layers: int = 0, num_units: int = 0, activation_function=None) \
        -> RadialBasisConstructor:
    return partial(Bessel_RadialBasis,
                   epsilon=epsilon, num_layers=num_layers, num_units=num_units, activation_function=activation_function)


_radial_basis_constructors = {
    'cosine': Cosine_RadialBasisConstructor,
    'gaussian': Gaussian_RadialBasisConstructor,
    'bessel': Bessel_RadialBasisConstructor
}


def build_radial_basis_constructor(name: str, **kwargs) -> RadialBasisConstructor:
    return _radial_basis_constructors[name](**kwargs)
