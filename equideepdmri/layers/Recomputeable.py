from abc import abstractmethod, ABC

from torch import nn


class Recomputable(ABC):
    @abstractmethod
    def recompute(self):
        pass


def recompute(module: nn.Module):
    if isinstance(module, Recomputable):
        module.recompute()
