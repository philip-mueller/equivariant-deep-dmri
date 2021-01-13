"""
Utitlies required for example.ipynb
"""
import torch
from torch.utils.data import DataLoader

from equideepdmri.utils.q_space import Q_SamplingSchema


def compute_binary_label_weights(training_dataloader: DataLoader) -> torch.Tensor:
    num_P_voxels = 0.
    num_total_voxels = 0.
    for i, batch in enumerate(training_dataloader):
        target: torch.Tensor = batch['target']
        brain_mask = batch['brain_mask'].bool()
        target = target[brain_mask]
        num_P_voxels += float(target.nonzero().size(0))
        num_total_voxels += float(target.numel())

    return torch.tensor(1 - (num_P_voxels/num_total_voxels))


class RandomDMriSegmentationDataset:
    def __init__(self, N, Q, num_b0, p_size: tuple):
        self.N = N
        assert Q >= num_b0

        q_vectors = torch.rand(Q, 3)
        q_vectors[:num_b0, :] = 0.0
        self.q_sampling_schema = Q_SamplingSchema(q_vectors)

        assert len(p_size) == 3
        self.scans = torch.randn(N, Q, *p_size)
        self.targets = (torch.randn(N, *p_size) > 0.8).float()
        self.brain_masks = torch.ones(N, *p_size)

    def __len__(self):
        return self.N

    def __getitem__(self, i):
        assert isinstance(i, int)  # only batch-size == 1

        return {'sample_id': str(i), 'input': self.scans[i],
                'target': self.targets[i], 'brain_mask': self.brain_masks[i] }
