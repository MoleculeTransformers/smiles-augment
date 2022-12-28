from typing import List, Tuple, Union
from smiles_augment import BaseAugmenter
from torch import Tensor
from smiles_featurizers import BaseFeaturizer
import torch
import numpy as np


class MixupAugmenter(BaseAugmenter):
    def __init__(
        self, featurizer: BaseFeaturizer, alpha: float = 1.0, use_gpu: bool = False
    ) -> None:
        super().__init__()
        self.featurizer = featurizer
        self.alpha = alpha
        self.use_gpu = use_gpu

    def get_perm(self, x):
        """get random permutation"""
        batch_size = x.size()[0]
        if self.use_gpu:
            index = torch.randperm(batch_size).cuda()
        else:
            index = torch.randperm(batch_size)
        return index

    def mixup(self, x: Tensor, y: Tensor = None) -> Union[Tuple, Tuple[Tensor, Tensor]]:
        index = self.get_perm(x)
        x1 = x[index]

        lam = np.random.beta(self.alpha, self.alpha)
        aug_x = lam * x + (1.0 - lam) * x1
        if y:
            y1 = y[index]
            aug_y = lam * y + (1.0 - lam) * y1
            return (aug_x, aug_y)
        else:
            return aug_x

    def augment(
        self,
        smiles: List[str],
        labels: Union[List[int], Tensor] = None,
        n_augment: int = 1,
    ) -> Tensor:
        assert len(smiles) > 0, "SMILES can not be empty!"
        augmentations_x, augmentations_y = [], []

        x = self.featurizer.embed(smiles=smiles)
        for _ in range(n_augment):
            if labels:
                x_aug, y_aug = self.mixup(x=x, y=labels)
                augmentations_x.append(x_aug)
                augmentations_y.append(y_aug)
            else:
                x_aug = self.mixup(x=x, y=labels)
                augmentations_x.append(x_aug)
        if augmentations_y:
            return torch.stack(augmentations_x, 0).squeeze(1), torch.stack(
                augmentations_y, 0
            )
        else:
            return torch.stack(augmentations_x, 0).squeeze(1)
