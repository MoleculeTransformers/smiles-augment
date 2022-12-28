from abc import ABC, abstractmethod
from typing import List, Union


class BaseAugmenter(ABC):
    @abstractmethod
    def augment(self, smiles: Union[str, List[str]], n_augment: int = 1):
        pass
