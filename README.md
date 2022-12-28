# SMILES Augment

Augment molecular SMILES with methods including enumeration, and mixup, for low-data regime settings for downstream supervised drug discovery tasks. For featurizers please lookup [smiles-featurizers](https://github.com/MoleculeTransformers/smiles-featurizers).

## Getting Started

## Use SMILES Augment

### Enumeration Augmenter

```python
from smiles_augment import EnumerationAugmenter
enum_augmenter = EnumerationAugmenter()
enum_augmenter.augment("CCC(C)(C)Br", n_augment=3)

## output
## ['C(C)C(C)(C)Br', 'CC(Br)(C)CC', 'CC(C)(Br)CC']
```

### Mixup Augmenter

```python
from smiles_augment import MixupAugmenter
from smiles_featurizers import BertFeaturizer
import torch

## set device
use_gpu = True if torch.cuda.is_available() else False

## define featurizer
featurizer = BertFeaturizer("shahrukhx01/smole-bert", use_gpu=use_gpu)

## init augmenter
mixup_augmenter = MixupAugmenter(featurizer=featurizer, use_gpu=use_gpu)
augmentations = mixup_augmenter.augment(["CCC(C)(C)Br"], n_augment=3)
augmentations.shape
```
