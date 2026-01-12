from torch import nn
from .isomaxplus import IsoMaxPlusLossSecondPart


def get_criterion(loss_name, reduction='mean', entropic_scale=20):
    if loss_name in ['ce', 'ce_concept']:
        criterion = nn.CrossEntropyLoss(reduction=reduction)
    elif loss_name == 'isomax':
        criterion = IsoMaxPlusLossSecondPart(entropic_scale=entropic_scale, reduction=reduction)
    else:
        raise NotImplementedError()
    return criterion
#