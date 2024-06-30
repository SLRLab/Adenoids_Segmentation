import torch
import torch.nn.functional as F


def loss_function(pred, mask, smooth=1):

    weit = 1 + 5*torch.abs(F.avg_pool2d(mask, kernel_size=15, stride=1, padding=7) - mask)
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduction='mean')
    wbce = (weit*wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3)) 

    pred = torch.sigmoid(pred)
    inter = ((pred * mask) * weit).sum(dim = (2, 3)) 
    union = ((pred + mask) * weit).sum(dim = (2, 3)) 
    wiou = 1 - (inter + smooth)/(union - inter + smooth)

    return (wbce + wiou).mean()
