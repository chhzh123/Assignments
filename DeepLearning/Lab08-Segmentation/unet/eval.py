import torch
import numpy as np
import torch.nn.functional as F

from .dice_loss import dice_coeff


def eval_net(net, generator, gpu=False, num_validation=50 ):
    """Evaluation without the densecrf with the dice coefficient"""
    net.eval()
    tot = 0
    for i in range(num_validation):

        image, mask = generator(1)

        image = np.transpose( image, axes=[0, 3, 1, 2] ).astype( np.float32 )
        mask  = np.transpose( mask,  axes=[0, 3, 1, 2] ).astype( np.float32 )

        image = torch.from_numpy(image)
        mask  = torch.from_numpy(mask )

        if gpu:
            image = image.cuda()
            mask  = mask.cuda()

        mask_pred = net(image)
        mask_pred = (F.sigmoid(mask_pred) > 0.5).float()

        tot += dice_coeff(mask_pred, mask ).item()
    return tot / (i+1)
