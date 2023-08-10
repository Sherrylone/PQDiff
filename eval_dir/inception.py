import torch
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F
import torch.utils.data

from torchvision.models.inception import inception_v3

import numpy as np
from scipy.stats import entropy

def inception_score(imgs, cuda=True, batch_size=32, resize=False, splits=1):
    """Computes the inception score of the generated images imgs
    imgs -- Torch dataset of (3xHxW) numpy images normalized in the range [-1, 1]
    cuda -- whether or not to run on GPU
    batch_size -- batch size for feeding into Inception v3
    splits -- number of splits
    """
    N = len(imgs)

    assert batch_size > 0
    assert N > batch_size

    # Set up dtype
    if cuda:
        dtype = torch.cuda.FloatTensor
        device = 'cuda:0'
    else:
        if torch.cuda.is_available():
            print("WARNING: You have a CUDA device, so you should probably set cuda=True")
        dtype = torch.FloatTensor

    # Set up dataloader
    dataloader = torch.utils.data.DataLoader(imgs, batch_size=batch_size)

    # Load inception model
    inception_model = inception_v3(pretrained=True, transform_input=False)
    inception_model = inception_model.to(device)
    inception_model.eval()
    up = nn.Upsample(size=(299, 299), mode='bilinear', align_corners=True).to(device)
    def get_pred(x):
        if resize:
            x = up(x)
        x = inception_model(x)
        return F.softmax(x, dim=-1).data.cpu().numpy()

    # Get predictions
    preds = np.zeros((N, 1000))

    for i, batch in enumerate(dataloader, 0):
        batch = batch.to(device)
        batchv = Variable(batch).float()
        batch_size_i = batch.size()[0]

        preds[i*batch_size:i*batch_size + batch_size_i] = get_pred(batchv)

    # Now compute the mean kl-div
    split_scores = []

    for k in range(splits):
        part = preds[k * (N // splits): (k+1) * (N // splits), :]
        py = np.mean(part, axis=0)
        scores = []
        for i in range(part.shape[0]):
            pyx = part[i, :]
            scores.append(entropy(pyx, py))
        split_scores.append(np.exp(np.mean(scores)))

    return np.mean(split_scores), np.std(split_scores)

def get_args_parser():
    parser = argparse.ArgumentParser('Inception Score', add_help=False)
    parser.add_argument('--path', type=str)
    return parser

if __name__ == '__main__':
    from PIL import Image
    import torchvision.datasets as dset
    import torchvision.transforms as transforms
    import os
    import argparse
    parser = argparse.ArgumentParser('Inception Score', parents=[get_args_parser()])
    args = parser.parse_args()
    imgs = []
    for f in os.listdir(args.path):
        im = np.array(Image.open(os.path.join(args.path, f))).transpose(2, 0, 1).astype(np.float32)[:3]
        im /= 255
        im = im * 2 - 1
        imgs.append(im)
    imgs = np.stack(imgs, 0)
    imgs = torch.from_numpy(imgs).cuda()
    
    print ("Calculating Inception Score...")
    print (inception_score(imgs, cuda=True, batch_size=32, resize=True, splits=1)[0])
