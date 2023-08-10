from einops import rearrange
import matplotlib.pyplot as plt
import argparse
import numpy as np
from PIL import Image
from torchvision.transforms import transforms
import sde
import ml_collections
import torch
from torch import multiprocessing as mp
from torchvision.utils import make_grid, save_image
import utils
import einops
from torch.utils._pytree import tree_map
import accelerate
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from dpm_solver_pytorch import NoiseScheduleVP, model_wrapper, DPM_Solver
import tempfile
import time
from tools.fid_score import calculate_fid_given_paths
from absl import logging
import builtins
import os
import libs.autoencoder
from absl import flags
from absl import app
from ml_collections import config_flags
import sys
from pathlib import Path
from dataset.pos import get_2d_local_sincos_pos_embed
from torchvision.utils import save_image
from tqdm import tqdm
import matplotlib.pyplot as plt
from eval_dir.inception import inception_score

def encode(_batch, autoencoder):
    return autoencoder.encode(_batch)

def decode(_batch, autoencoder):
    return autoencoder.decode(_batch)

def unpreprocess(v):
    v = 0.5 * (v + 1.)
    v.clamp_(0., 1.)
    return v

def destandard(v):
    v = (v + 1) * 127.5
    return v

def calculate_sin_cos(lpos, gpos, grid_size=12):
    kg = gpos[3] / grid_size
    w_bias = (lpos[1] - gpos[1]) / kg
    kl = lpos[3] / grid_size
    w_scale = kl / kg
    kg = gpos[2] / grid_size
    h_bias = (lpos[0] - gpos[0]) / kg
    kl = lpos[2] / grid_size
    h_scale = kl / kg
    return get_2d_local_sincos_pos_embed(1024, grid_size, w_bias, w_scale, h_bias, h_scale)

def calculate_input_pos(target):
    init_location = (1000, 1000, 256, 256)
    top, down, left, right = target
    i = init_location[0] - int(256 * top)
    j = init_location[1] - int(256 * left)
    h = int(256 * (top + down)) + 256
    w = int(256 * (left + right)) + 256
    target = (i, j, h, w)
    return init_location, target

def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print

def init_distributed_mode(args):
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK']) % torch.cuda.device_count()
    elif 'SLURM_PROCID' in os.environ:
        args.rank = int(os.environ['SLURM_PROCID'])
        args.gpu = args.rank % torch.cuda.device_count()
    else:
        print('Not using distributed mode')
        setup_for_distributed(is_master=True)  # hack
        args.distributed = False
        return
    if "SLURM_JOB_NODELIST" in os.environ:
        cmd = 'scontrol show hostnames ' + os.getenv('SLURM_JOB_NODELIST')
        stdout = subprocess.check_output(cmd.split())
        host_name = stdout.decode().splitlines()[0]
        args.dist_url = f'tcp://{host_name}:15752'
    if 'MASTER_ADDR' in os.environ and 'MASTER_PORT' in os.environ:
        args.dist_url = f'tcp://'+str(os.environ['MASTER_ADDR']) + ':' +str(os.environ['MASTER_PORT'])
    else:
        args.dist_url = f'tcp://localhost:27461'
    # args.dist_url = f'tcp://localhost:27461'
    args.distributed = True
    torch.cuda.set_device(args.gpu)
    args.dist_backend = 'nccl'
    print('| distributed init (rank {}): {}, gpu {}'.format(
        args.rank, args.dist_url, args.gpu), flush=True)
    torch.distributed.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                         world_size=args.world_size, rank=args.rank)
    torch.distributed.barrier()
    args.rank = torch.distributed.get_rank()
    setup_for_distributed(args.rank == 0)
    print("Initialization finish")

class WikiArtDataset(Dataset):
    def __init__(self, path='./dataset/wikiart/train/', size=56):
        f_name = os.listdir(path)
        self.path = [path+str(f_name[i]) for i in range(len(f_name))]
        print("Total evaluation images: ", len(self.path))
        self.input_crop = transforms.Compose([
            transforms.CenterCrop((size, size)),
            transforms.Resize((192, 192))
        ])
        self.target_crop = transforms.Compose([
            transforms.Resize((192, 192))
        ])
        self.to_tensor = transforms.ToTensor()
    def __len__(self):
        return len(self.path)
    def __getitem__(self, idx):
        path = self.path[idx]
        pil_image = Image.open(path)
        pil_image.load()
        pil_image = pil_image.convert("RGB")
        pil_image = self.target_crop(pil_image)
        target_img = np.array(pil_image)
        target_img = target_img / 127.5 - 1
        input_img = np.array(self.input_crop(pil_image))
        input_img = input_img / 127.5 - 1
        return self.to_tensor(input_img), self.to_tensor(target_img)

class BuildingDataset(Dataset):
    def __init__(self, path='./dataset/building/test/', size=56):
        f_name = os.listdir(path)        
        self.path = [path+str(f_name[i]) for i in range(len(f_name))]
        print("Total evaluation images: ", len(self.path))
        self.input_crop = transforms.Compose([
            transforms.CenterCrop((size, size)),
            transforms.Resize((192, 192))
        ])
        self.target_crop = transforms.Compose([
            transforms.Resize((192, 192))
        ])
        self.to_tensor = transforms.ToTensor()
    def __len__(self):
        return len(self.path)
    def __getitem__(self, idx):
        path = self.path[idx]
        pil_image = Image.open(path)
        pil_image.load()
        pil_image = pil_image.convert("RGB")
        # pil_image = self.target_crop(pil_image)
        target_img = np.array(pil_image)
        target_img = target_img / 127.5 - 1
        input_img = np.array(self.input_crop(pil_image))
        input_img = input_img / 127.5 - 1
        return self.to_tensor(input_img), self.to_tensor(target_img)

class FlickrDataset(Dataset):
    def __init__(self, path='./dataset/scenery/train/', size=56):
        f_name = os.listdir(path)     
        # self.path = [path+str(f_name[i]) for i in range(len(f_name))]   
        self.path = [path+str(f_name[i]) for i in range(len(f_name)) if int(f_name[i].split('_')[-1].split('.')[0].replace(',', ''))>5040]
        print("Total evaluation images: ", len(self.path))
        self.input_crop = transforms.Compose([
            transforms.CenterCrop((size, size)),
            transforms.Resize((192, 192))
        ])
        self.target_crop = transforms.Compose([
            transforms.Resize((192, 192))
        ])
        self.to_tensor = transforms.ToTensor()
    def __len__(self):
        return len(self.path)
    def __getitem__(self, idx):
        path = self.path[idx]
        pil_image = Image.open(path)
        pil_image.load()
        pil_image = pil_image.convert("RGB")
        pil_image = self.target_crop(pil_image)
        target_img = np.array(pil_image)
        target_img = target_img / 127.5 - 1
        input_img = np.array(self.input_crop(pil_image))
        input_img = input_img / 127.5 - 1
        return self.to_tensor(input_img), self.to_tensor(target_img)

def denorm_img(tensor):
    _mean = torch.tensor([0.5044838, 0.5044838, 0.5044838]).unsqueeze(-1).unsqueeze(-1).unsqueeze(0)
    _std = torch.tensor([0.1355051, 0.1355051, 0.1355051]).unsqueeze(-1).unsqueeze(-1).unsqueeze(0)
    tensor = tensor * _std.expand_as(tensor).cuda() + _mean.expand_as(tensor).cuda()
    tensor = rearrange(tensor[0:1], 'b c w h -> b w h c').detach().cpu()
    tensor = np.clip(tensor[0].numpy(), 0, 1)
    return tensor

def sampling(args, config):
    init_distributed_mode(args)
    # args.gpu = 'cuda:1'
    autoencoder = libs.autoencoder.get_model("assets/stable-diffusion/autoencoder_kl.pth")
    autoencoder.to(args.gpu)
    train_state = utils.initialize_train_state(config, args.gpu)
    train_state.resume(config.ckpt_root)
    nnet = train_state.nnet
    nnet_ema = train_state.nnet_ema
    nnet_ema.eval()
    score_model = sde.ScoreModel(nnet, pred=config.pred, sde=sde.VPSDE())
    score_model_ema = sde.ScoreModel(nnet_ema, pred=config.pred, sde=sde.VPSDE())
    # top, down, left, right
    # target_expansion = (0.1, 0.1, 0.1, 0.1)
    target_expansion = args.target_expansion
    anchor, target = calculate_input_pos(target_expansion)

    prime_target_pos = torch.FloatTensor(calculate_sin_cos(target, anchor)).to(args.gpu)
    dataset = FlickrDataset(size=args.size)
    sampler = torch.utils.data.distributed.DistributedSampler(dataset, shuffle=False)
    dataloader = DataLoader(dataset, batch_size=args.batch_size // 8, shuffle=False, num_workers=args.workers, sampler=sampler, drop_last=False)
    type_ = args.eval_dir.split('/')[-2]
    print(f"Start sampling..., type: {type_}")
    # o_scores, g_scores = [], []
    patch_mean, patch_std = 0.5044838, 0.1355051
    transform_out = transforms.Normalize(mean=torch.tensor((patch_mean,patch_mean,patch_mean)),  std=torch.tensor((patch_std,patch_std,patch_std)))
    for batch_idx, (input_img, target_img) in tqdm(enumerate(dataloader)):
        input_img = input_img.to(args.gpu).float()
        target_img = target_img.to(args.gpu).float()
        prime_target_position = prime_target_pos.unsqueeze(0).repeat(input_img.size(0), 1, 1).float()
        encode_anchor = encode(input_img, autoencoder)
        z_init = torch.randn(encode_anchor.size(), device=args.gpu)
        noise_schedule = NoiseScheduleVP(schedule='linear')
        kwargs = {'conditions': [encode_anchor, prime_target_position]}
        model_fn = model_wrapper(score_model_ema.noise_pred, noise_schedule, time_input_type='0', model_kwargs=kwargs)
        dpm_solver = DPM_Solver(model_fn, noise_schedule)
        z = dpm_solver.sample(z_init, steps=50, eps=1e-4, adaptive_step_size=False, fast_version=False)
        end = time.time()
        pred_target = decode(z, autoencoder)

        pred_target = unpreprocess(pred_target)
        input_img = unpreprocess(input_img)

        pred_target = transform_out(pred_target)
        input_img = transform_out(input_img)
        for i in range(pred_target.size(0)):
            index = batch_idx * args.batch_size + i + args.rank * pred_target.size(0)
            plt.imsave(f'{args.eval_dir}/gen/{index}.png', denorm_img(pred_target[i:i + 1]), vmin=0, vmax=1)
            plt.imsave(f'{args.eval_dir}/ori/{index}.png', denorm_img(input_img[i:i + 1]), vmin=0, vmax=1)
    print(f"Finished sampling")

def get_args_parser():
    parser = argparse.ArgumentParser('OutDiff', add_help=False)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--target_expansion', nargs='+', type=float, default=(0.25, 0.25, 0.25, 0.25))
    parser.add_argument('--size', type=float, default=56)
    parser.add_argument('--eval_dir', type=str, default="./eval_dir/scenery/3x/")
    parser.add_argument('--config', type=str, default="wikiart192_large")
    parser.add_argument('--workers', default=8, type=int, help='Number of data loading workers per GPU.')
    parser.add_argument("--dist_url", default="env://", type=str, help="""url used to set up
        distributed training; see https://pytorch.org/docs/stable/distributed.html""")
    parser.add_argument("--local_rank", default=0, type=int, help="Please ignore and do not set this argument.")
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')
    return parser

if __name__ == '__main__':
    parser = argparse.ArgumentParser('OutDiff', parents=[get_args_parser()])
    args = parser.parse_args()
    if 'wikiart' in args.eval_dir:
        from configs.wikiart192_large import get_config
    elif 'scenery' in args.eval_dir:
        from configs.flickr192_large import get_config
    elif 'building' in args.eval_dir:
        from configs.building192_large import get_config
    config = get_config()
    config.config_name = args.config
    config.hparams = "formal"
    config.workdir = os.path.join('workdir', config.config_name, config.hparams)
    config.ckpt_root = os.path.join(config.workdir, 'ckpts')
    config.sample_dir = os.path.join(config.workdir, 'samples')
    sampling(args, config)
