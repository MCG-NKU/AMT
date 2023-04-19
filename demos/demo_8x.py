"""
a simple 8x demo (please use the weights `gopro_amt-s.pth`)

    > Interpolating eight intermediate frames

    ```bash
    python demos/demo_8x.py -c [CFG] -p [CKPT_PATH] -x [IMG0] -y [IMG1] -o [OUT_PATH]
    ```

    - Results are in the `[OUT_PATH]` (default is `results/8x`) folder.

"""

import os
import sys
import tqdm
import torch
import argparse
import numpy as np
import os.path as osp
from omegaconf import OmegaConf
from torchvision.utils import make_grid

sys.path.append('.')
from utils.utils import read, img2tensor
from utils.utils import (
    read, write,
    img2tensor, tensor2img
    )
from utils.build_utils import build_from_cfg
from utils.utils import InputPadder

parser = argparse.ArgumentParser(
                prog = 'AMT',
                description = 'Demo 8x',
                )
parser.add_argument('-c', '--config', default='cfgs/AMT-S.yaml') 
parser.add_argument('-p', '--ckpt', default='pretrained/gopro_amt-s.pth') 
parser.add_argument('-x', '--img0', default='assets/quick_demo/img0.png') 
parser.add_argument('-y', '--img1', default='assets/quick_demo/img1.png') 
parser.add_argument('-o', '--out_path', default='results/8x') 
args = parser.parse_args()

# ----------------------- Initialization ----------------------- 
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
cfg_path = args.config
ckpt_path = args.ckpt
img0_path = args.img0
img1_path = args.img1
out_path = args.out_path
if osp.exists(out_path) is False:
    os.makedirs(out_path)

# -----------------------  Load model ----------------------- 
network_cfg = OmegaConf.load(cfg_path).network
network_name = network_cfg.name
print(f'Loading [{network_name}] from [{ckpt_path}]...')
model = build_from_cfg(network_cfg)
ckpt = torch.load(ckpt_path)
model.load_state_dict(ckpt['state_dict'])
model = model.to(device)
model.eval()

# -----------------------  Load input frames ----------------------- 
img0 = read(img0_path)
img1 = read(img1_path)
img0_t = img2tensor(img0).to(device)
img1_t = img2tensor(img1).to(device)
padder = InputPadder(img0_t.shape, 16)
img0_t, img1_t = padder.pad(img0_t, img1_t)
embt = torch.arange(1/8, 1, 1/8).float().view(1, 7, 1, 1).to(device)

# -----------------------  Interpolater ----------------------- 
imgt_preds = []
for i in range(7):
    with torch.no_grad():
        imgt_pred = model(img0_t, img1_t, embt[:, i: i + 1, ...], eval=True)['imgt_pred']
    imgt_pred = padder.unpad(imgt_pred)
    imgt_preds.append(imgt_pred.detach())
concat_img = torch.cat([img0_t, *imgt_preds, img1_t], 0)
concat_img = make_grid(concat_img, nrow=3)
concat_img = tensor2img(concat_img)
write(f'{out_path}/grid.png', concat_img)

# -----------------------  Write generate frames to disk ----------------------- 
for i, imgt_pred in enumerate(imgt_preds):
    imgt_pred = tensor2img(imgt_pred)
    write(f'{out_path}/imgt_pred_{i}.png', imgt_pred)