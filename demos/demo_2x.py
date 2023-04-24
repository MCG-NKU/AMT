import os
import cv2
import sys
import glob
import torch
import argparse
import numpy as np
import os.path as osp
from warnings import warn
from omegaconf import OmegaConf
from torchvision.utils import make_grid
sys.path.append('.')
from utils.utils import (
    read, write,
    img2tensor, tensor2img,
    check_dim_and_resize
    )
from utils.build_utils import build_from_cfg
from utils.utils import InputPadder
parser = argparse.ArgumentParser(
                prog = 'AMT',
                description = 'Demo 2n',
                )
parser.add_argument('-c', '--config', type=str, default='cfgs/AMT-S.yaml') 
parser.add_argument('-p', '--ckpt', type=str, default='pretrained/amt-s.pth') 
parser.add_argument('-n', '--niters', type=int, default=6) 
parser.add_argument('-i', '--input', default=['assets/quick_demo/img0.png', 
                                              'assets/quick_demo/img1.png'], nargs='+') 
parser.add_argument('-o', '--out_path', type=str, default='results/2x') 
parser.add_argument('-r', '--frame_rate', type=int, default=24)
parser.add_argument("--save_images", action='store_true', default=False)

args = parser.parse_args()
# ----------------------- Initialization ----------------------- 
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
cfg_path = args.config
ckpt_path = args.ckpt
input_path = args.input
out_path = args.out_path
iters = int(args.niters)
frame_rate = int(args.frame_rate)
save_images = args.save_images
if device == 'cuda':
    anchor_resolution = 1024 * 512
    anchor_memory = 1500 * 1024**2
    anchor_memory_bias = 2500 * 1024**2
    vram_avail = torch.cuda.get_device_properties(device).total_memory
    print("VRAM available: {:.1f} MB".format(vram_avail / 1024 ** 2))
else:
    # Do not resize in cpu mode
    anchor_resolution = 8192*8192
    anchor_memory = 1
    anchor_memory_bias = 0
    vram_avail = 1

# ----------------------- Parse input ----------------------- 
## ------------------- Video input -------------------
if osp.splitext(input_path[0])[1] in ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', 
                                    '.webm', '.MP4', '.AVI', '.MOV', '.MKV', '.FLV', 
                                    '.WMV', '.WEBM']:
    if len(input_path) > 1:
        warn("Multiple video inputs received. Only the first one will be interpolated.")
    vcap = cv2.VideoCapture(input_path[0])
    ori_frame_rate = vcap.get(cv2.CAP_PROP_FPS)
    if frame_rate == 0:
        frame_rate = ori_frame_rate * 2 ** iters
    inputs = []
    w = int(vcap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(vcap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    scale = anchor_resolution / (h * w) * np.sqrt((vram_avail - anchor_memory_bias) / anchor_memory)
    scale = 1 if scale > 1 else scale
    scale = 1 / np.floor(1 / np.sqrt(scale) * 16) * 16
    if scale < 1:
        print(f"Due to the limited VRAM, the video will be scaled by {scale:.2f}")
    padding = int(16 / scale)
    padder = InputPadder((h, w), padding)
    while True:
        ret, frame = vcap.read()
        if ret is False:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_t = img2tensor(frame).to(device)
        frame_t = padder.pad(frame_t)
        inputs.append(frame_t)
    print(f'Loading the [video] from {input_path}, the number of frames [{len(inputs)}]')
else:
    if len(input_path) > 1:
        ## ------------------- Image input ----------------------
        pass
    elif osp.isdir(input_path[0]):
        ## ------------------- Folder input ----------------------
        exts = ['jpg', 'png', 'jpeg', 'bmp', 'tif', 
                'tiff', 'JPG', 'PNG', 'JPEG', 'BMP', 
                'TIF', 'TIFF']
        input_path = glob.glob(osp.join(input_path[0], '*'))
        input_path = sorted([p for p in input_path 
                                if osp.splitext(p)[1][1:] in exts])
    else:
        ## ------------- Regular expression input ----------------
        input_path = sorted(glob.glob(input_path))

    input_path.sort()
    print(f'Loading [images] from [{input_path}], the number of images = [{len(input_path)}]')
    inputs = [img2tensor(read(img_path)).to(device) for img_path in input_path]
    assert len(inputs) > 1, f"The number of input should be more than one (current {len(inputs)})"
    inputs = check_dim_and_resize(inputs)
    h, w = inputs[0].shape[-2:]
    scale = anchor_resolution / (h * w) * np.sqrt((vram_avail - anchor_memory_bias) / anchor_memory)
    scale = 1 if scale > 1 else scale
    scale = 1 / np.floor(1 / np.sqrt(scale) * 16) * 16
    if scale < 1:
        print(f"Due to the limited VRAM, the video will be scaled by {scale:.2f}")
    padding = int(16 / scale)
    padder = InputPadder(inputs[0].shape, padding)
    inputs = padder.pad(*inputs)


embt = torch.tensor(1/2).float().view(1, 1, 1, 1).to(device)

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

# -----------------------  Interpolater ----------------------- 
print(f'Start frame interpolation:')
total_frames = (iters + 1) * (iters + 2) / 2
for i in range(iters):
    print(f'Iter {i+1}. input_frames={len(inputs)} output_frames={2*len(inputs)-1}')
    outputs = [inputs[0]]
    for in_0, in_1 in zip(inputs[:-1], inputs[1:]):
        in_0 = in_0.to(device)
        in_1 = in_1.to(device)
        with torch.no_grad():
            imgt_pred = model(in_0, in_1, embt, scale_factor=scale, eval=True)['imgt_pred']
        outputs += [imgt_pred.cpu(), in_1.cpu()]
    inputs = outputs

# -----------------------  Write video to disk ----------------------- 
outputs = padder.unpad(*outputs)
size = outputs[0].shape[2:][::-1]
if save_images:
    sample_path = os.path.join(out_path, "samples")
    os.makedirs(sample_path, exist_ok=True)
    sample_count = len(os.listdir(sample_path))

video_count = len(glob.glob(os.path.join(out_path, "*.mp4")))

save_video_path = f'{out_path}/demo_{video_count:04d}.mp4'
writer = cv2.VideoWriter(save_video_path, cv2.VideoWriter_fourcc(*"mp4v"), 
                    frame_rate, size)
if save_images and len(outputs) > 50:
        warn('Too many frames to write to disk. ' 
            'This process may take a long time. ' 
            'Consider disable "--save_images" to avoid write images.')
for i, imgt_pred in enumerate(outputs):
    imgt_pred = tensor2img(imgt_pred)
    if save_images:
        write(f'{sample_path}/sample_{sample_count:04d}.png', imgt_pred)
        sample_count += 1
    imgt_pred = cv2.cvtColor(imgt_pred, cv2.COLOR_RGB2BGR)
    writer.write(imgt_pred)        
print(f"Demo video is saved to [{save_video_path}]")
if not save_images:
    print(f"Please use `--save_images` if you also want to save the interpolated images, ")

writer.release()
