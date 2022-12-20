import os
import torch
from modules import UNet
from ddpm import Diffusion
from utils import plot_images


# Configuration: 'nt' is for windows
local = os.name == 'nt'

gpu_path = '/gpu-data2/gmil'
path = os.path.dirname(os.path.realpath(__file__))

if not local:
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)


model = UNet().to(device)
ckpt = torch.load("models/unconditional_ckpt.pt")
model.load_state_dict(ckpt)
diffusion = Diffusion(img_size=64, device=device)
x = diffusion.sample(model, n=4)
plot_images(x)