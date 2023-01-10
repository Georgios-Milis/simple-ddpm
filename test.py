import os
import torch
import numpy as np
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
diffusion = Diffusion(img_size=96, device=device)
x = diffusion.sample(model, n=1)
plot_images(x)

# def pos_encoding(t, channels):
#     inv_freq = 1.0 / (
#         10000
#         ** (torch.arange(0, channels, 2, device=device).float() / channels)
#     )
#     pos_enc_a = torch.sin(t.repeat(1, channels // 2) * inv_freq)
#     pos_enc_b = torch.cos(t.repeat(1, channels // 2) * inv_freq)
#     pos_enc = torch.cat([pos_enc_a, pos_enc_b], dim=-1)
#     return pos_enc

# Show time embeddings - they are the cat'ted sin and cos again!!!
# import matplotlib.pyplot as plt
# time = torch.arange(100).to(device)
# PE = np.array([pos_encoding(t, 512).detach().tolist() for _, t in enumerate(time)]).transpose(0, 2, 1)
# cax = plt.matshow(PE)
# plt.gcf().colorbar(cax)
# plt.show()
