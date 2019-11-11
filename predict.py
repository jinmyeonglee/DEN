import torch
import argparse
from den import DEN
from torchvision.utils import save_image

parser = argparse.ArgumentParser()

print("PyTorch Version: ",torch.__version__)
parser.add_argument("--image", type=str, required=True)
parser.add_argument("--model_path", type=str, required=True)
args = parser.parse_args()

img = args.image
den_wts = args.model_path
den = DEN(den_wts)
depth = den(img)
save_image(depth, "/root/DEN/images/depth_img" + args.model_path.split("/")[-1])




