import torch
import argparse
from den import DEN
from torchvision.utils import save_image
from PIL import Image
import torchvision.transforms.functional as TF



parser = argparse.ArgumentParser()

print("PyTorch Version: ",torch.__version__)
parser.add_argument("--image", type=str, required=True)
parser.add_argument("--model_path", type=str, required=True)
args = parser.parse_args()


image = Image.open(args.image)
x = TF.to_tensor(image)

den_wts = args.model_path

den = DEN(den_wts)
depth = den(x)
save_image(depth, "/root/DEN/images/depth_img/" + args.model_path.split("/")[-1])




