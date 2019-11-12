import torch
import argparse
from den import DEN
from fdc import FDC
from torchvision.utils import save_image
from PIL import Image
import torchvision.transforms.functional as TF
import transforms_nyu
from torchvision.transforms import Compose
import pickle


parser = argparse.ArgumentParser()
parser.add_argument("--image", type=str, required=True)
parser.add_argument("--model_path", type=str, required=True)
parser.add_argument("--fdc_weight_path", type=str, required=True)


class FDCPredictor:
    def __init__(self, den_path, fdc_path):
        self.den_model_pth = den_path
        self.den = DEN()
        self.den.load_state_dict(torch.load(self.den_model_pth), strict=False)

        self.fdc = FDC(self.den)
        self.fdc.load_weights(fdc_path)

        self.crop_ratios = [0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1]
        self.transform = Compose([
                                transforms_nyu.Normalize(),
                                transforms_nyu.FDCPreprocess(self.crop_ratios)
                        ])
    
    def prediction(self, img_path):
        with open(img_path, 'rb') as f_img:
            image = pickle.load(f_img)
        nyu_dict = {'image': image, 'depth': image}
        cropped = self.transform(nyu_dict)['stacked_images']
        cropped = cropped.unsqueeze(0)
        bsize, crops, c, h, w = cropped.size()
        print(bsize, crops, c, h, w)
        return self.fdc(cropped)[0]
    
    def save(self, img, des_path):
        img = Image.fromarray(img.numpy())
        img.save("/root/DEN/images/depth_img/" + des_path.split(".")[0] + ".tiff")
        print("/root/DEN/images/depth_img/" + des_path.split(".")[0] + ".tiff" + "saved!")

if __name__ == "__main__":
    args = parser.parse_args()

    predictor = FDCPredictor(args.model_path, args.fdc_weight_path)
    result = predictor.prediction(args.image)
    predictor.save(result, args.image.split("/")[-1])
