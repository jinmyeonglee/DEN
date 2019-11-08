import torch
from den import DEN

img = ""
den_wts = './models/den_gen2_v2_soomth_2/124_model.pt'
den = DEN(den_wts)
depth = den(img)
