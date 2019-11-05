import torch
from torch.nn import Module
from torch.autograd import Variable
from torch.nn import functional as F

class LaplacianLayer(Module):
    def __init__(self):
        super(LaplacianLayer, self).__init__()
        w_nom = torch.cuda.FloatTensor([[0, -1, 0], [-1, 4, -1], [0, -1, 0]]).view(1,1,3,3)
        w_den = torch.cuda.FloatTensor([[0, 1, 0], [1, 4, 1], [0, 1, 0]]).view(1,1,3,3)
        self.register_buffer('w_nom', w_nom)
        self.register_buffer('w_den', w_den)

    def forward(self, input, do_normalize=True):
        assert(input.dim() == 2 or input.dim()==3 or input.dim()==4)
        input_size = input.size()
        if input.dim()==4:
            x = input.view(input_size[0]*input_size[1], 1,
                            input_size[2], input_size[3])
        elif input.dim()==3:
            x = input.unsqueeze(1)
        else:
            x = input.unsqueeze(0).unsqueeze(0)
        x_nom = F.conv2d(input=x,
                        weight=Variable(self.w_nom),
                        stride=1,
                        padding=0)
        if do_normalize:
            x_den = F.conv2d(input=x,
                        weight=Variable(self.w_den),
                        stride=1,
                        padding=0)
            # x_den = x.std() + 1e-5
            x = (x_nom.abs()/x_den)
        else:
            x = x_nom.abs()
        if input.dim() == 4:
            return x.view(input_size[0], input_size[1], input_size[2]-2, input_size[3]-2)
        elif input.dim() == 3:
            return x.squeeze(1)
        elif input.dim() == 2:
            return x.squeeze(0).squeeze(0)


class DBELoss(Module):
    def __init__(self):
        super(DBELoss, self).__init__()
        self.laplacian_func = LaplacianLayer()
        self.lambda_S=.5

    def g(self, d, a1, a2):
            return a1*d + 0.5*a2*(d**2)
        
    def forward(self, d_hat, d, a1=1.5, a2=-0.1):
        
        g_d_hat = self.g(d_hat, a1, a2)    
        g_d = self.g(d, a1, a2)
        dbe = 0.5 * F.mse_loss(g_d_hat, g_d, reduction='sum')
        laplacian_smoothness_cost = self.compute_image_aware_laplacian_smoothness_cost(g_d, g_d_hat)
        
        return dbe + self.lambda_S * laplacian_smoothness_cost
    
    def compute_image_aware_laplacian_smoothness_cost(self, depth, img):
        print('before lap', img.shape, depth.shape)
        img_lap = self.laplacian_func(img/255, do_normalize=False)
        depth_lap = self.laplacian_func(depth, do_normalize=False)
        print(img_lap.shape, depth_lap.shape)
        print((-img_lap.mean()).exp().shape)
        x = (-img_lap.mean()).exp() * (depth_lap)
        return x.mean()
