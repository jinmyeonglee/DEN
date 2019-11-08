from torchsummary import summary
wts = './models/den_gen2_v2/154_model.pt'
model = torch.load(wts)
summary(
