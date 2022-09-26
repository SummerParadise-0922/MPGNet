import torch
from torch import nn

class Pair_Loss(nn.Module):
    def __init__(self):
        super(Pair_Loss,self).__init__()
        self.pair = nn.L1Loss()
    def forward(self,predict_img,ori_img):
        loss = self.pair(predict_img,ori_img)
        return loss

class LS_D_Loss(nn.Module):
    def __init__(self) -> None:
        super(LS_D_Loss,self).__init__()
        self.mse = nn.MSELoss()

    def forward(self,r_img,f_img):
        real_loss_img = 1/2 * self.mse(r_img, torch.ones_like(r_img))
        fake_loss_img = 1/2 * self.mse(f_img, torch.zeros_like(f_img))
        loss = real_loss_img + fake_loss_img
        return loss

class LS_G_Loss(nn.Module):
    def __init__(self) -> None:
        super(LS_G_Loss,self).__init__()
        self.mse = nn.MSELoss()
    def forward(self,f_img):
        loss_img = 1/2 * self.mse(f_img, torch.ones_like(f_img))
        loss = loss_img
        return loss


