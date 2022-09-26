import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0,1,2,3'
from torch.utils.data import DataLoader
import argparse
import torch
from tqdm import tqdm

from models import *
from utils import LS_D_Loss, LS_G_Loss, Pair_Loss, weight_init
from datasets import pair_dataset
import json

parser = argparse.ArgumentParser()
parser.add_argument('--saved_dir',type=str,default='./saved',help='save model dir')
parser.add_argument('--fn_root',type=str,default='./data',help='data root path')
parser.add_argument('--category',type=str,default='Poled',help='Poled or Toled')
args = parser.parse_args()

def train_G(data,netG,netD,G_Loss,Pair_Loss,optimizerG):
    netG.train()
    netD.train()
    clear_img = data['clear'].cuda()
    degraded_img = data['degraded'].cuda()
    generated_img = netG(clear_img)
    f_img = netD(torch.cat((generated_img,clear_img),dim=1))
    optimizerG.zero_grad()
    g_loss = G_Loss(f_img)
    pair_loss = Pair_Loss(generated_img,degraded_img)
    total_loss = g_loss + pair_loss * 10
    total_loss.backward()
    optimizerG.step()
    return 0

def train_D(data,netG,netD,D_Loss,optimizerD):
    netG.train()
    netD.train()
    clear_img = data['clear'].cuda()
    degraded_img = data['degraded'].cuda()
    generated_img = netG(clear_img)
    generated_img = generated_img.detach()
    f_img = netD(torch.cat((generated_img,clear_img),dim=1))
    r_img = netD(torch.cat((degraded_img,clear_img),dim=1))
    optimizerD.zero_grad()
    d_loss = D_Loss(r_img,f_img)
    d_loss.backward()
    optimizerD.step()
    return 0

if __name__ == '__main__':
    setting_filename = os.path.join('configs','MPGNet.json')
    if not os.path.exists(setting_filename):
        setting_filename = os.path.join('configs','Generate_default.json')
    with open(setting_filename, 'r') as f:
        setting = json.load(f)

    train_dataset = pair_dataset(fn_root=args.fn_root,subdir='Train',category=args.category,is_Train=True)
    train_data = DataLoader(
            dataset=train_dataset,
            batch_size=setting['batch_size'],
            shuffle=True,
            num_workers=setting['num_workers'],
            pin_memory=True,
            drop_last=False
    )
    #---->model
    # netG = MPGNet_P() # for Poled
    netG = MPGNet_T() # for Toled
    netD = UNetDiscriminatorSN()

    # initialization
    netG.apply(weight_init)
    netD.apply(weight_init)
    netG = netG.cuda()
    netD = netD.cuda()

    netG = torch.nn.DataParallel(netG).cuda()
    netD = torch.nn.DataParallel(netD).cuda()

    #---->criterion
    D_Loss = LS_D_Loss().cuda()
    G_Loss = LS_G_Loss().cuda()
    pair_loss = Pair_Loss().cuda()

    #---->optimizer
    optimizerG = torch.optim.Adam(netG.parameters(), lr = setting['G_lr'], betas=(setting['beta1'],setting['beta2']))
    optimizerD = torch.optim.Adam(netD.parameters(), lr = setting['D_lr'], betas=(setting['beta1'],setting['beta2']))
    schedulerG = torch.optim.lr_scheduler.CosineAnnealingLR(optimizerG,setting['epochs'],setting['min_G_lr'])
    schedulerD = torch.optim.lr_scheduler.CosineAnnealingLR(optimizerD,setting['epochs'],setting['min_D_lr'])

    #---->train
    for epoch in tqdm(range(1,setting['epochs']+1)):
        for data in tqdm(train_data):
            train_D(data,netG,netD,D_Loss,optimizerD)
            train_D(data,netG,netD,D_Loss,optimizerD)
            train_D(data,netG,netD,D_Loss,optimizerD)
            train_G(data,netG,netD,G_Loss,pair_loss,optimizerG)
        schedulerD.step()
        schedulerG.step()
        if epoch % setting['saved_epoch'] == 0:
            # save model
            os.makedirs(args.saved_dir,exist_ok=True)
            torch.save(
                {'state_dict':netG.state_dict()},
                os.path.join(args.saved_dir, 'MPGNet_{}_{}.pth'.format(args.category, epoch))
                )