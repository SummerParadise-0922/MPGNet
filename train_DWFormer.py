from torch.utils.data import DataLoader
import argparse
import torch
from tqdm import tqdm
import os
from torch import nn
from models import DWFormer
from datasets.dataset import pair_dataset
from utils import AveraegMeter
import torch.nn.functional as F
from torch.cuda.amp import autocast,GradScaler
import json
from torch.utils.tensorboard.writer import SummaryWriter

import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0,1,2,3'



parser = argparse.ArgumentParser()
parser.add_argument('--saved_dir',type=str,default='./saved',help='save model dir')
parser.add_argument('--fn_root',type=str,default='./data',help='data root path')
parser.add_argument('--log_dir',type=str,default='./logs',help='save log dir')
parser.add_argument('--category',type=str,default='Poled',help='Poled or Toled')
args = parser.parse_args()



def Train(train_data,net,Loss,optimizer,frozen_bn):
    avg_loss = AveraegMeter()
    torch.cuda.empty_cache()
    net.eval() if frozen_bn else net.train()
    for data in train_data:
        clear_img = data['clear'].cuda()
        degraded_img = data['degraded'].cuda()
        with autocast():
            predict_img = net(degraded_img).cuda()
            loss = Loss(clear_img,predict_img)
        avg_loss.update(loss.item())
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
    return avg_loss.avg

def Test(test_data,net):
    avg_psnr = AveraegMeter()
    torch.cuda.empty_cache()
    net.eval()
    for data in test_data:
        clear_img = data['clear'].cuda()
        degraded_img = data['degraded'].cuda()
        with torch.no_grad():
            predict_img = net(degraded_img).cuda()
            MSE = F.mse_loss(predict_img,clear_img,reduction='none').mean((1,2,3))
            psnr = 10 * torch.log10(1.0/MSE).mean()
        avg_psnr.update(psnr.item(),n=clear_img.size(0))
    return avg_psnr.avg


if __name__ == '__main__':
    setting_filename = os.path.join('configs','DWFormer.json')
    if not os.path.exists(setting_filename):
        setting_filename = os.path.join('configs','Restore_default.json')
    with open(setting_filename, 'r') as f:
        setting = json.load(f)

    train_dataset = pair_dataset(fn_root=args.fn_root,subdir='Train',category=args.category,is_Train=True)
    val_dataset   = pair_dataset(fn_root=args.fn_root,subdir='Val',category=args.category,is_Train=False)
    test_dataset  = pair_dataset(fn_root=args.fn_root,subdir='Test',category=args.category,is_Train=False)

    train_data = DataLoader(
            dataset=train_dataset,
            batch_size=setting['batch_size'],
            shuffle=True,
            num_workers=setting['num_workers'],
            pin_memory=False,
            drop_last=True
    )
    val_data = DataLoader(
            dataset=val_dataset,
            batch_size=setting['batch_size'],
            shuffle=False,
            num_workers=setting['num_workers'],
            pin_memory=False,
            drop_last=False
    )
    test_data = DataLoader(
            dataset=test_dataset,
            batch_size=setting['batch_size'],
            shuffle=False,
            num_workers=setting['num_workers'],
            pin_memory=False,
            drop_last=False
    )
    #---->model
    net = DWFormer()

    # net = net.to(device)
    net = nn.DataParallel(net).cuda()
    #---->criterion
    Loss  = nn.L1Loss().cuda()
    #---->optimizer
    optimizer = torch.optim.Adam(net.parameters(), lr = setting['lr'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,setting['epochs'],eta_min=setting['min_lr'])
    #---->train
    scaler = GradScaler()
    writer = SummaryWriter(log_dir=os.path.join(args.log_dir,'DWFormer'))
    for epoch in tqdm(range(1,setting['epochs']+1)):
        Frozen = ((setting['epochs'] - epoch) > setting['frozen_bn'])
        train_loss = Train(train_data,net,Loss,optimizer,Frozen)
        valid_psnr = Test(val_data,net)
        test_psnr = Test(test_data,net)
        writer.add_scalar('train_loss',train_loss,epoch)
        writer.add_scalar('valid_psnr',valid_psnr,epoch)
        writer.add_scalar('test_psnr',test_psnr,epoch)
        scheduler.step()
        if epoch % setting['saved_epoch'] == 0:
            # save model
            os.makedirs(args.saved_dir,exist_ok=True)
            torch.save(
                {'state_dict':net.state_dict()},
                os.path.join(args.saved_dir,'DWFormer_{}_{}_{}_{}.pth'.format(args.category,epoch,valid_psnr,test_psnr))
                )