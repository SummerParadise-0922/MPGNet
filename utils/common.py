from torch import nn

class AveraegMeter(object):
    def __init__(self):
        super(AveraegMeter,self).__init__()
        self.count = 0
        self.val = 0
        self.sum = 0
        self.avg = 0
    def reset(self):
        self.count = 0
        self.val = 0
        self.sum = 0
        self.avg = 0
    def update(self,x,n=1):
        self.count += n
        self.val = x
        self.sum += x
        self.avg = self.sum/self.count


def weight_init(m):
    if isinstance(m,nn.Conv2d):
        nn.init.trunc_normal_(m.weight.data,mean=0,std=0.02)
        if m.bias is not None:
            nn.init.constant_(m.bias.data,0)

    elif isinstance(m,nn.Linear):
        nn.init.trunc_normal_(m.weight.data,mean=0,std=0.02)
        if m.bias is not None:
            nn.init.constant_(m.bias.data,0.5)