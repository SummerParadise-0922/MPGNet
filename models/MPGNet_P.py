import torch.nn.functional as F
import torch
from torch import nn
import torch.distributions as torch_distb

import functools



class Light_rate(nn.Module):
    def __init__(self,in_ch=3,n_feat=32) -> None:
        super(Light_rate,self).__init__()
        self.feat_ext = nn.Sequential(
            nn.Conv2d(in_ch,n_feat,kernel_size=3,stride=4,padding=1),
            nn.Conv2d(n_feat,n_feat,kernel_size=3,stride=2,padding=1),
            nn.Conv2d(n_feat,n_feat*2,kernel_size=3,stride=2,padding=1),
            nn.Conv2d(n_feat*2,n_feat*2,kernel_size=3,stride=2,padding=1),
            nn.Conv2d(n_feat*2,n_feat,kernel_size=3,stride=2,padding=1)
        )
        self.to_light_rate = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(1),
                nn.Linear(n_feat,n_feat*4),
                nn.ReLU(),
                nn.Linear(n_feat*4,3),
                nn.Sigmoid()
        )
    def forward(self,x):
        feat_ext = self.feat_ext(x)
        light_rate = self.to_light_rate(feat_ext).unsqueeze(-1).unsqueeze(-1).expand(x.shape)
        return light_rate


class ResnetBlock(nn.Module):
	def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
		super(ResnetBlock, self).__init__()

		padAndConv = {
			'reflect': [
                nn.ReflectionPad2d(1),
                nn.Conv2d(dim, dim, kernel_size=3, bias=use_bias)],
			'replicate': [
                nn.ReplicationPad2d(1),
                nn.Conv2d(dim, dim, kernel_size=3, bias=use_bias)],
			'zero': [
                nn.Conv2d(dim, dim, kernel_size=3, padding=1, bias=use_bias)]
		}

		try:
			blocks = padAndConv[padding_type] + [
				norm_layer(dim),
				nn.ReLU(True)
            ] + [
				nn.Dropout(0.5)
			] if use_dropout else [] + padAndConv[padding_type] + [
				norm_layer(dim)
			]
		except:
			raise NotImplementedError('padding [%s] is not implemented' % padding_type)

		self.conv_block = nn.Sequential(*blocks)

	def forward(self, x):
		out = x + self.conv_block(x)
		return out

class Blur_Generator(nn.Module):
    def __init__(
            self, input_nc=3, output_nc=3, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False,
            n_blocks=9,padding_type='reflect'):
        assert (n_blocks >= 0)
        super(Blur_Generator, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf

        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        model = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
            norm_layer(ngf),
            nn.ReLU(True)
        ]


        model += [
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=use_bias),
            norm_layer(128),
            nn.ReLU(True),

            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=use_bias),
            norm_layer(256),
            nn.ReLU(True)
        ]
        for i in range(n_blocks):
            model += [
                ResnetBlock(256, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)
            ]

        model += [
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1, bias=use_bias),
            norm_layer(128),
            nn.ReLU(True),

            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1, bias=use_bias),
            norm_layer(64),
            nn.ReLU(True),
        ]

        model += [
            nn.ReflectionPad2d(3),
            nn.Conv2d(64, output_nc, kernel_size=7, padding=0),
            nn.Tanh()
        ]

        self.model = nn.Sequential(*model)

    def forward(self, input):
        blur = self.model(input)
        output = torch.clamp(input + blur, min=0, max=1)
        return output


class ResBlock(nn.Module):
    def __init__(self, n_ch_in, ksize=3, bias=True):
        self.n_ch = n_ch_in
        self.bias = bias

        super().__init__()

        layer = []
        layer.append(nn.Conv2d(self.n_ch, self.n_ch, ksize,
                               padding=(ksize // 2), bias=self.bias,
                               padding_mode='reflect'))
        layer.append(nn.PReLU())
        layer.append(nn.Conv2d(self.n_ch, self.n_ch, ksize,
                               padding=(ksize // 2), bias=self.bias,
                               padding_mode='reflect'))

        self.body = nn.Sequential(*layer)

    def forward(self, x):
        return x + self.body(x)



d = 1e-6
class Noise_Generator(nn.Module):
    def __init__(self, n_ch_in=3, n_ch_out=3, n_r=32):
        self.n_ch_unit = 64         # number of base channel
        self.n_ext = 5              # number of residual blocks in feature extractor
        self.n_block_indep = 3      # number of residual blocks in independent module
        self.n_block_dep = 2        # number of residual blocks in dependent module

        self.n_ch_in = n_ch_in      # number of input channels
        self.n_ch_out = n_ch_out    # number of output channels
        self.n_r = n_r        # length of r vector

        super().__init__()

        # feature extractor
        self.ext_head = nn.Sequential(
            nn.Conv2d(n_ch_in, self.n_ch_unit, 3,
                      padding=1, bias=True,
                      padding_mode='reflect'),
            nn.PReLU(),
            nn.Conv2d(self.n_ch_unit, self.n_ch_unit * 2, 3,
                      padding=1, bias=True,
                      padding_mode='reflect')
        )
        self.ext_merge = nn.Sequential(
            nn.Conv2d((self.n_ch_unit * 2) + self.n_r, 2 * self.n_ch_unit, 3,
                      padding=1, bias=True,
                      padding_mode='reflect'),
            nn.PReLU()
        )
        self.ext = nn.Sequential(
            *[ResBlock(2 * self.n_ch_unit) for _ in range(self.n_ext)]
        )

        # pipe-indep
        self.indep_merge = nn.Conv2d(self.n_ch_unit, self.n_ch_unit, 1,
                                     padding=0, bias=True,
                                     padding_mode='reflect')
        pipe_indep_1_layer = []
        pipe_indep_1_layer.append(ResBlock(self.n_ch_unit, ksize=1, bias=False))
        pipe_indep_1_layer.append(ResBlock(self.n_ch_unit, ksize=1, bias=False))
        pipe_indep_1_layer.append(ResBlock(self.n_ch_unit, ksize=1, bias=False))
        self.pipe_indep_1 = nn.Sequential(
            *pipe_indep_1_layer
        )
        pipe_indep_3_layer = []
        pipe_indep_3_layer.append(ResBlock(self.n_ch_unit, ksize=1, bias=False))
        pipe_indep_3_layer.append(ResBlock(self.n_ch_unit, ksize=3, bias=False))
        pipe_indep_3_layer.append(ResBlock(self.n_ch_unit, ksize=1, bias=False))
        self.pipe_indep_3 = nn.Sequential(
            *pipe_indep_3_layer
        )

        # pipe-dep
        self.dep_merge = nn.Conv2d(self.n_ch_unit, self.n_ch_unit, 1,
                                   padding=0, bias=True,
                                   padding_mode='reflect')
        pipe_dep_1_layer = []
        pipe_dep_1_layer.append(ResBlock(self.n_ch_unit, ksize=1, bias=False))
        pipe_dep_1_layer.append(ResBlock(self.n_ch_unit, ksize=1, bias=False))
        pipe_dep_1_layer.append(ResBlock(self.n_ch_unit, ksize=1, bias=False))
        self.pipe_dep_1 = nn.Sequential(
            *pipe_dep_1_layer
        )
        pipe_dep_3_layer = []
        pipe_dep_3_layer.append(ResBlock(self.n_ch_unit, ksize=1, bias=False))
        pipe_dep_3_layer.append(ResBlock(self.n_ch_unit, ksize=3, bias=False))
        pipe_dep_3_layer.append(ResBlock(self.n_ch_unit, ksize=1, bias=False))
        self.pipe_dep_3 = nn.Sequential(
            *pipe_dep_3_layer
        )

        # T tail
        self.T_tail = nn.Conv2d(self.n_ch_unit, self.n_ch_out, 1,
                                padding=0, bias=True,
                                padding_mode='reflect')



        # quantization part add last
        self.q_ext = nn.Sequential(
            nn.Conv2d(n_ch_in, self.n_ch_unit, 3,
                      padding=1, bias=True,
                      padding_mode='reflect'),
            nn.PReLU(),
            nn.Conv2d(self.n_ch_unit, 2 * self.n_ch_unit, 3,
                      padding=1, bias=True,
                      padding_mode='reflect')
        )
        self.quantization_merge = nn.Conv2d(self.n_ch_unit, self.n_ch_unit, 1,
                                   padding=0, bias=True,
                                   padding_mode='reflect')
        pipe_quantization_1_layer = []
        pipe_quantization_1_layer.append(ResBlock(self.n_ch_unit, ksize=1, bias=False))
        pipe_quantization_1_layer.append(ResBlock(self.n_ch_unit, ksize=1, bias=False))
        pipe_quantization_1_layer.append(ResBlock(self.n_ch_unit, ksize=1, bias=False))
        self.pipe_quantization_1 = nn.Sequential(
            *pipe_quantization_1_layer
        )
        pipe_quantization_3_layer = []
        pipe_quantization_3_layer.append(ResBlock(self.n_ch_unit, ksize=1, bias=False))
        pipe_quantization_3_layer.append(ResBlock(self.n_ch_unit, ksize=3, bias=False))
        pipe_quantization_3_layer.append(ResBlock(self.n_ch_unit, ksize=1, bias=False))
        self.pipe_quantization_3 = nn.Sequential(
            *pipe_quantization_3_layer
        )
        self.q_tail = nn.Conv2d(self.n_ch_unit, self.n_ch_out, 1,
                        padding=0, bias=True,
                        padding_mode='reflect')

    def forward(self, x, r_vector=None):
        (N, C, H, W) = x.size()

        # r map
        if r_vector is None:
            r_vector = torch.randn(N, self.n_r)
        r_map = r_vector.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, H, W)
        r_map = r_map.float().detach()
        r_map = r_map.to(x.device)

        # feat extractor
        feat_CL = self.ext_head(x)
        list_cat = [feat_CL, r_map]
        feat_CL = self.ext_merge(torch.cat(list_cat, 1))
        feat_CL = self.ext(feat_CL)

        # make initial dep noise feature
        normal_scale = F.relu(feat_CL[:, self.n_ch_unit:, :, :]) + d
        get_feat_dep = torch_distb.Normal(loc=feat_CL[:, :self.n_ch_unit, :, :],
                                          scale=normal_scale)
        feat_noise_dep = get_feat_dep.rsample().to(x.device)

        # make initial indep noise feature
        feat_noise_indep = torch.rand_like(feat_noise_dep, requires_grad=True)
        feat_noise_indep = feat_noise_indep.to(x.device)

        # =====

        # pipe-indep
        list_cat = [feat_noise_indep]
        feat_noise_indep = self.indep_merge(torch.cat(list_cat, 1))
        feat_noise_indep = self.pipe_indep_1(feat_noise_indep) + \
            self.pipe_indep_3(feat_noise_indep)

        # pipe-dep
        list_cat = [feat_noise_dep]
        feat_noise_dep = self.dep_merge(torch.cat(list_cat, 1))
        feat_noise_dep = self.pipe_dep_1(feat_noise_dep) + \
            self.pipe_dep_3(feat_noise_dep)

        feat_noise = feat_noise_indep + feat_noise_dep
        noise = self.T_tail(feat_noise)

        # add quantization 
        q_feat = self.q_ext(x + noise)
        q_low = torch.min(q_feat[:, :self.n_ch_unit, :, :],q_feat[:, :self.n_ch_unit, :, :])
        q_high = torch.max(q_feat[:, :self.n_ch_unit, :, :],q_feat[:, :self.n_ch_unit, :, :]) + 1e-6
        get_feat_q = torch_distb.Uniform(
            low=q_low,
            high=q_high
            )
        feat_noise_q = get_feat_q.rsample().to(x.device)
        feat_noise_q = self.quantization_merge(feat_noise_q)
        feat_noise_q = self.pipe_quantization_1(feat_noise_q) + self.pipe_quantization_3(feat_noise_q)

        q_noise = self.q_tail(feat_noise_q)


        return x + noise + q_noise


class MPGNet_P(nn.Module):
    def __init__(self) -> None:
        super(MPGNet_P,self).__init__()
        self.light_rate = Light_rate()
        self.psf = Blur_Generator()
        self.add_noise = Noise_Generator()

    def forward(self,x):
        light_rate = self.light_rate(x)
        scaled = light_rate * x
        blured_x = self.psf(scaled)
        out = self.add_noise(blured_x)
        return out
