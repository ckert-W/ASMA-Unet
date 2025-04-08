import math
import torch
import torch.nn as nn
import torch.nn.functional as F
#from .layers import *
import numbers
from einops import rearrange

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)

class BasicConv(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride, bias=True, norm=False, relu=True, transpose=False):
        super(BasicConv, self).__init__()
        if bias and norm:
            bias = False

        padding = kernel_size // 2
        layers = list()
        if transpose:
            padding = kernel_size // 2 -1
            layers.append(nn.ConvTranspose2d(in_channel, out_channel, kernel_size, padding=padding, stride=stride, bias=bias))
        else:
            layers.append(
                nn.Conv2d(in_channel, out_channel, kernel_size, padding=padding, stride=stride, bias=bias))
        if norm:
            layers.append(nn.BatchNorm2d(out_channel))
        if relu:
            layers.append(nn.ReLU(inplace=True))
        self.main = nn.Sequential(*layers)

    def forward(self, x):
        return self.main(x)

class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


class DFFN(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):

        super(DFFN, self).__init__()

        hidden_features = int(dim * ffn_expansion_factor)

        self.patch_size = 8

        self.dim = dim
        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=3, stride=1, padding=1,
                                groups=hidden_features * 2, bias=bias)

        self.fft = nn.Parameter(torch.ones((hidden_features * 2, 1, 1, self.patch_size, self.patch_size // 2 + 1)))
        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x_patch = rearrange(x, 'b c (h patch1) (w patch2) -> b c h w patch1 patch2', patch1=self.patch_size,
                            patch2=self.patch_size)
        x_patch_fft = torch.fft.rfft2(x_patch.float())
        x_patch_fft = x_patch_fft * self.fft
        x_patch = torch.fft.irfft2(x_patch_fft, s=(self.patch_size, self.patch_size))
        x = rearrange(x_patch, 'b c h w patch1 patch2 -> b c (h patch1) (w patch2)', patch1=self.patch_size,
                      patch2=self.patch_size)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)

        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x

class FSAS(nn.Module):
    def __init__(self, dim, bias):
        super(FSAS, self).__init__()

        self.to_hidden = nn.Conv2d(dim, dim * 6, kernel_size=1, bias=bias)
        self.to_hidden_dw = nn.Conv2d(dim * 6, dim * 6, kernel_size=3, stride=1, padding=1, groups=dim * 6, bias=bias)

        self.project_out = nn.Conv2d(dim * 2, dim, kernel_size=1, bias=bias)

        self.norm = LayerNorm(dim * 2, LayerNorm_type='WithBias')

        self.patch_size = 8

    def forward(self, x):
        hidden = self.to_hidden(x)

        q, k, v = self.to_hidden_dw(hidden).chunk(3, dim=1)

        q_patch = rearrange(q, 'b c (h patch1) (w patch2) -> b c h w patch1 patch2', patch1=self.patch_size,
                            patch2=self.patch_size)
        k_patch = rearrange(k, 'b c (h patch1) (w patch2) -> b c h w patch1 patch2', patch1=self.patch_size,
                            patch2=self.patch_size)
        q_fft = torch.fft.rfft2(q_patch.float())
        k_fft = torch.fft.rfft2(k_patch.float())

        out = q_fft * k_fft
        out = torch.fft.irfft2(out, s=(self.patch_size, self.patch_size))
        out = rearrange(out, 'b c h w patch1 patch2 -> b c (h patch1) (w patch2)', patch1=self.patch_size,
                        patch2=self.patch_size)

        out = self.norm(out)

        output = v * out
        output = self.project_out(output)

        return output

class TransformerBlock(nn.Module):
    def __init__(self, dim, ffn_expansion_factor=2.66, bias=False, LayerNorm_type='WithBias', att=False):
        super(TransformerBlock, self).__init__()

        self.att = att
        if self.att:
            self.norm1 = LayerNorm(dim, LayerNorm_type)
            self.attn = FSAS(dim, bias)

        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = DFFN(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        if self.att:
            x = x + self.attn(self.norm1(x))

        x = x + self.ffn(self.norm2(x))

        return x


###########fuse the aff&de n-1
class Fuse(nn.Module):
    def __init__(self, n_feat):
        super(Fuse, self).__init__()
        self.n_feat = n_feat
        self.att_channel = TransformerBlock(dim=n_feat * 2)

        self.conv = nn.Conv2d(n_feat * 2, n_feat * 2, 1, 1, 0)
        self.conv2 = nn.Conv2d(n_feat * 2, n_feat * 2, 1, 1, 0)

    def forward(self, enc, dnc):
        x = self.conv(torch.cat((enc, dnc), dim=1))
        x = self.att_channel(x)
        x = self.conv2(x)
        e, d = torch.split(x, [self.n_feat, self.n_feat], dim=1)
        output = e + d

        return output


## Overlapped image patch embedding with 3x3 Conv
class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c=3, embed_dim=48, bias=False):
        super(OverlapPatchEmbed, self).__init__()

        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, x):
        x = self.proj(x)

        return x

##Resizing modules
class Downsample(nn.Module):
    def __init__(self, n_feat):
        super(Downsample, self).__init__()

        self.body = nn.Sequential(nn.Upsample(scale_factor=0.5, mode='bilinear', align_corners=False),
                                  nn.Conv2d(n_feat, n_feat * 2, 3, stride=1, padding=1, bias=False))

    def forward(self, x):
        return self.body(x)

class Upsample(nn.Module):
    def __init__(self, n_feat):
        super(Upsample, self).__init__()

        self.body = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                                  nn.Conv2d(n_feat, n_feat // 2, 3, stride=1, padding=1, bias=False))

    def forward(self, x):
        return self.body(x)

##########################
##mimo_component
class SCM(nn.Module):
    def __init__(self, out_plane):
        super(SCM, self).__init__()
        self.main = nn.Sequential(
            BasicConv(3, out_plane//4, kernel_size=3, stride=1, relu=True),
            BasicConv(out_plane // 4, out_plane // 2, kernel_size=1, stride=1, relu=True),
            BasicConv(out_plane // 2, out_plane // 2, kernel_size=3, stride=1, relu=True),
            BasicConv(out_plane // 2, out_plane-3, kernel_size=1, stride=1, relu=True)
        )

        self.conv = BasicConv(out_plane, out_plane, kernel_size=1, stride=1, relu=False)

    def forward(self, x):
        x = torch.cat([x, self.main(x)], dim=1)
        return self.conv(x)


class FAM(nn.Module):
    def __init__(self, channel):
        super(FAM, self).__init__()
        self.merge = BasicConv(channel, channel, kernel_size=3, stride=1, relu=False)

    def forward(self, x1, x2):
        x = x1 * x2
        out = x1 + self.merge(x)
        return out
class AFF(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(AFF, self).__init__()
        self.conv = nn.Sequential(
            BasicConv(in_channel, out_channel, kernel_size=1, stride=1, relu=True),
            BasicConv(out_channel, out_channel, kernel_size=3, stride=1, relu=False)
        )

    def forward(self, x1, x2, x4):
        x = torch.cat([x1, x2, x4], dim=1)
        return self.conv(x)

#########################
class MIMOTransfromNet(nn.Module):
    def __init__(self,
                 inp_channels=3,
                 out_channels=3,
                 dim=32,
                 num_blocks=[1, 1, 2, 3],
                 num_refinement_blocks=4,
                 ffn_expansion_factor=3,
                 bias=False,
                 ):
        super(MIMOTransfromNet,self).__init__()

        self.patch_embed=OverlapPatchEmbed(inp_channels,dim)

        self.Encoder1=nn.Sequential(*[
            TransformerBlock(dim=dim,ffn_expansion_factor=ffn_expansion_factor,bias=bias) for i in
            range(num_blocks[0])])

        self.down1to2=Downsample(dim)
        self.scm2 = SCM(int(dim * 2))
        self.fm2 = FAM(int(dim * 2))
        self.Encoder2=nn.Sequential(*[
            TransformerBlock(dim=int(dim*2**1),ffn_expansion_factor=ffn_expansion_factor,bias=bias) for i in
            range(num_blocks[1])])

        self.down2to3 = Downsample(int(dim * 2 ** 1))
        self.scm3 = SCM(int(dim * 2**2))
        self.fm3 = FAM(int(dim * 2**2))
        self.Encoder3 = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 2), ffn_expansion_factor=ffn_expansion_factor,bias=bias) for i in
            range(num_blocks[2])])

        self.Decoder3=nn.Sequential(*[
            TransformerBlock(dim=int(dim*2**2),ffn_expansion_factor=ffn_expansion_factor,bias=bias,att=True) for i in
            range(num_blocks[2])])


        self.aff2 = AFF(dim * 7, dim * 2)
        self.up3to2=Upsample(int(dim*2**2))
        self.fu2=Fuse(dim*2)
        self.Decoder2 = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 1), ffn_expansion_factor=ffn_expansion_factor, bias=bias, att=True) for i in
            range(num_blocks[1])])

        self.aff1 = AFF(dim * 7 , dim)
        self.up2to1 = Upsample(int(dim * 2 ** 1))
        self.fu1 = Fuse(dim)
        self.Decoder1 = nn.Sequential(*[
            TransformerBlock(dim=int(dim), ffn_expansion_factor=ffn_expansion_factor, bias=bias, att=True) for i in
            range(num_blocks[0])])

        self.refinement=nn.Sequential(*[
            TransformerBlock(dim=int(dim),ffn_expansion_factor=ffn_expansion_factor,bias=bias,att=False) for i in
            range(num_refinement_blocks)])

        self.refinement3 = nn.Sequential(*[
            TransformerBlock(dim=int(dim*2**2), ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, att=False) for i in range(num_refinement_blocks)])
        self.refinement2 = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2), ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, att=False) for i in range(num_refinement_blocks)])
        self.refinement1 = nn.Sequential(*[
            TransformerBlock(dim=int(dim), ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, att=False) for i in range(num_refinement_blocks)])

        self.output3 = nn.Conv2d(int(dim*2**2), out_channels, kernel_size=3, stride=1, padding=1, bias=bias)
        self.output2 = nn.Conv2d(int(dim*2), out_channels, kernel_size=3, stride=1, padding=1, bias=bias)
        self.output1 = nn.Conv2d(int(dim), out_channels, kernel_size=3, stride=1, padding=1, bias=bias)


    def forward(self,x):
        x2=F.interpolate(x,scale_factor=0.5)
        x3=F.interpolate(x2,scale_factor=0.5)
        x_2=self.scm2(x2)
        x_3=self.scm3(x3)

        outputs = list()
        x1_ec_in = self.patch_embed(x)
        x1_ec_out=self.Encoder1(x1_ec_in)
        x1_res1=x1_ec_out
        x1_res2=F.interpolate(x1_ec_out,scale_factor=0.5)
        x1_ec_out=self.down1to2(x1_ec_out)

        x2_ec_in=self.fm2(x1_ec_out,x_2)
        x2_ec_out=self.Encoder2(x2_ec_in)
        x2_res1=F.interpolate(x2_ec_out,scale_factor=2)
        x2_res2=x2_ec_out
        x2_ec_out = self.down2to3(x2_ec_out)

        x3_ec_in=self.fm3(x2_ec_out,x_3)
        x3_ec_out=self.Encoder3(x3_ec_in)
        x3_res1=F.interpolate(x3_ec_out,scale_factor=4)
        x3_res2=F.interpolate(x3_ec_out,scale_factor=2)

        x1_dc=self.aff1(x1_res1,x2_res1,x3_res1)
        x2_dc=self.aff2(x1_res2,x2_res2,x3_res2)

        x3_dc_out=self.Decoder3(x3_ec_out)
        x3_out=self.refinement3(x3_dc_out)
        x3_dc_out=self.up3to2(x3_dc_out)
        x3_out=self.output3(x3_out)
        outputs.append(x3_out+x3)

        x2_dc_in=self.fu2(x2_dc,x3_dc_out)
        x2_dc_out=self.Decoder2(x2_dc_in)
        x2_out=self.refinement2(x2_dc_out)
        x2_dc_out = self.up2to1(x2_dc_out)
        x2_out = self.output2(x2_out)
        outputs.append(x2_out+x2)

        x1_dc_in=self.fu1(x2_dc_out,x1_dc)
        x1_dc_out=self.Decoder1(x1_dc_in)
        x1_out = self.refinement1(x1_dc_out)
        x1_out = self.output1(x1_out)
        outputs.append(x1_out + x)

        return outputs


def build_net(model_name):
    class ModelError(Exception):
        def __init__(self, msg):
            self.msg = msg

        def __str__(self):
            return self.msg

    if model_name == "MIMO-Transform-UNET":
        return MIMOTransfromNet()
    raise ModelError('Wrong Model!\nYou should choose MIMO-Transform-UNET.')