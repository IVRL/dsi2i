import os
import math
import time
import yaml

import numpy as np
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torch.optim import lr_scheduler
from torchvision import transforms
import torch
import torch.nn as nn
import torchvision
import torchvision.utils as vutils
import torch.nn.init as init

# from semantics.vgg import Vgg16

def get_config(config):
    with open(config, 'r') as stream:
        return yaml.safe_load(stream)
    
def load_vgg16(model_dir):
    """ Use the model from https://github.com/abhiskk/fast-neural-style/blob/master/neural_style/utils.py """
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)
    if not os.path.exists(os.path.join(model_dir, 'vgg16.weight')):
        if not os.path.exists(os.path.join(model_dir, 'vgg16.pth')):
            raise Exception('VGG weights are missing at ', os.path.join(model_dir, 'vgg16.pth'))
        vgg_dict = torch.load(os.path.join(model_dir, 'vgg16.pth'))
        vgg = Vgg16()
        for (src, dst) in zip(vgg_dict.values(), vgg.parameters()):
            dst.data[:] = src
        torch.save(vgg.state_dict(), os.path.join(model_dir, 'vgg16.weight'))
    vgg = Vgg16()
    vgg.load_state_dict(torch.load(os.path.join(model_dir, 'vgg16.weight')))
    return vgg

def load_inception():
    model = torch.hub.load('pytorch/vision:v0.10.0', 'inception_v3', pretrained=True)
    model.eval()
    return model

def normalize(image):## this name used to refer to denormalization operation beforehand
    return image * 2 - 1

def denormalize(image):
    return (image + 1.0) / 2

def normalize_imagenet(image, normalized=False):
    if normalized:
        image = denormalize_gan(image)
    im = F.normalize(image, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    return im

def denormalize_imagenet(image):
    im = F.normalize(image, mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225], std=[1/0.229, 1/0.224, 1/0.225])
    return im

def normalize_clip(image):
    im = F.normalize(image, mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])
    return im

def denormalize_clip(image):
    im = F.normalize(image, mean=[-0.48145466/0.26862954, -0.4578275/0.26130258, -0.40821073/0.27577711],\
                        std=[1/0.26862954, 1/0.26130258, 1/0.27577711])
    return im

def get_pseudo(att, source_label, region=None):
    label = nn.functional.interpolate(source_label, size=att.shape[1:3])
    target_label = torch.einsum('abcd, acdmn -> abmn', label, att)
    return target_label

def flatten_att(att):
    return att.view(att.shape[0], att.shape[1]*att.shape[2],\
                               att.shape[3]*att.shape[4])

def get_hyperpixel(feats):
    index_max = max(range(len(feats)), key=lambda v: feats[v].shape[-1])
    shape = feats[index_max].shape[-2:]
    hpixel = torch.cat([nnF.interpolate(f, shape, mode='bilinear') for f in feats],dim=1)
    return hpixel

def normalize_feature(f):
    f_view = f.view(*f.shape[:2],-1)
    mean = f.mean(-1)
    f_norm = f - f.mean(-1,keepdims=True)
    f_normalized = f_norm.view(*f.shape)
    return f_normalized

def standardize_feature(f):
    f_view = f.view(*f.shape[:2],-1)
    f_std = f_view / (f_view.std(-1, keepdims=True) + 1e-8)
    f_standardized = f_std.view(*f.shape)
    return f_standardized

def normalize_feature_pair(f1, f2):
    f1_view = f1.view(*f1.shape[:2],-1)
    f2_view = f2.view(*f2.shape[:2],-1)
    f1_mean = f1_view.mean(-1,keepdims=True)
    f2_mean = f2_view.mean(-1,keepdims=True)
    mean = 0.5 * (f1_mean + f2_mean)
    f1_norm = f1_view - f1_mean
    f2_norm = f2_view - f2_mean
    f1_normalized = f1_norm.view(*f1.shape)
    f2_normalized = f2_norm.view(*f2.shape)
    return f1_normalized, f2_normalized

def get_attention(f_a, f_b):
    f_a = f_a/torch.sqrt(torch.sum(f_a**2, dim=1, keepdims=True))
    f_b = f_b/torch.sqrt(torch.sum(f_b**2, dim=1, keepdims=True))
    att = torch.einsum('abcd, abmn -> acdmn', f_a, f_b)
    att = torch.clamp(att, min=0)
    return att

def marg_att(att, inverse=False):
    marg = flatten_att(att).sum(dim=-1, keepdims=True).squeeze(0)
    if inverse:
        marg = 1 / (marg+1e-8)
    marg = marg / marg.sum()
    return marg

def marg_double_att(att_x, att_xy):
    #att_x[att_x.le(4.0*att_x.mean([-1, -2], keepdims=True))] = 0
    #att_xy[att_xy.le(4.0*att_xy.mean([-1, -2], keepdims=True))] = 0
    marg_x = flatten_att(att_x).sum(dim=-1, keepdims=True).squeeze(0)
    marg_xy = flatten_att(att_xy).sum(dim=-1, keepdims=True).squeeze(0)
    marg = marg_xy / (marg_x + 1e-8)
    marg = marg / marg.sum()
    return marg
    

def unbalanced_sinkhorn(x, y, eps, marg=False, demarg=True):
    xx = get_attention(x, x)
    xy = get_attention(x, y)
    yx = get_attention(y, x)
    yy = get_attention(y, y)
    
    sim = flatten_att(xy).squeeze(0)
    sim_xy = sim;# sim_xy[sim_xy.le(sim_xy.sum(1,keepdims=True)/sim_xy.gt(0).sum(1,keepdims=True))] = 0
    sim_yx = sim.T;# sim_yx[sim_yx.le(sim_yx.sum(1,keepdims=True)/sim_yx.gt(0).sum(1,keepdims=True))] = 0
    
    if marg:
        marg_xx = marg_att(xx, True)
        marg_xy = marg_att(xy, False)
        marg_xyx = marg_double_att(xx, xy)
        marg_yy = marg_att(yy, True)
        marg_yx = marg_att(yx, False)
        marg_yxy = marg_double_att(yy, yx)
    else:
        marg_xx = torch.ones((sim.shape[0],1), device=sim.get_device()) / sim.shape[0]
        marg_xy = torch.ones((sim.shape[0],1), device=sim.get_device()) / sim.shape[0]
        marg_xyx = torch.ones((sim.shape[0],1), device=sim.get_device()) / sim.shape[0]
        marg_yy = torch.ones((sim.shape[1],1), device=sim.get_device()) / sim.shape[1]
        marg_yx = torch.ones((sim.shape[1],1), device=sim.get_device()) / sim.shape[1]
        marg_yxy = torch.ones((sim.shape[1],1), device=sim.get_device()) / sim.shape[1]
    
    OTxy = apply_sinkhorn(sim_xy, eps, marg_xyx, None)[0]
    OTyx = apply_sinkhorn(sim_yx, eps, marg_yxy, None)[0]
    
    if demarg:
        OTxy, OTyx = (OTxy/OTxy.sum(0,keepdims=True)).view(xy.shape), (OTyx/OTyx.sum(0,keepdims=True)).view(yx.shape)
    return OTxy, OTyx

def apply_sinkhorn(C,eps,mu=None,nu=None, niter=1,tol=10e-9):
    if mu is None:
        mu = torch.ones((C.shape[0],1)) / C.shape[0]
    mu = mu.cuda()
    if nu is None:
        nu = torch.ones((C.shape[1],1)) / C.shape[1]
    nu = nu.cuda()

    C = 1 - C
    with torch.no_grad():
        epsilon = eps
        cnt = 0
        while True:
            PI,mu,nu,Err = perform_sinkhorn(C, epsilon, mu, nu, niter=niter, tol=tol)
            #PI = sinkhorn_stabilized(mu, nu, cost, reg=epsilon, numItermax=50, method='sinkhorn_stabilized', cuda=True)
            if not torch.isnan(PI).any():
                if cnt>0:
                    #print(cnt)
                    ...
                break
            else: # Nan encountered caused by overflow issue is sinkhorn
                epsilon *= 2.0
                #print(epsilon)
                cnt += 1
    PI = torch.clamp(PI, min=0)
    return PI, (Err, mu, nu)

def perform_sinkhorn(C,epsilon,mu,nu,a=[],warm=False,niter=1,tol=10e-9):
    """Main Sinkhorn Algorithm"""
    if not warm:
        a = torch.ones((C.shape[0],1)) / C.shape[0]
        a = a.cuda()

    K = torch.exp(-C/epsilon)

    Err = torch.zeros((niter,2)).cuda()
    for i in range(niter):
        b = nu/torch.mm(K.t(), a)
        if i%2==0:
            Err[i,0] = torch.norm(a*(torch.mm(K, b)) - mu, p=1)
            if i>0 and (Err[i,0]) < tol:
                break

        a = mu / torch.mm(K, b)

        if i%2==0:
            Err[i,1] = torch.norm(b*(torch.mm(K.t(), a)) - nu, p=1)
            if i>0 and (Err[i,1]) < tol:
                break

        PI = torch.mm(torch.mm(torch.diag(a[:,-1]),K), torch.diag(b[:,-1]))

    del a; del b; del K
    return PI,mu,nu,Err