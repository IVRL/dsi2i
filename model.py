"""
Copyright (C) 2017 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
import os

import torch
import torch.nn as nn
import torch.nn.functional as nnF

import semantics.segmenter as segmenter
from .networks import AdaINGen, MsImageDis, VAEGen
from utils import weights_init, get_model_list, vgg_preprocess, load_vgg16, get_scheduler
from utils import aggregate_style, expand_style, labels_vector, compute_sigma, fill_style, translate_style, swap_style

class MUNIT_Trainer(nn.Module):
    def __init__(self, hyperparameters):
        super(MUNIT_Trainer, self).__init__()
        lr = hyperparameters['lr']
        # Initiate the networks
        self.gen_a = AdaINGen(hyperparameters['input_dim_a'], hyperparameters['gen'])  # auto-encoder for domain a
        self.gen_b = AdaINGen(hyperparameters['input_dim_b'], hyperparameters['gen'])  # auto-encoder for domain b
        self.dis_a = MsImageDis(hyperparameters['input_dim_a'], hyperparameters['dis'])  # discriminator for domain a
        self.dis_b = MsImageDis(hyperparameters['input_dim_b'], hyperparameters['dis'])  # discriminator for domain b
        self.instancenorm = nn.InstanceNorm2d(512, affine=False)
        self.style_dim = hyperparameters['gen']['style_dim']

        # fix the noise used in sampling
        display_size = int(hyperparameters['display_size'])
        self.s_a = torch.randn(display_size, self.style_dim, 1, 1).cuda()
        self.s_b = torch.randn(display_size, self.style_dim, 1, 1).cuda()
        

        # Setup the optimizers
        beta1 = hyperparameters['beta1']
        beta2 = hyperparameters['beta2']
        dis_params = list(self.dis_a.parameters()) + list(self.dis_b.parameters())
        gen_params = list(self.gen_a.parameters()) + list(self.gen_b.parameters())
        self.dis_opt = torch.optim.Adam([p for p in dis_params if p.requires_grad],
                                        lr=lr, betas=(beta1, beta2), weight_decay=hyperparameters['weight_decay'])
        self.gen_opt = torch.optim.Adam([p for p in gen_params if p.requires_grad],
                                        lr=lr, betas=(beta1, beta2), weight_decay=hyperparameters['weight_decay'])
        self.dis_scheduler = get_scheduler(self.dis_opt, hyperparameters)
        self.gen_scheduler = get_scheduler(self.gen_opt, hyperparameters)

        # Network weight initialization
        self.apply(weights_init(hyperparameters['init']))
        self.dis_a.apply(weights_init('gaussian'))
        self.dis_b.apply(weights_init('gaussian'))

        # Load VGG model if needed
        if 'vgg_w' in hyperparameters.keys() and hyperparameters['vgg_w'] > 0:
            self.vgg = load_vgg16('semantics/weights')
            self.vgg.eval()
            for param in self.vgg.parameters():
                param.requires_grad = False

        # Segmentation networks
        #self.semA = segmenter.Segmenter(hyperparameters['sem'])
        #self.semB = segmenter.Segmenter(hyperparameters['sem'])
        #self.semC = segmenter.Segmenter(hyperparameters['sem'], real=False)

    def recon_criterion(self, input, target):
        return torch.mean(torch.abs(input - target))

    def forward(self, x_a, x_b):
        self.eval()
        s_a = self.s_a
        s_b = self.s_b
        c_a, s_a_fake = self.gen_a.encode(x_a)
        c_b, s_b_fake = self.gen_b.encode(x_b)
        x_ba = self.gen_a.decode(c_b, s_a)
        x_ab = self.gen_b.decode(c_a, s_b)
        self.train()
        return x_ab, x_ba

    def compute_vgg_loss(self, vgg, img, target):
        img_vgg = vgg_preprocess(img)
        target_vgg = vgg_preprocess(target)
        img_fea = vgg(img_vgg)
        target_fea = vgg(target_vgg)
        return torch.mean((self.instancenorm(img_fea) - self.instancenorm(target_fea)) ** 2)

    def sample(self, x_a, x_b):
        self.eval()
        s_a1 = self.s_a
        s_b1 = self.s_b
        s_a2 = torch.randn(x_a.size(0), self.style_dim, 1, 1).cuda()
        s_b2 = torch.randn(x_b.size(0), self.style_dim, 1, 1).cuda()
        x_a_recon, x_b_recon, x_ba1, x_ba2, x_ab1, x_ab2 = [], [], [], [], [], []
        for i in range(x_a.size(0)):
            c_a, s_a_fake = self.gen_a.encode(x_a[i].unsqueeze(0))
            c_b, s_b_fake = self.gen_b.encode(x_b[i].unsqueeze(0))
            x_a_recon.append(self.gen_a.decode(c_a, s_a_fake))
            x_b_recon.append(self.gen_b.decode(c_b, s_b_fake))
            x_ba1.append(self.gen_a.decode(c_b, s_a1[i].unsqueeze(0)))
            x_ba2.append(self.gen_a.decode(c_b, s_a_fake))
            x_ab1.append(self.gen_b.decode(c_a, s_b1[i].unsqueeze(0)))
            x_ab2.append(self.gen_b.decode(c_a, s_b_fake))
        x_a_recon, x_b_recon = torch.cat(x_a_recon), torch.cat(x_b_recon)
        x_ba1, x_ba2 = torch.cat(x_ba1), torch.cat(x_ba2)
        x_ab1, x_ab2 = torch.cat(x_ab1), torch.cat(x_ab2)
        self.train()
        return x_a, x_a_recon, x_ab1, x_ab2, x_b, x_b_recon, x_ba1, x_ba2

    def update_learning_rate(self):
        if self.dis_scheduler is not None:
            self.dis_scheduler.step()
        if self.gen_scheduler is not None:
            self.gen_scheduler.step()

    def resume(self, checkpoint_dir, hyperparameters):
        # Load generators
        strct = False
        #last_model_name = get_model_list(checkpoint_dir, "gen")
        gen_model_name = checkpoint_dir.replace('dis', 'gen')
        state_dict = torch.load(gen_model_name)
        self.gen_a.load_state_dict(state_dict['a'], strict=strct)
        self.gen_b.load_state_dict(state_dict['b'], strict=strct)
        iterations = int(gen_model_name[-11:-3])
        # Load discriminators
        #last_model_name = get_model_list(checkpoint_dir, "dis")
        dis_model_name = checkpoint_dir.replace('gen', 'dis')
        state_dict = torch.load(dis_model_name)
        self.dis_a.load_state_dict(state_dict['a'], strict=strct)
        self.dis_b.load_state_dict(state_dict['b'], strict=strct)
        # Load optimizers
        optim_model_name = os.path.join(os.path.dirname(checkpoint_dir), 'optimizer.pt')
        state_dict = torch.load(optim_model_name)
        #self.dis_opt.load_state_dict(state_dict['dis'])
        #self.gen_opt.load_state_dict(state_dict['gen'])
        # Reinitilize schedulers
        self.dis_scheduler = get_scheduler(self.dis_opt, hyperparameters, iterations)
        self.gen_scheduler = get_scheduler(self.gen_opt, hyperparameters, iterations)
        print('Resume from iteration %d' % iterations)
        
        return iterations

    def save(self, snapshot_dir, iterations):
        # Save generators, discriminators, and optimizers
        gen_name = os.path.join(snapshot_dir, 'gen_%08d.pt' % (iterations + 1))
        dis_name = os.path.join(snapshot_dir, 'dis_%08d.pt' % (iterations + 1))
        opt_name = os.path.join(snapshot_dir, 'optimizer.pt')
        torch.save({'a': self.gen_a.state_dict(), 'b': self.gen_b.state_dict()}, gen_name)
        torch.save({'a': self.dis_a.state_dict(), 'b': self.dis_b.state_dict()}, dis_name)
        torch.save({'gen': self.gen_opt.state_dict(), 'dis': self.dis_opt.state_dict()}, opt_name)
    
    def gen_dense_update(self, x_a, x_b, hyperparameters, glb=False, noise=False, mix=False, recon=False, adv=True, perc=True, random=True):
        self.gen_opt.zero_grad()
        # encode
        c_a, s_a_prime = self.gen_a.encode(x_a, dense=True)
        c_b, s_b_prime = self.gen_b.encode(x_b, dense=True)
        # decode (within domain)
        x_a_recon = self.gen_a.decode(c_a, s_a_prime, dense=True)
        x_b_recon = self.gen_b.decode(c_b, s_b_prime, dense=True)
        # sample random style
        s_a_r = torch.randn([x_a.size(0), self.style_dim, 1, 1], device=x_a.device)
        s_b_r = torch.randn([x_b.size(0), self.style_dim, 1, 1], device=x_b.device)
        if noise:
            n_a = torch.randn([x_a.size(0), 1, *s_a_prime.shape[2:]], device=x_a.device) * self.gen_a.noise_strength
            n_b = torch.randn([x_b.size(0), 1, *s_b_prime.shape[2:]], device=x_b.device) * self.gen_b.noise_strength
            s_a_rn = s_a_r + n_a
            s_b_rn = s_b_r + n_b
        else:
            s_a_rn = s_a_r
            s_b_rn = s_b_r
        if not random:
            s_a_rn = torch.zeros_like(s_a_rn)
            s_b_rn = torch.zeros_like(s_b_rn)
            s_a_r = torch.zeros_like(s_a_r)
            s_b_r = torch.zeros_like(s_b_r)
        # decode (cross domain)
        x_ba = self.gen_a.decode(c_b, s_a_rn, dense=True)
        x_ab = self.gen_b.decode(c_a, s_b_rn, dense=True)
        # encode again
        c_b_recon, s_a_r_recon = self.gen_a.encode(x_ba, dense=False)
        c_a_recon, s_b_r_recon = self.gen_b.encode(x_ab, dense=False)
        # decode again (if needed)
        x_aba = self.gen_a.decode(c_a_recon, s_a_prime, dense=True) if hyperparameters['recon_x_cyc_w'] > 0 else None
        x_bab = self.gen_b.decode(c_b_recon, s_b_prime, dense=True) if hyperparameters['recon_x_cyc_w'] > 0 else None

        # reconstruction loss
        self.loss_gen_recon_x_a = self.recon_criterion(x_a_recon, x_a)
        self.loss_gen_recon_x_b = self.recon_criterion(x_b_recon, x_b)
        self.loss_gen_recon_s_a = self.recon_criterion(s_a_r_recon, s_a_r)
        self.loss_gen_recon_s_b = self.recon_criterion(s_b_r_recon, s_b_r)
        self.loss_gen_recon_c_a = self.recon_criterion(c_a_recon, c_a)
        self.loss_gen_recon_c_b = self.recon_criterion(c_b_recon, c_b)
        self.loss_gen_cycrecon_x_a = self.recon_criterion(x_aba, x_a) if hyperparameters['recon_x_cyc_w'] > 0 else 0
        self.loss_gen_cycrecon_x_b = self.recon_criterion(x_bab, x_b) if hyperparameters['recon_x_cyc_w'] > 0 else 0
        # GAN loss
        self.loss_gen_adv_a = self.dis_a.calc_gen_loss(x_ba)
        self.loss_gen_adv_b = self.dis_b.calc_gen_loss(x_ab)
        # domain-invariant perceptual loss
        self.loss_gen_vgg_a = self.compute_vgg_loss(self.vgg, x_ba, x_b) if hyperparameters['vgg_w'] > 0 else 0
        self.loss_gen_vgg_b = self.compute_vgg_loss(self.vgg, x_ab, x_a) if hyperparameters['vgg_w'] > 0 else 0
        # regularization
        self.loss_reg_s_a = torch.mean(s_a_prime.mean(-1, keepdims=True).mean(-2, keepdims=True)**2)
        self.loss_reg_s_b = torch.mean(s_b_prime.mean(-1, keepdims=True).mean(-2, keepdims=True)**2)
        # total loss
        self.loss_gen_total = hyperparameters['gan_w'] * self.loss_gen_adv_a + \
                              hyperparameters['gan_w'] * self.loss_gen_adv_b + \
                              hyperparameters['recon_x_w'] * self.loss_gen_recon_x_a + \
                              hyperparameters['recon_s_w'] * self.loss_gen_recon_s_a + \
                              hyperparameters['recon_c_w'] * self.loss_gen_recon_c_a + \
                              hyperparameters['recon_x_w'] * self.loss_gen_recon_x_b + \
                              hyperparameters['recon_s_w'] * self.loss_gen_recon_s_b + \
                              hyperparameters['recon_c_w'] * self.loss_gen_recon_c_b + \
                              hyperparameters['recon_x_cyc_w'] * self.loss_gen_cycrecon_x_a + \
                              hyperparameters['recon_x_cyc_w'] * self.loss_gen_cycrecon_x_b + \
                              hyperparameters['vgg_w'] * self.loss_gen_vgg_a + \
                              hyperparameters['vgg_w'] * self.loss_gen_vgg_b + \
                              0.04 * self.loss_reg_s_a + \
                              0.04 * self.loss_reg_s_b
    
        if glb:
            s_a_glb = s_a_prime.mean(-1, keepdims=True).mean(-2, keepdims=True)
            s_b_glb = s_b_prime.mean(-1, keepdims=True).mean(-2, keepdims=True)
            x_ba_glb = self.gen_a.decode(c_b, s_a_glb.detach(), dense=True)
            x_ab_glb = self.gen_b.decode(c_a, s_b_glb.detach(), dense=True)
            
            self.loss_gen_adv_a_glb = self.dis_a.calc_gen_loss(x_ba_glb)
            self.loss_gen_adv_b_glb = self.dis_b.calc_gen_loss(x_ab_glb)
            self.loss_gen_vgg_a_glb = self.compute_vgg_loss(self.vgg, x_ba_glb, x_b) if hyperparameters['vgg_w'] > 0 else 0
            self.loss_gen_vgg_b_glb = self.compute_vgg_loss(self.vgg, x_ab_glb, x_a) if hyperparameters['vgg_w'] > 0 else 0
            
            self.loss_gen_total += hyperparameters['gan_w'] * self.loss_gen_adv_a_glb * 0.5 * int(adv) + \
                                   hyperparameters['gan_w'] * self.loss_gen_adv_b_glb * 0.5 * int(adv) + \
                                   hyperparameters['vgg_w'] * self.loss_gen_vgg_a_glb * int(perc) + \
                                   hyperparameters['vgg_w'] * self.loss_gen_vgg_b_glb * int(perc)
            
            if mix:
                s_a_glb_mix = 0.5 * s_a_glb + 0.5 * s_a_prime
                s_b_glb_mix = 0.5 * s_b_glb + 0.5 * s_b_prime
                x_aa_glb_mix = self.gen_a.decode(c_a, s_a_glb_mix.detach(), dense=True)
                x_bb_glb_mix = self.gen_b.decode(c_b, s_b_glb_mix.detach(), dense=True)
                self.loss_gen_adv_a_glb_mix = self.dis_a.calc_gen_loss(x_aa_glb_mix)
                self.loss_gen_adv_b_glb_mix = self.dis_b.calc_gen_loss(x_bb_glb_mix)
                self.loss_gen_vgg_a_glb_mix = self.compute_vgg_loss(self.vgg, x_aa_glb_mix, x_a)\
                                                        if hyperparameters['vgg_w'] > 0 else 0
                self.loss_gen_vgg_b_glb_mix = self.compute_vgg_loss(self.vgg, x_bb_glb_mix, x_b)\
                                                        if hyperparameters['vgg_w'] > 0 else 0
                self.loss_gen_total += hyperparameters['gan_w'] * self.loss_gen_adv_a_glb_mix * 0.5 * int(adv) + \
                                       hyperparameters['gan_w'] * self.loss_gen_adv_b_glb_mix * 0.5 * int(adv) + \
                                       hyperparameters['gan_w'] * self.loss_gen_vgg_a_glb_mix * int(perc) + \
                                       hyperparameters['gan_w'] * self.loss_gen_vgg_b_glb_mix * int(perc)
                
            if recon:
                _, s_a_glb_recon = self.gen_a.encode(x_ba_glb, dense=False)
                _, s_b_glb_recon = self.gen_b.encode(x_ab_glb, dense=False)
                self.loss_gen_recon_s_a_glb = self.recon_criterion(s_a_glb_recon, s_a_glb.detach())
                self.loss_gen_recon_s_b_glb = self.recon_criterion(s_b_glb_recon, s_b_glb.detach())
                self.loss_gen_total += hyperparameters['recon_s_w'] * self.loss_gen_recon_s_a_glb * 0.5 + \
                                       hyperparameters['recon_s_w'] * self.loss_gen_recon_s_b_glb * 0.5
                if mix:
                    _, s_a_glb_mix_recon = self.gen_a.encode(x_aa_glb_mix, dense=False)
                    _, s_b_glb_mix_recon = self.gen_b.encode(x_bb_glb_mix, dense=False)
                    self.loss_gen_recon_s_a_glb_mix = self.recon_criterion(s_a_glb_mix_recon, s_a_glb.detach())
                    self.loss_gen_recon_s_b_glb_mix = self.recon_criterion(s_b_glb_mix_recon, s_b_glb.detach())
                    self.loss_gen_total += hyperparameters['recon_s_w'] * self.loss_gen_recon_s_a_glb_mix * 0.5 + \
                                           hyperparameters['recon_s_w'] * self.loss_gen_recon_s_b_glb_mix * 0.5

        self.loss_gen_total.backward()
        self.gen_opt.step()
    
    def dis_dense_update(self, x_a, x_b, hyperparameters, glb=False, noise=False, mix=False, adv=True, random=True):
        self.dis_opt.zero_grad()
        # encode
        c_a, s_a_prime = self.gen_a.encode(x_a, dense=True)
        c_b, s_b_prime = self.gen_b.encode(x_b, dense=True)
        # sample random style
        s_a_r = torch.randn([x_a.size(0), self.style_dim, 1, 1], device=x_a.device)
        s_b_r = torch.randn([x_b.size(0), self.style_dim, 1, 1], device=x_b.device)
        if noise:
            n_a = torch.randn([x_a.size(0), 1, *s_a_prime.shape[2:]], device=x_a.device) * self.gen_a.noise_strength
            n_b = torch.randn([x_b.size(0), 1, *s_b_prime.shape[2:]], device=x_b.device) * self.gen_b.noise_strength
            s_a_rn = s_a_r + n_a
            s_b_rn = s_b_r + n_b
        else:
            s_a_rn = s_a_r
            s_b_rn = s_b_r
        if not random:
            s_a_rn = torch.zeros_like(s_a_rn)
            s_b_rn = torch.zeros_like(s_b_rn)
            s_a_r = torch.zeros_like(s_a_r)
            s_b_r = torch.zeros_like(s_b_r)
        # decode (cross domain)
        x_ba = self.gen_a.decode(c_b, s_a_rn, dense=True)
        x_ab = self.gen_b.decode(c_a, s_b_rn, dense=True)
        # D loss
        self.loss_dis_a = self.dis_a.calc_dis_loss(x_ba.detach(), x_a)
        self.loss_dis_b = self.dis_b.calc_dis_loss(x_ab.detach(), x_b)
        self.loss_dis_total = hyperparameters['gan_w'] * self.loss_dis_a + hyperparameters['gan_w'] * self.loss_dis_b
        
        if glb:
            s_a_glb = s_a_prime.mean(-1, keepdims=True).mean(-2, keepdims=True)
            s_b_glb = s_b_prime.mean(-1, keepdims=True).mean(-2, keepdims=True)
            x_ba_glb = self.gen_a.decode(c_b, s_a_glb.detach(), dense=True)
            x_ab_glb = self.gen_b.decode(c_a, s_b_glb.detach(), dense=True)
            self.loss_dis_a_glb = self.dis_a.calc_dis_loss(x_ba_glb.detach(), x_a)
            self.loss_dis_b_glb = self.dis_b.calc_dis_loss(x_ab_glb.detach(), x_b)
            self.loss_dis_total += hyperparameters['gan_w'] * self.loss_dis_a_glb * 0.5 * int(adv) + hyperparameters['gan_w'] * self.loss_dis_b_glb * 0.5 * int(adv)
            
            if mix:
                s_a_glb_mix = 0.5 * s_a_glb + 0.5 * s_a_prime
                s_b_glb_mix = 0.5 * s_b_glb + 0.5 * s_b_prime
                x_aa_glb_mix = self.gen_a.decode(c_a, s_a_glb_mix.detach(), dense=True)
                x_bb_glb_mix = self.gen_b.decode(c_b, s_b_glb_mix.detach(), dense=True)
                self.loss_dis_a_glb_mix = self.dis_a.calc_dis_loss(x_aa_glb_mix.detach(), x_a)
                self.loss_dis_b_glb_mix = self.dis_b.calc_dis_loss(x_bb_glb_mix.detach(), x_b)
                self.loss_dis_total += hyperparameters['gan_w'] * self.loss_dis_a_glb_mix * 0.5 * int(adv) + hyperparameters['gan_w'] * self.loss_dis_b_glb_mix * 0.5 * int(adv)
        
        self.loss_dis_total.backward()
        self.dis_opt.step()

class MUNIT_Tester(MUNIT_Trainer):
    def __init__(self, hyperparameters):
        super(MUNIT_Tester, self).__init__()
        # Initiate the networks
        self.gen_a = AdaINGen(hyperparameters['input_dim_a'], hyperparameters['gen'])  # auto-encoder for domain a
        self.gen_b = AdaINGen(hyperparameters['input_dim_b'], hyperparameters['gen'])  # auto-encoder for domain b
        self.instancenorm = nn.InstanceNorm2d(512, affine=False)
        self.style_dim = hyperparameters['gen']['style_dim']

        # fix the noise used in sampling
        display_size = int(hyperparameters['display_size'])
        self.s_a = torch.randn(display_size, self.style_dim, 1, 1).cuda()
        self.s_b = torch.randn(display_size, self.style_dim, 1, 1).cuda()

        # Load VGG model if needed
        self.vgg = load_vgg16(hyperparameters['vgg_model_path'] + '/models')
        self.vgg.eval()
        for param in self.vgg.parameters():
            param.requires_grad = False
                
        # Segmentation networks
        self.semA = segmenter.Segmenter(hyperparameters['sem'])
        self.semB = segmenter.Segmenter(hyperparameters['sem'])
        self.semC = segmenter.Segmenter(hyperparameters['sem'], real=False)
    
    def att_forward(self, x_a, x_b, ):
        _ = clip_model(utils.normalize_clip(utils.denormalize(x_a)))
        feats_a = [clip_hook(f_layer).float() for f_layer in ['layer1', 'layer3']]
        feat_a = utils.get_hyperpixel(feats_a)
        _ = clip_model(utils.normalize_clip(utils.denormalize(x_b)))
        feats_b = [clip_hook(f_layer).float() for f_layer in ['layer1', 'layer3']]
        feat_b = utils.get_hyperpixel(feats_b)
        feat_a, feat_b = utils.normalize_feature_pair(feat_a, feat_b)
        
        uot_a2b, uot_b2a = utils.unbalanced_sinkhorn(feat_a, feat_b, 0.03, marg=True)
        return uot_a2b, uot_b2a
    
    def dense_forward(self, x_a, x_b, att_ab):
        c_a, s_a_prime = self.gen_a.encode(x_a, dense=True)
        c_b, s_b_prime = self.gen_b.encode(x_b, dense=True)
        
        uot_a2b, uot_b2a = self.att_forward(x_a, x_b)
        
        s_a2b = utils.get_pseudo(uot_b2a, s_a_prime)
        s_b2a = utils.get_pseudo(uot_a2b, s_b_prime)
                
        x_a2b = self.gen_a.decode(c_a, s_b2a, dense=True)
        x_b2a = self.gen_b.decode(c_b, s_a2b, dense=True)
        return x_a2b, x_b2a

if __name__ == "__main__":
    from utils import get_config
    config = get_config('configs/synthia2cityscape_folder.yaml')
    config['sem']['n_cls'] = 19
    trainer = MUNIT_Trainer(config)
    print(trainer.gen_a.parameters())

    
