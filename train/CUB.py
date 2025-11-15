import random
import torch
import torch.nn as nn
import torch.autograd as autograd
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import model
import util
import classifier as classifier_zero
import time
from center_loss import TripCenterLoss_margin
from dataset import CUBDataLoader
import numpy as np
import wandb
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import os
from PIL import Image
import cv2
from sklearn.decomposition import PCA
from transformers import ViTModel
import torchvision.transforms as transforms
from wandb import Settings
import torchvision.models.resnet as models
from os.path import join

wandb.init(project='DKGZSL', config='wandb_config/config_cub.yml', mode="offline")
opt = wandb.config
opt.lambda2 = opt.lambda1
opt.encoder_layer_sizes[0] = opt.resSize
opt.decoder_layer_sizes[-1] = opt.resSize
opt.latent_size = opt.attSize
opt.device = 'cuda' if opt.cuda else 'cpu'
print('Config file from wandb:', opt)

random.seed(opt.manualSeed)
np.random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)
torch.cuda.manual_seed_all(opt.manualSeed)
cudnn.benchmark = True

dataloader = CUBDataLoader('./', opt.device, is_balance=True)

cls_criterion = nn.NLLLoss()

netE = model. Encoder(opt)
netG = model.Generator(opt)
netD = model.Discriminator(opt)
netF = model.Layer_Attention(opt)
netV = model.VAE(opt)
netFR = model.FR(opt, opt.attSize)
net_PFR = model.Pre_FR(opt, dataloader.w2v_att, dataloader.att,
                       dataloader.seenclasses, dataloader.unseenclasses)

netGA = model.GA(opt, dataloader.w2v_att, dataloader.att, dataloader.seenclasses, dataloader.unseenclasses)

# Init Tensors
input_res = torch.FloatTensor(opt.batch_size, opt.resSize)
input_att = torch.FloatTensor(opt.batch_size, opt.attSize)
noise = torch.FloatTensor(opt.batch_size, opt.nz)
input_label = torch.LongTensor(opt.batch_size)
one = torch.tensor(1, dtype=torch.float)
mone = one * -1
beta = 0
# Cuda
if opt.cuda:
    netD.cuda()
    netF.cuda()
    netG.cuda()
    netV.cuda()
    netFR.cuda()
    input_res = input_res.cuda()
    noise, input_att = noise.cuda(), input_att.cuda()
    one = one.cuda()
    mone = mone.cuda()
    input_label = input_label.cuda()

# optimizer
optimizer = optim.Adam(netE.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
optimizerV = optim.Adam(netV.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
optimizerD = optim.Adam(netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
optimizerF = optim.Adam(
    netF.parameters(), lr=opt.feed_lr, betas=(opt.beta1, 0.999))
optimizer_PFR = optim.RMSprop(
    net_PFR.parameters(), lr=0.0001, weight_decay=0.0001, momentum=0.9)
optimizer_GA = optim.RMSprop(
    netGA.parameters(), lr=0.0001, weight_decay=0.0001, momentum=0.9)
optimizerFR = optim.Adam(netFR.parameters(), lr=opt.dec_lr, betas=(opt.beta1, 0.999))
optimizer_center = optim.Adam(center_criterion.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))


import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import cv2
from PIL import Image
import os
from transformers import ViTModel
import skimage.transform
from torchvision import transforms

def loss_fn(recon_x, x, mean, log_var):
    BCE = torch.nn.functional.binary_cross_entropy(recon_x + 1e-12, x.detach(), reduction='sum')
    BCE = BCE.sum() / x.size(0)
    KLD = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp()) / x.size(0)
    return (BCE + KLD)


def bec(recon_x, x):
    BCE = torch.nn.functional.binary_cross_entropy(recon_x + 1e-12, x.detach(), reduction='sum')
    BCE = BCE.sum() / x.size(0)
    return BCE


def sample():
    batch_feature, batch_label, batch_att = data.next_seen_batch(opt.batch_size)
    input_res.copy_(batch_feature)
    input_att.copy_(batch_att)
    input_label.copy_(util.map_label(batch_label, data.seenclasses))


def WeightedL1(pred, gt):
    wt = (pred - gt).pow(2)
    wt /= wt.sum(1).sqrt().unsqueeze(1).expand(wt.size(0), wt.size(1))
    loss = wt * (pred - gt).abs()
    return loss.sum() / loss.size(0)

def cosine_loss(x, y):
    x_norm = F.normalize(x, p=2, dim=1)
    y_norm = F.normalize(y, p=2, dim=1)
    return 1 - torch.sum(x_norm * y_norm, dim=1).mean()

def ce_loss(vsp_output, updated_prototypes, labels):
    updated_prototypes = updated_prototypes.float()
    logits = torch.mm(vsp_output, updated_prototypes.t())
    logits = logits - logits.max(dim=1, keepdim=True)[0]
    loss = nn.CrossEntropyLoss()(logits, labels)
    return loss

def generate_syn_feature(generator, classes, attribute, num, netF=None, netV=None):
    nclass = classes.size(0)
    syn_feature = torch.FloatTensor(nclass * num, opt.resSize)
    syn_label = torch.LongTensor(nclass * num)
    syn_att = torch.FloatTensor(num, opt.attSize)
    syn_noise = torch.FloatTensor(num, opt.nz)
    if opt.cuda:
        syn_att = syn_att.cuda()
        syn_noise = syn_noise.cuda()
    for i in range(nclass):
        iclass = classes[i]
        iclass_att = attribute[iclass]
        syn_att.copy_(iclass_att.repeat(num, 1))
        syn_noise.normal_(0, 1)
        syn_noisev = syn_noise
        syn_attv = syn_att
        fake = generator(syn_noisev, c=syn_attv)
        if netV is not None:
            _ = netV(syn_attv, syn_noisev)
            layer1 = netV.getLayersOutD1()
            layer2 = netV.getLayersOutD2()
            layer1, layer2 = netF(syn_attv, layer1, layer2, False)
            fake = netG(syn_noisev, c=syn_attv, layerDi=layer1, layerDj=layer2)

        output = fake
        syn_feature.narrow(0, i * num, num).copy_(output.data.cpu())
        syn_label.narrow(0, i * num, num).fill_(iclass)

    return syn_feature, syn_label

def generate_syn_feature_all(generator, classes, attribute, num, netV=None, netF=None):
    nclass = classes.size(0)
    syn_feature = torch.FloatTensor(nclass * num, opt.resSize)
    syn_label = torch.LongTensor(nclass * num)
    syn_att = torch.FloatTensor(num, opt.attSize)
    syn_noise = torch.FloatTensor(num, opt.nz)
    if opt.cuda:
        syn_att = syn_att.cuda()
        syn_noise = syn_noise.cuda()
    for i in range(nclass):
        iclass = classes[i]
        iclass_att = attribute[iclass]
        syn_att.copy_(iclass_att.repeat(num, 1))
        syn_noise.normal_(0, 1)
        syn_noisev = syn_noise
        syn_attv = syn_att
        fake = generator(syn_noisev, c=syn_attv)
        if netV is not None:
            _ = netV(syn_attv, syn_noisev)
            layer1 = netV.getLayersOutD1()
            layer2 = netV.getLayersOutD2()
            layer1, layer2 = netF(syn_attv, layer1, layer2, False)
            fake = netG(syn_noisev, c=syn_attv, layerDi=layer1, layerDj=layer2)

        output = fake
        syn_feature.narrow(0, i * num, num).copy_(output.data.cpu())
        syn_label.narrow(0, i * num, num).fill_(iclass)

    return syn_feature, syn_label

def kl_divergence_loss(visual_features, semantic_features):
    visual_log_probs = F.log_softmax(visual_features, dim=1)
    semantic_probs = F.softmax(semantic_features, dim=1)
    loss = F.kl_div(visual_log_probs, semantic_probs, reduction='batchmean')

    return loss

def calc_gradient_penalty(netD, real_data, fake_data, input_att):
    alpha = torch.rand(opt.batch_size, 1)
    alpha = alpha.expand(real_data.size())
    if opt.cuda:
        alpha = alpha.cuda()
    interpolates = alpha * real_data + ((1 - alpha) * fake_data)
    if opt.cuda:
        interpolates = interpolates.cuda()
    interpolates = Variable(interpolates, requires_grad=True)
    disc_interpolates = netD(interpolates, Variable(input_att))
    ones = torch.ones(disc_interpolates.size())
    if opt.cuda:
        ones = ones.cuda()
    gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=ones,
                              create_graph=True, retain_graph=True, only_inputs=True)[0]
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * lambda1
    return gradient_penalty

def calc_gradient_penalty_FR(netFR, real_data, fake_data):
    #print real_data.size()
    alpha = torch.rand(opt.batch_size, 1)
    alpha = alpha.expand(real_data.size())
    if opt.cuda:
        alpha = alpha.cuda()
    interpolates = alpha * real_data + ((1 - alpha) * fake_data)
    if opt.cuda:
        interpolates = interpolates.cuda()

    interpolates = Variable(interpolates, requires_grad=True)
    _,_,disc_interpolates,_ ,_, _ = netFR(interpolates)
    ones = torch.ones(disc_interpolates.size())
    if opt.cuda:
        ones = ones.cuda()
    gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=ones,
                              create_graph=True, retain_graph=True, only_inputs=True)[0]
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * opt.lambda1
    return gradient_penalty


def MI_loss(mus, sigmas, i_c, alpha=1e-8):
    kl_divergence = (0.5 * torch.sum((mus ** 2) + (sigmas ** 2)
                                     - torch.log((sigmas ** 2) + alpha) - 1, dim=1))
    MI_loss = (torch.mean(kl_divergence) - i_c)
    return MI_loss


def optimize_beta(beta, MI_loss, alpha2=1e-6):
    beta_new = max(0, beta + (alpha2 * MI_loss))
    return beta_new

def WeightedL2(x1, x2):
    return ((x1 - x2) ** 2).mean()

for i in range(0, opt.pre_iters):
    batch_label, batch_feature, batch_att = dataloader.next_batch(opt.pre_bs)

    net_PFR.train()
    optimizer_PFR.zero_grad()
    # x, _, _ = netGA(batch_feature)
    Fs = batch_feature.permute(0, 2, 1)

    Fg = Fs[:, :, 1:]  # [B, C, 1]
    out_package = net_PFR(Fg)

    in_package = out_package
    in_package['batch_label'] = batch_label

    out_package = net_PFR.compute_loss(in_package)
    loss = out_package['loss']
    loss.backward()
    optimizer_PFR.step()

    if i == opt.pre_iters - 1:
        net_PFR.eval()
        vit_model = ViTModel.from_pretrained(
            "model/vit-base-patch16-224",
            local_files_only=True,
            output_hidden_states=True
        ).to('cuda' if torch.cuda.is_available() else 'cpu')
        process_image_folder(
            input_folder=input_folder,
            output_folder=output_folder,
            vit_model=vit_model,
            netGA=netGA,
            net_PFR=net_PFR
        )
        net_PFR.train()

data = util.DATA_LOADER_refine(opt, net_PFR, netGA, dataloader)
# train generative model
lambda1 = opt.lambda1
best_gzsl_acc = 0
best_zsl_acc = 0
for epoch in range(0, opt.nepoch):
    for loop in range(0, opt.feedback_loop):
        mean_lossD = 0
        mean_lossG = 0
        for i in range(0, data.ntrain, opt.batch_size):
            ######### Discriminator training ##############
            for p in netD.parameters():
                p.requires_grad = True

            for p in netV.parameters():
                p.requires_grad = True

            for p in netF.parameters():
                p.requires_grad = False

            # for p in netFR.parameters(): #unfreeze deocder
            #     p.requires_grad = True
            # Train D1 and Decoder (and Decoder Discriminator)
            gp_sum = 0
            for iter_d in range(opt.critic_iter):
                sample()
                netD.zero_grad()
                input_resv = Variable(input_res)
                input_attv = Variable(input_att)

                noise.normal_(0, 1)
                z = Variable(noise)

                netV.zero_grad()  # train VAE
                recon_x = netV(input_attv, z)
                V_cost = 0.1 * bec(recon_x, input_resv)
                V_cost.backward()
                optimizerV.step()

                criticD_real = netD(input_resv, input_attv)
                criticD_real = opt.gammaD * criticD_real.mean()
                criticD_real.backward(mone)

                # if opt.encoded_noise:
                #     means, log_var = netE(input_resv, input_attv)
                #     std = torch.exp(0.5 * log_var)
                #     eps = torch.randn([opt.batch_size, opt.latent_size]).cpu()
                #     eps = Variable(eps.cuda())
                #     z = eps * std + means  # torch.Size([64, 312])
                # else:
                noise.normal_(0, 1)
                z = Variable(noise)

                # noise.normal_(0, 1)
                # z = Variable(noise)

                if loop == 1:  # Generator
                    _ = netV(input_attv, z)
                    layer1 = netV.getLayersOutD1()
                    layer2 = netV.getLayersOutD2()
                    layer1, layer2 = netF(input_attv, layer1, layer2)
                    fake = netG(z, c=input_attv, layerDi=layer1, layerDj=layer2)
                else:
                    fake = netG(z, c=input_attv)

                criticD_fake = netD(fake.detach(), input_attv)
                criticD_fake = opt.gammaD * criticD_fake.mean()
                criticD_fake.backward(one)
                # gradient penalty
                gradient_penalty = opt.gammaD * \
                                   calc_gradient_penalty(
                                       netD, input_res, fake.data, input_att)
                gp_sum += gradient_penalty.data
                gradient_penalty.backward()
                Wasserstein_D = criticD_real - criticD_fake
                D_cost = criticD_fake - criticD_real + gradient_penalty
                optimizerD.step()

            gp_sum /= (opt.gammaD * lambda1 * opt.critic_iter)
            if (gp_sum > 1.05).sum() > 0:
                lambda1 *= 1.1
            elif (gp_sum < 1.001).sum() > 0:
                lambda1 /= 1.1

            ############# Generator training ##############
            # Train Generator and Decoder
            for p in netD.parameters():
                p.requires_grad = False

            for p in netV.parameters():
                p.requires_grad = False

            for p in netF.parameters():
                p.requires_grad = True

            # for p in netFR.parameters(): #unfreeze deocder
            #     p.requires_grad = False

            netE.zero_grad()
            netG.zero_grad()
            input_resv = Variable(input_res)
            input_attv = Variable(input_att)
            # means, log_var = netE(input_resv, input_attv)
            # std = torch.exp(0.5 * log_var)
            # eps = torch.randn([opt.batch_size, opt.latent_size]).cpu()
            # eps = Variable(eps.cuda())
            # z = eps * std + means  # torch.Size([64, 312])
            noise.normal_(0, 1)
            z = Variable(noise)

            if loop >= 1:
                _ = netV(input_attv, z)
                layer1 = netV.getLayersOutD1()
                layer2 = netV.getLayersOutD2()
                layer1, layer2 = netF(input_attv, layer1, layer2)
                recon_x = netG(z, c=input_attv, layerDi=layer1, layerDj=layer2)
            else:
                recon_x = netG(z, c=input_attv)

            # vae_loss_seen = loss_fn(recon_x, input_resv, means, log_var)
            # errG = vae_loss_seen

            criticG_fake = netD(recon_x, input_attv).mean()
            G_cost = -criticG_fake
            errG = opt.gammaG * G_cost

            errG.backward()
            optimizerG.step()


            if loop == 1:
                optimizerF.step()

    print('[%d/%d]  Loss_D: %.2f Loss_G: %.2f, Wasserstein_dist:%.2f' % (epoch,
                                                                         opt.nepoch, D_cost.item(), G_cost.item(),
                                                                         Wasserstein_D.item()))
    netG.eval()
    netV.eval()
    netF.eval()
    syn_feature, syn_label = generate_syn_feature(netG, data.unseenclasses, data.attribute, opt.syn_num, netF=netF,
                                                  netV=netV)

    train_X = torch.cat((data.train_feature, syn_feature), 0)
    train_Y = torch.cat((data.train_label, syn_label), 0)
    nclass = opt.nclass_all
    if opt.gzsl:
        if opt.final_classifier == 'softmax':
            # Train GZSL classifier
            gzsl_cls = classifier_zero.CLASSIFIER(train_X, train_Y, data, nclass, opt.cuda, opt.classifier_lr, 0.5,
                                                  25, opt.syn_num, generalized=True,
                                                  final_classifier=opt.final_classifier,
                                                  dec_size=opt.attSize, dec_hidden_size=4096, opt=opt)

            if best_gzsl_acc <= gzsl_cls.H:
                best_gzsl_epoch = epoch
                best_acc_seen, best_acc_unseen, best_gzsl_acc = gzsl_cls.acc_seen, gzsl_cls.acc_unseen, gzsl_cls.H
            print('GZSL: seen=%.3f, unseen=%.3f, h=%.3f' % (gzsl_cls.acc_seen, gzsl_cls.acc_unseen, gzsl_cls.H),
                  end=" ")

    # Train CZSL classifier
    if opt.final_classifier == 'softmax':
        zsl = classifier_zero.CLASSIFIER(syn_feature, util.map_label(syn_label, data.unseenclasses),
                                         data, data.unseenclasses.size(
                0), opt.cuda, opt.classifier_lr, 0.5, 25, opt.syn_num,
                                         generalized=False, final_classifier=opt.final_classifier,
                                         dec_size=opt.attSize, dec_hidden_size=4096, opt=opt)
        acc = zsl.acc
        if best_zsl_acc <= acc:
            best_zsl_epoch = epoch
            best_zsl_acc = acc
        print('ZSL: unseen accuracy=%.4f' % (acc))

    if epoch % 10 == 0:
        print('GZSL: epoch=%d, best_unseen=%.3f, best_seen=%.3f, best_h=%.3f' % (
        best_gzsl_epoch, best_acc_unseen, best_acc_seen, best_gzsl_acc))
        print('ZSL: epoch=%d, best unseen accuracy=%.3f' % (best_zsl_epoch, best_zsl_acc))

    # reset G to training mode
    netG.train()
    netV.train()
    netF.train()
    # netFR.train()

print(time.strftime('ending time:%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
print('Dataset', opt.dataset)
print('the best ZSL unseen accuracy is', best_zsl_acc)
if opt.gzsl:
    print('Dataset', opt.dataset)
    print('the best GZSL unseen accuracy is', best_acc_seen)
    print('the best GZSL seen accuracy is', best_acc_unseen)
    print('the best GZSL H is', best_gzsl_acc)

