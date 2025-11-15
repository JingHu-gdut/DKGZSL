import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from collections import OrderedDict
import typing as t
from einops import rearrange
import cv2

import torch_geometric as pyg
import torch_geometric.nn as pygnn
import torch_geometric.utils as pyg_utils

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

def WeightedL1(pred, gt):
    wt = (pred - gt).pow(2)
    wt /= wt.sum(1).sqrt().unsqueeze(1).expand(wt.size(0), wt.size(1))
    loss = wt * (pred - gt).abs()
    return loss.sum() / loss.size(0)

def cosine_loss(x, y):
    x_norm = F.normalize(x, p=2, dim=1)
    y_norm = F.normalize(y, p=2, dim=1)
    return 1 - torch.sum(x_norm * y_norm, dim=1).mean()

# Encoder
class Encoder(nn.Module):
    def __init__(self, opt):
        super(Encoder, self).__init__()
        layer_sizes = opt.encoder_layer_sizes
        latent_size = opt.latent_size
        layer_sizes[0] += latent_size
        self.fc1=nn.Linear(layer_sizes[0], layer_sizes[-1])
        self.fc3=nn.Linear(layer_sizes[-1], latent_size*2)
        self.lrelu = nn.LeakyReLU(0.2, True)
        self.linear_means = nn.Linear(latent_size*2, latent_size)
        self.linear_log_var = nn.Linear(latent_size*2, latent_size)
        self.apply(weights_init)

    def forward(self, x, c=None):
        if c is not None: x = torch.cat((x, c), dim=-1)
        x = self.lrelu(self.fc1(x))
        x = self.lrelu(self.fc3(x))
        means = self.linear_means(x)
        log_vars = self.linear_log_var(x)
        return means, log_vars

# Generator
class Generator(nn.Module):
    def __init__(self, opt):
        super(Generator, self).__init__()
        latent_size = opt.latent_size
        input_size = latent_size * 2
        self.fc1 = nn.Linear(input_size, opt.ndh)
        self.fc3 = nn.Linear(opt.ndh, opt.resSize)
        self.relu = nn.LeakyReLU(0.2, True)
        self.lrelu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.apply(weights_init)
        self.layerGi = None
        self.layerGj = None

    def _forward(self, z, c):
        z = torch.cat((z, c), dim=-1)
        self.layerGi = self.lrelu(self.fc1(z))
        self.layerGj = self.fc3(self.layerGi)
        x = self.sigmoid(self.layerGj)
        return x

    def forward(self, z, c, layerDi=None, layerDj=None):
        if layerDi is None:
            return self._forward(z, c)
        else:
            z = torch.cat((z, c), dim=-1)  # feature merge
            self.layerGi = self.lrelu(self.fc1(z)) + 0.1 * F.normalize(layerDi)
            self.layerGj = self.fc3(self.layerGi) + 0.1 * F.normalize(layerDj)
            x = self.sigmoid(self.layerGj)
            return x

    def getLayersOutLayerG1(self):
        return self.layerGi.detach()

    def getLayersOutLayerG2(self):
        return self.layerGj.detach()

# conditional discriminator for inductive
class Discriminator(nn.Module):
    def __init__(self, opt): 
        super(Discriminator, self).__init__()
        self.fc1 = nn.Linear(opt.resSize + opt.attSize, opt.ndh)
        self.fc2 = nn.Linear(opt.ndh, 1)
        self.lrelu = nn.LeakyReLU(0.2, True)
        self.apply(weights_init)

    def forward(self, x, att):
        h = torch.cat((x, att), 1) 
        self.hidden = self.lrelu(self.fc1(h))
        h = self.fc2(self.hidden)
        return h
        
# Layer_Attention
class Layer_Attention(nn.Module):
    def __init__(self, opt):
        super(Layer_Attention, self).__init__()
        self.fc1 = nn.Linear(opt.latent_size, opt.ndh)
        self.fc2 = nn.Linear(opt.latent_size, opt.resSize)

        self.lrelu = nn.ReLU()
        self.z1 = None
        self.z2 = None
        self.apply(weights_init)
        self.bias1 = nn.Parameter(torch.empty(
            1, opt.ndh), requires_grad=True)
        self.bias2 = nn.Parameter(torch.empty(
            1, opt.resSize), requires_grad=True)

        nn.init.normal_(self.bias1)  # init vector
        nn.init.normal_(self.bias2)  # init vector
        self.net_feat1 = nn.Sequential(
            nn.Linear(opt.ndh, opt.ndh // 2),
            nn.ReLU(),
            nn.Linear(opt.ndh // 2, opt.ndh)
        )

        self.net_feat2 = nn.Sequential(
            nn.Linear(opt.resSize, opt.ndh // 2),
            nn.ReLU(),
            nn.Linear(opt.ndh // 2, opt.resSize)
        )

    def forward(self, att, feat1=None, feat2=None, TrainG=True):
        if TrainG == True:
            z1 = self.net_feat1(feat1) + F.normalize(self.fc1(att)) * self.bias1
            z2 = self.net_feat2(feat2) + F.normalize(self.fc2(att)) * self.bias2
        else:
            z1 = self.net_feat1(feat1)
            z2 = self.net_feat2(feat2)
        return z1, z2

# Res
class GA(nn.Module):
    def __init__(self, opt, init_w2v_att, att, seenclass, unseenclass):
        super(GA, self).__init__()
        self.dim_f = opt.resSize
        self.dim_att = att.shape[1]
        self.nclass = att.shape[0]
        self.device = opt.device
        self.relu = nn.ReLU(True)
        self.seenclass = seenclass
        self.unseenclass = unseenclass
        self.pro = nn.Sequential(
            nn.Linear(self.dim_f, self.dim_f // 2),
            nn.ReLU(True),
            nn.Linear(self.dim_f // 2, self.dim_f // 2),
            nn.ReLU(True),
            nn.Linear(self.dim_f // 2, self.nclass),
        )
        # for param in self.pro.parameters():
        #     param.requires_grad = False

        self.AK = nn.Sequential(
            nn.Linear(self.dim_f, self.dim_f // 2),
            nn.ReLU(True),
            nn.Linear(self.dim_f // 2, self.dim_f // 2),
            nn.ReLU(True),
            nn.Linear(self.dim_f // 2, self.dim_att),
        )
        for param in self.AK.parameters():
            param.requires_grad = False

        self.gamma = nn.Parameter(torch.ones(1))

        self.M = nn.MultiheadAttention(
            embed_dim=self.dim_f,
            num_heads=32,
            dropout=0.6,
            batch_first=True
        )
        self.Q = nn.MultiheadAttention(
            embed_dim=self.dim_f,
            num_heads=32,
            dropout=0.6,
            batch_first=True
        )
        self.convQ = nn.Conv1d(
            in_channels=self.dim_f,
            out_channels=self.dim_f,
            kernel_size=3,
            stride=1,
            padding=1
        )
        self.convK = nn.Conv1d(
            in_channels=self.dim_f,
            out_channels=self.dim_f,
            kernel_size=3,
            stride=1,
            padding=1
        )
        self.foreground_token = nn.Parameter(nn.init.zeros_(torch.empty(self.dim_f, 1)))
        self.weight_ce = torch.eye(self.nclass).float().to(self.device)
        self.w_q = nn.Parameter(torch.randn(196, self.dim_f))
        self.mask_enhancer = nn.Conv2d(1, 1, kernel_size=3, padding=1)  # 平滑掩码边缘
        self.log_softmax_func = nn.LogSoftmax(dim=1)
        self.to(self.device)

    def remove_small_attention(self, features, threshold=0.05, min_area=30, method="cca"):
        B, C, L = features.shape
        H, W = 14, 14
        spatial_feat = features.view(B, C, H, W)
        attn_maps = spatial_feat.mean(dim=1, keepdim=True)

        filtered_masks = []
        for b in range(B):
            attn_map = attn_maps[b, 0].detach().cpu().numpy()  # 单样本热图 [14,14]
            binary_map = (attn_map > threshold).astype(np.uint8)

            if method == "cca":
                num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_map, connectivity=8)
                filtered = np.zeros_like(binary_map)
                for label in range(1, num_labels):
                    if stats[label, cv2.CC_STAT_AREA] >= min_area:
                        filtered[labels == label] = 1
            else:
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
                filtered = cv2.morphologyEx(binary_map, cv2.MORPH_OPEN, kernel)

            filtered_masks.append(filtered)

        filtered_masks = torch.tensor(
            np.array(filtered_masks),
            dtype=torch.float32,
            device=features.device
        ).unsqueeze(1)  # [B, 1, 14, 14]

        filtered_spatial = spatial_feat * filtered_masks
        filtered_features = filtered_spatial.view(B, C, L)

        return filtered_features

    def compute_attention_matrix(self, A, B):
        attention_scores = torch.matmul(A.transpose(1, 2), B)
        attention_matrix = torch.mean(F.softmax(attention_scores, dim=2), dim=1, keepdim=True)

        return attention_matrix

    def compute_aug_cross_entropy(self, Fs, batch_label):
        if len(batch_label.size()) == 1:
            batch_label = self.weight_ce[batch_label]

        S_pp = Fs[:, self.seenclass]
        Labels = batch_label[:, self.seenclass]
        assert S_pp.size(1) == len(self.seenclass)
        Prob = self.log_softmax_func(S_pp)
        loss = -torch.einsum('bk,bk->b', Prob, Labels)
        loss = torch.mean(loss)
        return loss

    def forward(self, Fs, Training=True):
        B, H, C = Fs.shape
        # shape = Fs.shape
        # Fs = Fs.reshape(shape[0], shape[1], shape[2] * shape[3])  # [B, C, H*W]
        Fs = Fs.permute(0, 2, 1)

        Fg = Fs[:, :, :1]  # [B, C, 1]
        Fs = Fs[:, :, 1:]

        # activate layer
        Fs = F.sigmoid(self.convQ(Fs))
        max_values, _ = Fs.max(dim=2)
        att = self.AK(max_values)  # [B, C]
        w_q = F.normalize(self.w_q, dim=1)
        # learn-mask
        A = torch.einsum('bfi,vf->bfv', Fg, w_q)  # [B, 768,1] * [768, 196] = [B, 1, 196]
        A = self.compute_attention_matrix(Fs, A)  # [B, 1, 196]
        A = F.normalize(A, dim=-1)
        A = self.remove_small_attention(A)  # [B, 1, 196]
        # Fk = Fs * A
        Fk = Fs * F.sigmoid(A)

        mask = torch.zeros_like(Fk)
        _, top_indices = torch.topk(Fk, k=120, dim=-1)
        mask.scatter_(2, top_indices, 1)
        Fs = Fk * mask

        max_values, _ = Fk.max(dim=2)
        lable = self.pro(max_values)  # [B, C]

        return Fs, att, lable


class Pre_FR(nn.Module):
    def __init__(self, opt, init_w2v_att, att, seenclass, unseenclass):
        super(Pre_FR, self).__init__()
        self.config = opt
        self.dim_f = opt.resSize
        self.dim_v = opt.pre_dim_v
        self.dim_att = att.shape[1]
        self.nclass = att.shape[0]
        self.hidden = self.dim_att//2
        self.init_w2v_att = init_w2v_att
        device = opt.device
        self.normalize_V = opt.pre_normalize_V
        # init parameters
        self.init_w2v_att = F.normalize(torch.tensor(init_w2v_att))
        self.V = nn.Parameter(self.init_w2v_att.clone().to(device))
        self.att = F.normalize(torch.tensor(att).to(device))
        # visual-to-semantic mapping
        self.W_1 = nn.Parameter(nn.init.normal_(torch.empty(self.dim_v, self.dim_f)).to(device))
        self.W_2 = nn.Parameter(nn.init.zeros_(torch.empty(self.dim_v, self.dim_f)).to(device))
        # for loss
        self.weight_ce = torch.eye(self.nclass).float().to(device)
        self.seenclass = seenclass
        self.unseenclass = unseenclass
        self.log_softmax_func = nn.LogSoftmax(dim=1)
        self.softmax = nn.Softmax(dim=-1)
        self.gamma = nn.Parameter(torch.ones(1))

        self.loss = 0

    def compute_aug_cross_entropy(self, in_package):
        batch_label = in_package['batch_label']

        if len(in_package['batch_label'].size()) == 1:
            batch_label = self.weight_ce[batch_label]

        S_pp = in_package['S_pp']

        Labels = batch_label

        S_pp = S_pp[:, self.seenclass]
        Labels = Labels[:, self.seenclass]
        assert S_pp.size(1) == len(self.seenclass)

        Prob = self.log_softmax_func(S_pp)

        loss = -torch.einsum('bk,bk->b', Prob, Labels)
        loss = torch.mean(loss)
        return loss

    def compute_loss(self, in_package):
        if len(in_package['batch_label'].size()) == 1:
            in_package['batch_label'] = self.weight_ce[in_package['batch_label']]

        loss = self.compute_aug_cross_entropy(in_package)
        return {'loss': loss}

    def extract_attention(self, Fs, att=None, labels=None):  # Fs [B, 196, 768]
        # shape = Fs.shape
        # Fs = Fs.reshape(shape[0], shape[1], shape[2]*shape[3])  # [B, C, H*W]

        # Fs = Fs.permute(0, 2, 1)   # [B, 768, 196]
        V_n = F.normalize(self.V) if self.normalize_V else self.V
        Fs = F.normalize(Fs, dim=1)
        A = torch.einsum('iv,vf,bfr->bir', V_n, self.W_2, Fs)

        mask = torch.zeros_like(A)
        _, top_indices = torch.topk(A, k=80, dim=-1)
        mask.scatter_(2, top_indices, 1)
        A = A * mask

        A = F.softmax(A, dim=-1)  # [B, n_attr, H*W]
        # H = torch.einsum('bir,bfr->bfr', A, Fs)
        Hs = torch.einsum('bir,bfr->bif', A, Fs)

        return {'A': A, 'Hs': Hs}  # Hs shape: [B, n_attr, C]

    def compute_attribute_embed(self, Hs):
        V_n = F.normalize(self.V) if self.normalize_V else self.V
        S_p = torch.einsum('iv,vf,bif->bi', V_n, self.W_1, Hs)  # [B, n_attr]
        S_pp = torch.einsum('ki,bi->bik', self.att, S_p)  # [B, n_attr, classes]
        S = torch.sum(S_pp, axis=1)
        return {'S_pp': S, 'A_p': None}

    def forward(self, Fs, att=None, labels=None, return_attention=False):
        package_1 = self.extract_attention(Fs, att, labels)
        Hs = package_1['Hs']
        package_2 = self.compute_attribute_embed(Hs)
        # return {'A': package_1['A'],
        #         'A_p': package_2['A_p'],
        #         'S_pp': package_2['S_pp']}
        output = {
            'A': package_1['A'],
            'A_p': package_2['A_p'],
            'S_pp': package_2['S_pp']
        }

        if return_attention:
            output['attention_weights'] = package_1['A']
        return output

class VAE(nn.Module):
    def __init__(self, opt):
        super(VAE, self).__init__()
        self.apply(weights_init)
        self.lrelu = nn.ReLU()
        self.fc2 = nn.Linear(opt.latent_size, opt.ndh)
        self.fc4 = nn.Linear(opt.ndh, opt.resSize)
        self.DecoderLayer = None
        self.output = None

    def forward(self, att=None, noise=None):
        self.DecoderLayer = self.lrelu(self.fc2(att))
        self.output = self.fc4(self.DecoderLayer)
        recon_x = F.sigmoid(self.output)

        return recon_x

    def getLayersOutD1(self):
        return self.DecoderLayer.detach()

    def getLayersOutD2(self):
        return self.output.detach()


class FR(nn.Module):
    def __init__(self, opt, attSize):
        super(FR, self).__init__()
        self.embedSz = 0
        self.hidden = None
        self.lantent = None
        self.latensize = opt.latensize
        self.attSize = opt.attSize
        self.fc1 = nn.Linear(opt.resSize, opt.ngh)
        self.fc3 = nn.Linear(opt.ngh, attSize * 2)
        # self.encoder_linear = nn.Linear(opt.resSize, opt.latensize*2)
        self.discriminator = nn.Linear(opt.attSize, 1)
        self.classifier = nn.Linear(opt.attSize, opt.nclass_seen)
        self.lrelu = nn.LeakyReLU(0.2, True)
        self.sigmoid = nn.Sigmoid()
        self.logic = nn.LogSoftmax(dim=1)
        self.apply(weights_init)

    def forward(self, feat, train_G=False):
        h = feat
        # if self.embedSz > 0:
        #   assert att is not None, 'Conditional Decoder requires attribute input'
        #    h = torch.cat((feat,att),1)
        self.hidden = self.lrelu(self.fc1(h))
        self.lantent = self.fc3(self.hidden)
        mus, stds = self.lantent[:, :self.attSize], self.lantent[:, self.attSize:]
        stds = self.sigmoid(stds)
        encoder_out = reparameter(mus, stds)
        h = encoder_out
        if not train_G:
            dis_out = self.discriminator(encoder_out)
        else:
            dis_out = self.discriminator(mus)
        pred = self.logic(self.classifier(mus))
        if self.sigmoid is not None:
            h = self.sigmoid(h)
        else:
            h = h / h.pow(2).sum(1).sqrt().unsqueeze(1).expand(h.size(0), h.size(1))
        return mus, stds, dis_out, pred, encoder_out, h

    def getLayersOutDet(self):
        # used at synthesis time and feature transformation
        return self.hidden.detach()


def reparameter(mu, sigma):
    return (torch.randn_like(mu) * sigma) + mu

