import numpy as np
import torch
import torch.nn as nn
from torchvision import models


def calc_coeff(iter_num, high=1.0, low=0.0, alpha=10.0, max_iter=10000.0):
    return np.float(2.0 * (high - low) / (1.0 + np.exp(-alpha * iter_num / max_iter)) - (high - low) + low)


def init_weights(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1 or classname.find('ConvTranspose2d') != -1:
        nn.init.kaiming_uniform_(m.weight)
        nn.init.zeros_(m.bias)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.zeros_(m.bias)
    elif classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight)
        nn.init.zeros_(m.bias)


def one_weights(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1 or classname.find('ConvTranspose2d') != -1:
        nn.init.kaiming_uniform_(m.weight)
        nn.init.zeros_(m.bias)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.zeros_(m.bias)
    elif classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=1)
        nn.init.zeros_(m.bias)


def two_weights(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1 or classname.find('ConvTranspose2d') != -1:
        nn.init.kaiming_uniform_(m.weight)
        nn.init.zeros_(m.bias)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.zeros_(m.bias)
    elif classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=2)
        nn.init.zeros_(m.bias)


def three_weights(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1 or classname.find('ConvTranspose2d') != -1:
        nn.init.kaiming_uniform_(m.weight)
        nn.init.zeros_(m.bias)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.zeros_(m.bias)
    elif classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=3)
        nn.init.zeros_(m.bias)


def four_weights(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1 or classname.find('ConvTranspose2d') != -1:
        nn.init.kaiming_uniform_(m.weight)
        nn.init.zeros_(m.bias)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.zeros_(m.bias)
    elif classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=4)
        nn.init.zeros_(m.bias)


def hun_weights(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1 or classname.find('ConvTranspose2d') != -1:
        nn.init.kaiming_uniform_(m.weight)
        nn.init.zeros_(m.bias)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.zeros_(m.bias)
    elif classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=100)
        nn.init.zeros_(m.bias)


resnet_dict = {"ResNet18": models.resnet18, "ResNet34": models.resnet34, "ResNet50": models.resnet50,
               "ResNet101": models.resnet101, "ResNet152": models.resnet152}


def grl_hook(coeff):
    def fun1(grad):
        return -coeff * grad.clone()

    return fun1


class ResNetFc(nn.Module):
    def __init__(self, in_features=64, class_num=220, heuristic_num=3, heuristic_initial=False, embedding_dim=128):
        super(ResNetFc, self).__init__()
        self.heuristic_num = heuristic_num
        self.embedding_layer = nn.Linear(in_features, embedding_dim)
        self.fc = nn.Linear(embedding_dim, class_num)
        if heuristic_initial:
            self.fc.apply(hun_weights)
        else:
            self.fc.apply(init_weights)
        self.heuristic = nn.Linear(embedding_dim, class_num)
        self.heuristic.apply(init_weights)
        self.heuristic1 = nn.Linear(embedding_dim, class_num)
        self.heuristic1.apply(one_weights)
        self.heuristic2 = nn.Linear(embedding_dim, class_num)
        self.heuristic2.apply(two_weights)
        self.heuristic3 = nn.Linear(embedding_dim, class_num)
        self.heuristic3.apply(three_weights)
        self.heuristic4 = nn.Linear(embedding_dim, class_num)
        self.heuristic4.apply(four_weights)

    def forward(self, x, heuristic=True):
        x = nn.Flatten()(x)
        x = self.embedding_layer(x)
        if self.heuristic_num == 1:
            geuristic = self.heuristic(x)
        elif self.heuristic_num == 2:
            now1 = self.heuristic(x)
            now2 = self.heuristic1(x)
            geuristic = now1 + now2
        elif self.heuristic_num == 3:
            now1 = self.heuristic(x)
            now2 = self.heuristic1(x)
            now3 = self.heuristic2(x)
            geuristic = (now1 + now2 + now3)
        elif self.heuristic_num == 4:
            now1 = self.heuristic(x)
            now2 = self.heuristic1(x)
            now3 = self.heuristic2(x)
            now4 = self.heuristic3(x)
            geuristic = (now1 + now2 + now3 + now4)
        elif self.heuristic_num == 5:
            now1 = self.heuristic(x)
            now2 = self.heuristic1(x)
            now3 = self.heuristic2(x)
            now4 = self.heuristic3(x)
            now5 = self.heuristic4(x)
            geuristic = (now1 + now2 + now3 + now4 + now5)
        y = self.fc(x)
        if heuristic:
            y = y - geuristic
        return x, y, geuristic

    def get_parameters(self):
        parameter_list = [{"params": self.embedding_layer.parameters(), "lr_mult": 5, 'decay_mult': 2},
                          {"params": self.heuristic1.parameters(), "lr_mult": 5, 'decay_mult': 2},
                          {"params": self.heuristic2.parameters(), "lr_mult": 5, 'decay_mult': 2},
                          {"params": self.heuristic3.parameters(), "lr_mult": 5, 'decay_mult': 2},
                          {"params": self.heuristic4.parameters(), "lr_mult": 5, 'decay_mult': 2},
                          {"params": self.fc.parameters(), "lr_mult": 5, 'decay_mult': 2},
                          {"params": self.heuristic.parameters(), "lr_mult": 5, 'decay_mult': 2}]

        return parameter_list


class AdversarialNetwork(nn.Module):
    def __init__(self, in_feature=220, hidden_size=256, multi=1):
        super(AdversarialNetwork, self).__init__()
        self.ad_layer1 = nn.Linear(in_feature, hidden_size)
        self.ad_layer2 = nn.Linear(hidden_size, hidden_size)
        self.ad_layer3 = nn.Linear(hidden_size, multi)
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.5)
        self.dropout2 = nn.Dropout(0.5)
        self.dropout3 = nn.Dropout(0.5)
        self.sigmoid = nn.Sigmoid()
        self.apply(init_weights)
        self.iter_num = 0

    def forward(self, x):
        if self.training:
            self.iter_num += 1
        coeff = calc_coeff(self.iter_num)
        x = x * 1.0
        x.register_hook(grl_hook(coeff))
        x = self.ad_layer1(x)
        x = self.relu1(x)
        x = self.dropout1(x)
        x = self.ad_layer2(x)
        x = self.relu2(x)
        x = self.dropout2(x)
        y = self.ad_layer3(x)
        return y

    def get_parameters(self):
        return [{"params": self.parameters(), "lr_mult": 5, 'decay_mult': 2}]


class Myloss(nn.Module):
    def __init__(self, epsilon=1e-8):
        super(Myloss, self).__init__()
        self.epsilon = epsilon
        return

    def forward(self, input_, label, weight):
        entropy = - label * torch.log(input_ + self.epsilon) - (1 - label) * torch.log(1 - input_ + self.epsilon)
        return torch.sum(entropy * weight) / 2


class GaussianNoiseLayer(nn.Module):
    def __init__(self, mean=0.0, std=0.2):
        super(GaussianNoiseLayer, self).__init__()
        self.mean = mean
        self.std = std

    def forward(self, x):
        if self.training:
            noise = x.data.new(x.size()).normal_(self.mean, self.std).cuda()
            x = x + noise
        return x


class Discriminator(nn.Module):
    def __init__(self, in_dim=300, out_dim=1, noise=True, use_batchnorm=True, use_dropout=False,
                 use_sigmoid=False, use_prelu=True, drop=0.5):
        super(Discriminator, self).__init__()
        hid_dim = in_dim // 2

        modules = list()

        if noise:
            modules.append(GaussianNoiseLayer(mean=0.0, std=0.3))
        modules.append(nn.Linear(in_dim, hid_dim))
        if use_batchnorm:
            modules.append(nn.BatchNorm1d(hid_dim))
        if use_prelu:
            modules.append(nn.PReLU())
        else:
            modules.append(nn.LeakyReLU(0.2))
        if use_dropout:
            modules.append(nn.Dropout(p=drop))

        modules.append(nn.Linear(hid_dim, hid_dim))
        if use_batchnorm:
            modules.append(nn.BatchNorm1d(hid_dim))
        if use_prelu:
            modules.append(nn.PReLU())
        else:
            modules.append(nn.LeakyReLU(0.2))
        if use_dropout:
            modules.append(nn.Dropout(p=drop))
        modules.append(nn.Linear(hid_dim, out_dim))
        if use_sigmoid:
            modules.append(nn.Sigmoid())

        self.disc = nn.Sequential(*modules)

    def forward(self, x):
        return self.disc(x)


class Generator(nn.Module):
    def __init__(self, in_dim=512, out_dim=300, noise=True, use_batchnorm=True, use_dropout=False, use_prelu=True,
                 drop=0.5):
        super(Generator, self).__init__()
        hid_dim = (in_dim + out_dim) // 2

        # main model
        modules = list()

        modules.append(nn.Linear(in_dim, hid_dim))
        if use_batchnorm:
            modules.append(nn.BatchNorm1d(hid_dim))
        if use_prelu:
            modules.append(nn.PReLU())
        else:
            modules.append(nn.LeakyReLU(0.2))
        if noise:
            modules.append(GaussianNoiseLayer(mean=0.0, std=0.2))
        if use_dropout:
            modules.append(nn.Dropout(p=drop))

        modules.append(nn.Linear(hid_dim, hid_dim))
        if use_batchnorm:
            modules.append(nn.BatchNorm1d(hid_dim))
        if use_prelu:
            modules.append(nn.PReLU())
        else:
            modules.append(nn.LeakyReLU(0.2))
        if noise:
            modules.append(GaussianNoiseLayer(mean=0.0, std=0.2))
        if use_dropout:
            modules.append(nn.Dropout(p=drop))

        modules.append(nn.Linear(hid_dim, out_dim))

        self.gen = nn.Sequential(*modules)

    def forward(self, x):
        return self.gen(x)


def Entropy(input_):
    bs = input_.size(0)
    epsilon = 1e-5
    entropy = -input_ * torch.log(input_ + epsilon)
    entropy = torch.sum(entropy, dim=1)
    return entropy


def HDA_UDA(input_list, ad_net, coeff=None, myloss=Myloss(), sketch_size=8):
    softmax_output = input_list[0]
    focals = input_list[1].reshape(-1)
    ad_out = ad_net(softmax_output)
    ad_out = nn.Sigmoid()(ad_out)
    dc_target = torch.from_numpy(
        np.array([[1]] * sketch_size + [[0]] * (softmax_output.size(0) - sketch_size))).float().cuda()

    x = softmax_output
    entropy = Entropy(x)
    entropy.register_hook(grl_hook(coeff))
    entropy = torch.exp(-entropy)
    mean_entropy = torch.mean(entropy)
    heuristic = torch.mean(torch.abs(focals))

    source_mask = torch.ones_like(entropy)
    source_mask[sketch_size:] = 0
    source_weight = entropy * source_mask
    target_mask = torch.ones_like(entropy)
    target_mask[0:sketch_size] = 0
    target_weight = entropy * target_mask
    weight = source_weight / torch.sum(source_weight).detach().item() + \
             target_weight / torch.sum(target_weight).detach().item()
    return myloss(ad_out, dc_target, weight.view(-1, 1)), mean_entropy, heuristic
