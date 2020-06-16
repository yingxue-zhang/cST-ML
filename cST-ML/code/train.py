import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as Data
from torch.autograd import Variable
import torchvision
from torchvision import transforms
from torchvision.utils import save_image
from torchvision import datasets
import matplotlib.pyplot as plt
import imageio
import itertools
import numpy as np
import struct
import argparse
import random
import seaborn as sns
from VAE import Encoder
from VAE import Decoder
from VAE import VAE


parser = argparse.ArgumentParser()
parser.add_argument("--region_i", type=int, default=15, help="i of region index.")
parser.add_argument("--region_j", type=int, default=20, help="j of region index.")
parser.add_argument("--task_length", type=int, default=12, help="task length.")
parser.add_argument("--WINDOW_SIZE", type=int, default=5, help="rolling window size of each sub task.")
parser.add_argument("--num_layers", type=int, default=1, help="num layers of lstm in encoder.")
parser.add_argument("--D_out", type=int, default=128, help="out features.")
parser.add_argument("--z_dim", type=int, default=128, help="latent dimension.")


parser.add_argument("--task_iteration", type=int, default=1000, help="number of tasks sampled for training")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--beta1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--beta2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")


opt = parser.parse_args()
print(opt)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cuda = True if torch.cuda.is_available() else False

city_width = 50
city_length = 50

region_width = 5
region_length = 5

#np.random.seed(opt.seed)
#torch.manual_seed(opt.seed)
#if cuda:
#    torch.cuda.manual_seed(opt.seed)

if not os.path.exists('./Meta_VAE_train'):
    os.mkdir('./Meta_VAE_train')


################################ Data Preprocessing ##########################################
########################################################################################
# if use speed, set the speeds larger than 200 to be 200, then normalize
speed = np.loadtxt(open('/Users/yingxuezhang/Desktop/meta_VAE/speed_city.csv', "rb"), delimiter=",", skiprows=0)
inflow = np.loadtxt(open('/Users/yingxuezhang/Desktop/meta_VAE/inflow_city.csv', "rb"), delimiter=",", skiprows=0)
demand = np.loadtxt(open('/Users/yingxuezhang/Desktop/meta_VAE/demand_city.csv', "rb"), delimiter=",", skiprows=0)

speed = speed.reshape(-1, city_width, city_length)
inflow = inflow.reshape(-1, city_width, city_length)
demand = demand.reshape(-1, city_width, city_length)

# select the region
spd = speed[:, opt.region_i:(opt.region_i + region_width), opt.region_j:(opt.region_j + region_length)]
inf = inflow[:, opt.region_i:(opt.region_i + region_width), opt.region_j:(opt.region_j + region_length)]
dmd = demand[:, opt.region_i:(opt.region_i + region_width), opt.region_j:(opt.region_j + region_length)]

# period
num_day = int(spd.shape[0] / 12)
period_per_day = np.arange(12)
period = np.tile(period_per_day, int(speed.shape[0] / 12))  # 1-D vector

# normalize data


def min_max_normal(x):
    max_x = x.max()
    min_x = x.min()
    x = (x - min_x) / (max_x - min_x)
    return x


# input normalization: x -> [-1,1], because the last activation func in G is tanh
spd = min_max_normal(spd)
inf = min_max_normal(inf)
dmd = min_max_normal(dmd)
period = min_max_normal(period)

spd = torch.tensor(spd)
inf = torch.tensor(inf)
dmd = torch.tensor(dmd)
period = torch.tensor(period)

# prepare train set and test set
sz = int(int(spd.size(0)/12)*0.8) * 12
spd_train, spd_test = torch.split(spd,[sz, spd.size(0) - sz], dim = 0)
inf_train, inf_test = torch.split(inf,[sz, spd.size(0) - sz], dim = 0)
dmd_train, dmd_test = torch.split(dmd,[sz, spd.size(0) - sz], dim = 0)
period_train, period_test = torch.split(period,[sz, spd.size(0) - sz], dim = 0)

dataset = Data.TensorDataset(spd_train, inf_train, dmd_train, period_train)

################################ Meta VAE ##########################################
########################################################################################
encoder = Encoder(4, opt.D_out, opt.num_layers, opt.z_dim)
decoder = Decoder(3, opt.D_out, opt.z_dim)
vae = VAE(encoder, decoder)

if torch.cuda.device_count() > 1:
    print("number of GPU: ", torch.cuda.device_count())
    vae = nn.DataParallel(vae).to(device)
if torch.cuda.device_count() == 1:
    vae = vae.to(device)

optimizer = torch.optim.Adam(vae.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2))


def loss_function(y_test, y_pred, mu, log_var):
    MSE = F.mse_loss(y_pred, y_test)
    KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    return MSE + KLD

################################ Training ##########################################
########################################################################################


def task_sampler(dataset):  # return a Data.TensorDataset object
    #begin_idx = random.randint(0, len(dataset) - opt.task_length)
    begin_idx = random.randint(0, int(len(dataset)/12) - 1) * opt.task_length
    end_idx = begin_idx + opt.task_length
    task = dataset[int(begin_idx): int(end_idx)]
    task = Data.TensorDataset(task[0], task[1], task[2], task[3])

    return task


class subtask_sampler(Data.Sampler):
    r"""Samples elements sequentially like a rolling window.

    Arguments:
        data_source (Dataset): dataset to sample from
        window_size
        step_size
    """

    def __init__(self, data_source, window_size, step_size):
        self.data_source = data_source
        self.window_size = window_size
        self.step_size = step_size

    def __iter__(self):
        n = len(self.data_source)
        all_indices = torch.arange(0, n, dtype=torch.int64).unfold(0, self.window_size, self.step_size)   # size tenser(m,  window_size)
        return iter(all_indices.contiguous().view(-1,).tolist())    # convert tenser(m, window_size) to i-d tensor and to list

    def __len__(self):
        return len(self.data_source)


def train(train_loader):
    vae.train()
    train_loss = 0
    memory = Variable(torch.zeros(opt.D_out * opt.num_layers).to(device))
    optimizer.zero_grad()
    for step, (sub_spd, sub_inf, sub_dmd, sub_period) in enumerate(train_loader):
        #optimizer.zero_grad()

        ################# prepare input of vae ###############
        # prepare period in subtask, enlarge a number into an image
        period = torch.zeros(region_width, region_length).float() + sub_period[0].float()
        for i in range(1, opt.WINDOW_SIZE):
            tmp_pod = torch.zeros(region_width, region_length).float() + sub_period[i].float()
            period = torch.cat((period, tmp_pod), dim=0)
        period = period.view(opt.WINDOW_SIZE, 1, region_width, region_length).float()

        sub_inf = sub_inf.view(opt.WINDOW_SIZE, 1, region_width, region_length).float()
        sub_dmd = sub_dmd.view(opt.WINDOW_SIZE, 1, region_width, region_length).float()

        # prepare x_train
        x_train = torch.cat((sub_inf[0: (opt.WINDOW_SIZE - 1)], sub_dmd[0: (opt.WINDOW_SIZE - 1)]), dim=1)
        x_train = torch.cat((x_train, period[0: (opt.WINDOW_SIZE - 1)]), dim=1)

        # prepare y, select the center value of an image as the label, and enlarge a number into an image
        y = [sub_spd[0, int(region_width / 2), int(region_length / 2)]]
        for i in range(1, opt.WINDOW_SIZE):
            y_tmp = sub_spd[i, int(region_width / 2), int(region_length / 2)]
            y.append(y_tmp)

        y = torch.FloatTensor(y)
        y_train = y[0: (opt.WINDOW_SIZE - 1)]

        tmp_y_train = torch.zeros(region_width, region_length) + y_train[0]
        for i in range(1, opt.WINDOW_SIZE - 1):
            tmp = torch.zeros(region_width, region_length) + y_train[i]
            tmp_y_train = torch.cat((tmp_y_train, tmp), dim=0)
        y_train = tmp_y_train.view(opt.WINDOW_SIZE - 1, 1, region_width, region_length)

        # prepare xy_train
        xy_train = torch.cat((x_train, y_train), dim=1)    # shape: (window_size-1, 4, 5, 5)

        # prepare x_test, y_test
        x_test = torch.cat((sub_inf[opt.WINDOW_SIZE - 1].view(1, 1, region_width, region_length), sub_dmd[opt.WINDOW_SIZE - 1].view(1, 1, region_width, region_length)), dim=1)
        x_test = torch.cat((x_test, period[opt.WINDOW_SIZE - 1].view(1, 1, region_width, region_length)), dim=1)

        y_test = y[opt.WINDOW_SIZE - 1]
        ################################################
        xy_train = Variable(xy_train.to(device))
        x_test = Variable(x_test.to(device))
        y_test = Variable(y_test.to(device))

        y_pred, mu, log_var, memory = vae(xy_train, x_test, memory)
        loss = loss_function(y_pred, y_test, mu, log_var)

        #loss.backward(retain_graph=True)
        #train_loss += loss.item()
        #optimizer.step()
        train_loss += loss
    train_loss.backward(retain_graph=True)
    optimizer.step()


if __name__ == '__main__':
    for task_iter in range(opt.task_iteration):

        task = task_sampler(dataset)
        Sampler = subtask_sampler(task, opt.WINDOW_SIZE, 1)
        train_loader = Data.DataLoader(dataset=task, batch_size=opt.WINDOW_SIZE, shuffle=False, sampler=Sampler)

        train(train_loader)

    torch.save(vae.state_dict(), './Meta_VAE_train/VAE_params_' + str(opt.task_iteration) + '_' + str(opt.region_i) + '_' + str(opt.region_j) + '_' + str(opt.WINDOW_SIZE) + '_' + str(opt.D_out) + '_' + str(opt.z_dim) + '_' + str(opt.num_layers) + '.pkl')   # save parameters
