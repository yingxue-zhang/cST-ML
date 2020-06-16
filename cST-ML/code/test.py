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
import seaborn as sns
import argparse
import math
import random
from VAE import Encoder
from VAE import Decoder
from VAE import VAE
from train import subtask_sampler

parser = argparse.ArgumentParser()
parser.add_argument("--region_i", type=int, default=5, help="i of region index.")
parser.add_argument("--region_j", type=int, default=25, help="j of region index.")
parser.add_argument("--task_length", type=int, default=12, help="task length.")
parser.add_argument("--WINDOW_SIZE", type=int, default=5, help="rolling window size of each sub task.")
parser.add_argument("--num_layers", type=int, default=1, help="num layers of lstm in encoder.")
parser.add_argument("--D_out", type=int, default=128, help="out features.")
parser.add_argument("--z_dim", type=int, default=128, help="latent dimension.")
parser.add_argument('--seed', type=int, default=1234, help='Random seed.')
parser.add_argument("--task_iteration", type=int, default=1000, help="number of tasks sampled for training")
parser.add_argument('--begin_idx', type=int, default=372, help='Random seed.')

opt = parser.parse_args()
print(opt)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cuda = True if torch.cuda.is_available() else False

np.random.seed(opt.seed)
torch.manual_seed(opt.seed)
if cuda:
    torch.cuda.manual_seed(opt.seed)

city_width = 50
city_length = 50

region_width = 5
region_length = 5

#if not os.path.exists('./Meta_VAE_test'):
 #   os.mkdir('./Meta_VAE_test')

################################ Data Preprocessing ##########################################
########################################################################################
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

max_num = spd.max()
print("max num: ", max_num)

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

dataset = Data.TensorDataset(spd_test, inf_test, dmd_test, period_test)

################################ Load Model ##########################################
########################################################################################
encoder = Encoder(4, opt.D_out, opt.num_layers, opt.z_dim)
decoder = Decoder(3, opt.D_out, opt.z_dim)
vae = VAE(encoder, decoder).to(device)
vae.load_state_dict(torch.load('./Meta_VAE_train/VAE_params_' + str(opt.task_iteration) + '_' + str(opt.region_i) + '_' + str(opt.region_j) + '_' + str(opt.WINDOW_SIZE) + '_' + str(opt.D_out) + '_' + str(opt.z_dim) + '_' + str(opt.num_layers) + '.pkl', map_location='cpu'))

def task_sampler(dataset):  # return a Data.TensorDataset object
    #begin_idx = random.randint(0, len(dataset) - opt.task_length)
    #begin_idx = random.randint(0, int(len(dataset)/12) - 1) * opt.task_length
    #print("begin_idx: ", begin_idx)
    end_idx = opt.begin_idx + opt.task_length
    task = dataset[int(opt.begin_idx): int(end_idx)]
    task = Data.TensorDataset(task[0], task[1], task[2], task[3])

    return task

def loss_function(y_test, y_pred):
    MSE = F.mse_loss(y_pred, y_test)
    return MSE

def test(test_loader):
    vae.eval()
    test_loss = 0
    all_MAPE = 0
    memory = Variable(torch.zeros(opt.D_out * opt.num_layers).to(device)).float()
    for step, (sub_spd, sub_inf, sub_dmd, sub_period) in enumerate(test_loader):
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

        y_test = y[opt.WINDOW_SIZE - 1] * max_num
        ################################################
        xy_train = Variable(xy_train.to(device)).float()
        x_test = Variable(x_test.to(device)).float()
        y_test = Variable(y_test.to(device)).float()

        y_pred, mu, log_var, memory = vae(xy_train, x_test, memory)
        y_pred = y_pred.cpu().data * max_num
        y_pred = Variable(y_pred.to(device)).float()

        loss = loss_function(y_pred, y_test)
        test_loss += loss.item()
        #print("single step prediction MSE: ", loss.item())

        print("single step prediction RMSE: ", math.sqrt(loss.item()))

        MAPE = np.absolute(y_pred.cpu().data.numpy() - y_test.cpu().data.numpy())/y_test.cpu().data.numpy()
        all_MAPE = all_MAPE + MAPE
        print("single step prediction MAPE: ", MAPE)

    print("Multiple steps prediction MSE: ", test_loss/(opt.task_length-opt.WINDOW_SIZE + 1))
    print("Multiple steps prediction RMSE: ", math.sqrt(test_loss/(opt.task_length-opt.WINDOW_SIZE + 1)))
    print("Multiple steps prediction MAPE: ", all_MAPE/(opt.task_length-opt.WINDOW_SIZE + 1))

if __name__ == '__main__':

    task = task_sampler(dataset)
    Sampler = subtask_sampler(task, opt.WINDOW_SIZE, 1)
    test_loader = Data.DataLoader(dataset=task, batch_size=opt.WINDOW_SIZE, shuffle=False, sampler=Sampler)

    test(test_loader)
