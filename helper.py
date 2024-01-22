import os
import numpy as np
import torch
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math
import numpy as np
import torch.optim as optim
from tqdm import tqdm  
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader, random_split
import torch
from torch.autograd import Variable
import torch.utils.data as data_utils
import gurobipy as gp
from gurobipy import GRB
from torchvision.utils import save_image
import zipfile 
from PIL import Image
from types import SimpleNamespace
from math import ceil
from typing import Type  
from sklearn import preprocessing
from scipy.stats import wasserstein_distance 

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 

class all_data(Dataset):
    def __init__(self, sens_attr='race', train=True, dataset='lsac'):
        self.sens_attribute = sens_attr
        self.dataset = dataset
        if train:
            x_train = pd.read_csv('Data/'+dataset+'/X_train.csv')
            y_train = pd.read_csv('Data/'+dataset+'/y_train.csv')
            sens_train = pd.read_csv('Data/'+dataset+'/'+sens_attr+'_train.csv')

            self.x_train = x_train
            self.X = torch.tensor(x_train.values.astype(np.float32))
            self.y = torch.squeeze(torch.tensor(y_train.values.astype(np.float64)))
            self.sens = torch.squeeze(torch.tensor(sens_train.values.astype(np.float64)))

        else:
            x_test = pd.read_csv('Data/'+dataset+'/X_test.csv')
            y_test = pd.read_csv('Data/'+dataset+'/y_test.csv')
            sens_test = pd.read_csv('Data/'+dataset+'/'+sens_attr+'_test.csv')

            self.x_test = x_test
            self.X = torch.tensor(x_test.values.astype(np.float32))
            self.y = torch.squeeze(torch.tensor(y_test.values.astype(np.float64)))
            self.sens = torch.squeeze(torch.tensor(sens_test.values.astype(np.float64)))

        self.y[self.y==0] = float(-1)
        self.sens[self.sens==0] = float(-1)

        self.dataset = dataset

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        data, target, sens = self.X[index], self.y[index], self.sens[index]
        return data, target, sens, index

    def get_values(self):
        return self.X, self.y, self.sens

    def get_flag(self, p_vals=False):
      p_1, p_2, n_2 = 0, 0, 0
      for i in range(len(self.y)):
        if self.sens[i] == -1:
          n_2 += 1
        if self.y[i] == 1:
          if self.sens[i] == 1:
            p_1 += 1
          else:
            p_2 += 1
      n_1 = len(self.y) - n_2
      if p_vals:
        return p_1, p_2
      return (p_1/n_1 > p_2/n_2)

    def get_moments(self):
        p_1, p_2 = self.get_flag(p_vals=True)
        n, m = self.X.shape
        first_moments = [sum(self.X[i,j]*(self.y[i]+1) for i in range(n))/(2*(p_1+p_2)) for j in range(m)]
        second_moments = [sum(self.X[i,j]**2*(self.y[i]+1) for i in range(n))/(2*(p_1+p_2)) for j in range(m)]
        return [first_moments, second_moments]

    def get_columns(self, train=True):
      return list(self.x_train.columns)

    def get_df(self,train):
      if not train:
          df = self.x_test
          df['y'] = self.y
          return df
      else:
          df = self.x_train
          df['y'] = self.y
          return df

def calc_vals(sens, y, flag, epsilon):
    indices_1 = (sens==1).nonzero().squeeze()
    indices_2 = (sens==-1).nonzero().squeeze()

    try:
        n_1 = indices_1.size(dim=0)
    except:
        try:
            value = indices_1.item()
            n_1 = 1
        except:
            n_1 = 0
    try:
        n_2 = indices_2.size(dim=0)
    except:
        try:
            value = indices_2.item()
            n_2 = 1
        except:
            n_2 = 0
    n = n_1 + n_2

    p_1_ten = torch.gather(y,0,indices_1)
    p_1 = (p_1_ten == 1.).sum(dim=0)

    p_2_ten = torch.gather(y,0,indices_2)
    p_2 = (p_2_ten == 1.).sum(dim=0)

    try:
        if flag:
            tau_1 = ((n_2*p_1)-(n_1*p_2)-(n_1*n_2*epsilon))/(n_1*(n_1+n_2))
        else:
            tau_1 = ((n_1*p_2)-(n_2*p_1)-(n_1*n_2*epsilon))/(n_1*(n_1+n_2))
    except:
        tau_1 = 0
    try:
        if flag:
            tau_2 = ((n_2*p_1)-(n_1*p_2)-(n_1*n_2*epsilon))/(n_2*(n_1+n_2))
        else:
            tau_2 = ((n_1*p_2)-(n_2*p_1)-(n_1*n_2*epsilon))/(n_2*(n_1+n_2))
    except:
        tau_2 = 0
    return indices_1, indices_2, tau_1, tau_2, n_1, n_2, p_1, p_2

def get_flag(sens, y):
    p_2 = 0
    p_1 = 0
    n_2 = 0
    for i, x in enumerate(y):
      if sens[i] == -1:
        n_2 += 1
      if x == 1:
        if sens[i] == 1:
          p_1 += 1
        else:
          p_2 += 1
    n_1 = len(y) - n_2
    return (p_1/n_1 > p_2/n_2)

def proj_z_str(moments, merit_atts, zt, X, y, sens, flag, epsilon, delta=1, merit=False):
    indices_1, indices_2, tau_1, tau_2, n_1, n_2, p_1, p_2 = calc_vals(sens, y, flag, epsilon)
    indices_1 = indices_1.tolist()
    indices_2 = indices_2.tolist()
    if type(indices_1) != list:
        indices_1 = [indices_1]
    if type(indices_2) != list:
        indices_2 = [indices_2]

    zt = zt.tolist()
    n = n_1 + n_2
    k_1 = math.ceil(tau_1*n_1)
    k_2 = math.ceil(tau_2*n_2)
    env = gp.Env()
    m = gp.Model(env=env)
    m.Params.LogToConsole = 0
    z = m.addVars((i for i in range(n)), vtype=GRB.BINARY)
    u = m.addVars((i for i in range(n)))
    m.addConstr(sum(z[i] for i in indices_1) == k_1)
    m.addConstr(sum(z[i] for i in indices_2) == k_2)
    m.addConstr(sum((y[i]+1)/2 for i in range(n)) == sum((y[i]*(1-2*z[i])+1)/2 for i in range(n)))
    if flag:
        m.addConstr(sum((y[i]+1)/2 for i in indices_1) >= sum((y[i]*(1-2*z[i])+1)/2 for i in indices_1))
        m.addConstr(sum((y[i]+1)/2 for i in indices_2) <= sum((y[i]*(1-2*z[i])+1)/2 for i in indices_2))
    else:
        m.addConstr(sum((y[i]+1)/2 for i in indices_1) <= sum((y[i]*(1-2*z[i])+1)/2 for i in indices_1))
        m.addConstr(sum((y[i]+1)/2 for i in indices_2) >= sum((y[i]*(1-2*z[i])+1)/2 for i in indices_2))

    first_moments, second_moments = moments[0], moments[1]
    if merit:
       for ind, j in enumerate(merit_atts):
        # first moment constraints
        mu_1 = first_moments[j]
        m.addConstr(sum(X[i,j]*(y[i]*(1-2*z[i])+1) for i in range(n)) <= (1+delta)*mu_1*sum(y[i]*(1-2*z[i])+1 for i in range(n)))
        m.addConstr(sum(X[i,j]*(y[i]*(1-2*z[i])+1) for i in range(n)) >= (1-delta)*mu_1*sum(y[i]*(1-2*z[i])+1 for i in range(n)))
        # second moment constraints
        mu_2 = second_moments[j]
        m.addConstr(sum(X[i,j]**2*(y[i]*(1-2*z[i])+1) for i in range(n)) <= (1+delta)*mu_2*sum(y[i]*(1-2*z[i])+1 for i in range(n)))
        m.addConstr(sum(X[i,j]**2*(y[i]*(1-2*z[i])+1) for i in range(n)) >= (1-delta)*mu_2*sum(y[i]*(1-2*z[i])+1 for i in range(n)))

    m.addConstrs(zt[i] - z[i] <= u[i] for i in range(n))
    m.addConstrs(zt[i] - z[i] >= -u[i] for i in range(n))

    m.setObjective(sum(u[i] for i in range(n)), GRB.MINIMIZE)

    m.optimize()

    z_curr = m.getAttr('x', z).values()
    obj = m.getAttr("ObjVal")
    z_new = [y for y in z_curr]
    z_new = torch.tensor(z_new)
    return z_new 

def compute_error(output, y_true, device):
    L = output.shape[0]
    y_pred = torch.ones(L).to(device)
    for i in range(L):
        if output[i] < 0.5:
            y_pred[i] = -1
    return (y_pred != y_true).sum().item() 


def epoch_manual_str(loader, model, z, X_train_moments, X_train, y_train, sens_train, flag, device, merit_attrs = [2,3], delta=1, lr_z=1e-3, epsilon=0.01, opt=None, merit=False):
    total_loss, total_err = 0.,0.
    zt = z.to(torch.float64)
    zt = zt.to(device)
    zt = zt.clone().detach().requires_grad_(True)
    for batch_idx, (X, y, sens, idx) in enumerate(tqdm(loader)):
        X, y, sens, idx = X.to(device), y.to(device), sens.to(device), idx.to(device)
        yp = model(X)[:,0]
        if opt:
            m = X.size(dim=1)
            n = X.size(dim=0)
            z_curr = zt[idx]
            y_hat = yp*(1-2*z_curr)
            yp = yp.detach()
            f = torch.sum((torch.log(1+torch.exp(-y*yp*(1-2*zt[idx])))))
            f.backward()
            z_grad = zt.grad
            loss = torch.nn.SoftMarginLoss()(y_hat, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            with torch.no_grad():
                zt -= lr_z*z_grad
            total_err += compute_error(y_hat, y, device)
        else:
            loss = torch.nn.SoftMarginLoss()(yp, y)
            total_err += compute_error(yp, y, device)

        total_loss += loss.item() * X.shape[0]

    z_new = proj_z_str(X_train_moments, merit_attrs, zt, X_train, y_train, sens_train, flag, epsilon, delta, merit)

    return total_err / len(loader.dataset), total_loss / len(loader.dataset), z_new 

# Function used for evaluation
def get_preds(loader, model, device):
    y_pred_main = []
    for batch_idx, (X, y, sens, idx) in enumerate(tqdm(loader)):
        X = X
        y = y
        sens = sens
        idx = idx
        X, sens, y = X.to(device), sens.to(device), y.to(device)
        yp = model(X)
        for i in range(len(y)):
            if yp[i] < 0.5:
                y_pred_main.append(-1)
            else:
                y_pred_main.append(1)
    return np.array(y_pred_main)  

# Function used for evaluation
def compute_wass_distance(merit_attr_index, X_test, y_test, y_pred):
  samples_1, samples_2 = [], []
  for i in range(X_test.shape[0]):
    if y_test[i] > 0:
      samples_1.append(X_test[i,merit_attr_index])
    if y_pred[i] > 0:
      samples_2.append(X_test[i,merit_attr_index])
  n_1 = len(samples_1)
  n_2 = len(samples_2)
  obj = wasserstein_distance(samples_1, samples_2)
  return obj 

# Function used for evaluation
def avg_values(merit_attrs, columns, X_test, y_test, y_pred):
    df_avgs = {}
    for j, val in enumerate(merit_attrs):
        df_avgs[str(columns[val])] = [round(compute_wass_distance(j, X_test, y_test, y_pred), 4)]
    return df_avgs 

# Function used for evaluation
def metrics_str(merit_attrs, columns, X_test, y_test, sens_test, y_pred):

    indices_1 = list(np.where(sens_test == 1)[0])
    indices_2 = list(np.where(sens_test == -1)[0])
    count_1 = len(indices_1)
    count_2 = len(indices_2)

    y_1_1 = sum((y_test[i] == 1) for i in indices_1)
    y_1_2 = sum((y_test[i] == 1) for i in indices_2)
    y_2_1 = sum((y_test[i] == -1) for i in indices_1)
    y_2_2 = sum((y_test[i] == -1) for i in indices_2)
    yp_1_1 = sum((y_pred[i] == 1) for i in indices_1)
    yp_1_2 = sum((y_pred[i] == 1) for i in indices_2)

    y_1_yp_1_1 = sum(((y_pred[i] == 1) and (y_test[i] == 1)) for i in indices_1)
    y_1_yp_1_2 = sum(((y_pred[i] == 1) and (y_test[i] == 1)) for i in indices_2)
    y_2_yp_1_1 = sum(((y_pred[i] == 1) and (y_test[i] == -1)) for i in indices_1)
    y_2_yp_1_2 = sum(((y_pred[i] == 1) and (y_test[i] == -1)) for i in indices_2)

    corr_1 = sum((y_pred[i] == y_test[i]) for i in indices_1)
    corr_2 = sum((y_pred[i] == y_test[i]) for i in indices_2)

    if y_1_1 != 0:
        TPR_1 = y_1_yp_1_1/y_1_1
    else:
        TPR_1 = torch.tensor(-1)

    if y_1_2 != 0:
        TPR_2 = y_1_yp_1_2/y_1_2
    else:
        TPR_2 = torch.tensor(-1)

    if y_2_1 != 0:
        FPR_1 = y_2_yp_1_1/y_2_1
    else:
        FPR_1 = torch.tensor(-1)

    if y_2_2 != 0:
        FPR_2 = y_2_yp_1_2/y_2_2
    else:
        FPR_2 = torch.tensor(-1)

    SPD = abs(yp_1_1/count_1 - yp_1_2/count_2)
    EOD = abs(TPR_2 - TPR_1)
    DEO = 0.5*( abs(y_1_yp_1_1/y_1_1 - y_1_yp_1_2/y_1_2) + abs(y_2_yp_1_1/y_2_1 - y_2_yp_1_2/y_2_2) )

    correct_overall = (corr_1+corr_2)/(count_1+count_2)

    df_fair = {"Acc_Total": [round(correct_overall.item(),4)],
               "SPD": [round(SPD.item(),4)],
               "EOD": [round(EOD.item(),4)],
               "DEO": [round(DEO.item(),4)]}

    df_merit = avg_values(merit_attrs, columns, X_test, y_test, y_pred)

    return pd.DataFrame.from_dict(df_fair), pd.DataFrame.from_dict(df_merit)


########################   Unstructured    #####################

class MyDataset(Dataset):
    def __init__(self, targ='Smiling', sens_attr='Male', dataset_type='train'):

        if dataset_type == 'train':
          self.dataset = datasets.LFWPeople("./data", split='train', download=True, transform=transforms.Compose([transforms.Resize((224,224)),transforms.ToTensor(),]))
          self.is_train = True
        elif dataset_type == 'val':
          init_dataset = datasets.LFWPeople("./data", split='test', download=True, transform=transforms.Compose([transforms.Resize((224,224)),transforms.ToTensor(),]))
          self.dataset = torch.utils.data.Subset(init_dataset, [x for x in range(1000)])
          self.is_train = False
        else:
          init_dataset = datasets.LFWPeople("./data", split='test', download=True, transform=transforms.Compose([transforms.Resize((224,224)),transforms.ToTensor(),]))
          self.dataset =  torch.utils.data.Subset(init_dataset, [x for x in range(1000,len(init_dataset))])
          self.is_train = False

        self.targ = targ
        self.sens_attr = sens_attr
        self.attr_df = pd.read_csv('./data_LFW/lfw-py/lfw_attributes.txt', sep='\t', skiprows=1)

        self.train_sens = []
        self.train_y = []
        self.missing = set()
        self.flag = False


        for i in tqdm(range(self.__len__())):
            image, sens, target, idx = self.__getitem__(i)
            if target != 0 and sens != 0:
                self.train_sens.append(sens)
                self.train_y.append(target)

        self.train_sens = torch.Tensor(self.train_sens)
        self.train_y = torch.Tensor(self.train_y)

        self.keep_indices = [i for i in range(len(self.dataset)) if i not in self.missing]
        self.flag = True

    def __getitem__(self, idx):

        if self.flag:
            old_idx = self.keep_indices[idx]
        else:
            old_idx = idx
        if self.is_train:
          image, _ = self.dataset[old_idx]
          image_path = self.dataset.data[old_idx]
        else:
          image, _ = self.dataset.dataset[old_idx]
          image_path = self.dataset.dataset.data[old_idx]
        image_name = os.path.basename(image_path)
        name = image_name[:-9].replace("_", " ")
        number = int(image_name[-8:-4])

        try:
            if self.targ == "Attractive":
                target_1 = self.attr_df.loc[(self.attr_df['person'] == name) & (self.attr_df['imagenum'] == number), "Attractive Man"].item()
                target_2 = self.attr_df.loc[(self.attr_df['person'] == name) & (self.attr_df['imagenum'] == number), "Attractive Woman"].item()
                if target_1 > 0 or target_2 > 0:
                    target = 1
                else:
                    target = -1
            elif self.targ == "Young":
                target_1 = self.attr_df.loc[(self.attr_df['person'] == name) & (self.attr_df['imagenum'] == number), "Baby"].item()
                target_2 = self.attr_df.loc[(self.attr_df['person'] == name) & (self.attr_df['imagenum'] == number), "Child"].item()
                if target_1 > 0 or target_2 > 0:
                    target = 1
                else:
                    target = -1
            else:
                target = self.attr_df.loc[(self.attr_df['person'] == name) & (self.attr_df['imagenum'] == number), self.targ].item()
                if target > 0:
                    target = 1
                else:
                    target = -1

            sens = self.attr_df.loc[(self.attr_df['person'] == name) & (self.attr_df['imagenum'] == number), self.sens_attr].item()
            if sens > 0:
                sens = 1
            else:
                sens = -1
        except:
            self.missing.add(idx)
            target = 0
            sens = 0

        return image, sens, target, idx

    def __len__(self):
        if self.flag:
            return len(self.keep_indices)
        else:
            return len(self.dataset)

    def get_values(self):
        return self.train_sens, self.train_y



def proj_z_unstr(zt, sens, y, flag, epsilon):
    indices_1, indices_2, tau_1, tau_2, n_1, n_2, p_1, p_2 = calc_vals(sens, y, flag, epsilon)
    indices_1 = indices_1.tolist()
    indices_2 = indices_2.tolist()
    if type(indices_1) != list:
        indices_1 = [indices_1]
    if type(indices_2) != list:
        indices_2 = [indices_2]

    zt = zt.tolist()
    n = n_1 + n_2
    env = gp.Env()
    m = gp.Model(env=env)
    m.Params.LogToConsole = 0
    z = m.addVars((i for i in range(n)), vtype=GRB.BINARY)
    u = m.addVars((i for i in range(n)))

    m.addConstr(sum(z[i] for i in indices_1) == math.ceil(tau_1*n_1))
    m.addConstr(sum(z[i] for i in indices_2) == math.ceil(tau_2*n_2))

    m.addConstr(sum((y[i]+1)/2 for i in range(n)) == sum((y[i]*(1-2*z[i])+1)/2 for i in range(n)))
    if flag:
        m.addConstr(sum((y[i]+1)/2 for i in indices_1) >= sum((y[i]*(1-2*z[i])+1)/2 for i in indices_1))
        m.addConstr(sum((y[i]+1)/2 for i in indices_2) <= sum((y[i]*(1-2*z[i])+1)/2 for i in indices_2))
    else:
        m.addConstr(sum((y[i]+1)/2 for i in indices_1) <= sum((y[i]*(1-2*z[i])+1)/2 for i in indices_1))
        m.addConstr(sum((y[i]+1)/2 for i in indices_2) >= sum((y[i]*(1-2*z[i])+1)/2 for i in indices_2))

    m.addConstrs(zt[i] - z[i] <= u[i] for i in range(n))
    m.addConstrs(zt[i] - z[i] >= -u[i] for i in range(n))

    m.setObjective(sum(u[i] for i in range(n)), GRB.MINIMIZE)

    m.optimize()

    z_curr = m.getAttr('x', z).values()
    obj = m.getAttr("ObjVal")
    z_new = [y for y in z_curr]
    z_new = torch.tensor(z_new)
    return z_new 

def epoch_manual_unstr(loader, model, z, sens_train, y_train, flag, device, epsilon=1e-2, lr_z=1e-3, opt=None):
    total_loss, total_err = 0.,0.
    zt = z.to(torch.float64)
    zt = zt.to(device)
    zt = zt.clone().detach().requires_grad_(True)

    for batch_idx, (X, sens, y, idx) in enumerate(tqdm(loader)):
        X, sens, y, idx = X.to(device), sens.to(device), y.to(device), idx.to(device)
        y = y.float()
        yp = model(X)[:,0]

        if opt:
            m = X.size(dim=1)
            n = X.size(dim=0)
            #get z values for current batch
            z_curr = zt[idx]
            #update predictions with z values
            y_hat = yp*(1-2*z_curr)
            #function for batch
            yp = yp.detach()
            f = torch.sum((torch.log(1+torch.exp(-y*yp*(1-2*zt[idx])))))
            #get gradients
            f.backward()
            z_grad = zt.grad
            loss = torch.nn.SoftMarginLoss()(y_hat, y)
            opt.zero_grad()
            # Compute gradients
            loss.backward()
            # Update parameters
            opt.step()
            #update z values
            with torch.no_grad():
                zt -= lr_z*z_grad
            total_err += compute_error(y_hat, y, device)
        else:
            loss = torch.nn.SoftMarginLoss()(yp, y)
            total_err += compute_error(yp, y, device)

        total_loss += loss.item() * X.shape[0]

    z_new = proj_z_unstr(zt, sens_train, y_train, flag, epsilon)

    return total_err / len(loader.dataset), total_loss / len(loader.dataset), z_new

def metrics_unstr(loader, model, flag, device, org=False):
    count_1 = 0
    count_2 = 0
    y_1_1 = 0
    y_1_2 = 0
    y_n1_1 = 0
    y_n1_2 = 0
    y_n1_1 = 0
    y_n1_2 = 0
    yp_1_1 = 0
    yp_1_2 = 0
    y_1_yp_1_1 = 0
    y_1_yp_1_2 = 0
    y_n1_yp_1_1 = 0
    y_n1_yp_1_2 = 0
    corr_1 = 0
    corr_2 = 0

    for batch_idx, (X, sens, y, idx) in enumerate(tqdm(loader)):
        X = X
        y = y
        sens = sens
        idx = idx
        X, sens, y = X.to(device), sens.to(device), y.to(device)

        indices_1, indices_2, tau_1, tau_2, n_1, n_2, p_1, p_2 = calc_vals(sens, y, flag=flag, epsilon=0.01)
        indices_1 = indices_1.tolist()
        indices_2 = indices_2.tolist()
        if type(indices_1) != list:
            indices_1 = [indices_1]
        if type(indices_2) != list:
            indices_2 = [indices_2]

        count_1 += len(indices_1)
        count_2 += len(indices_2)
        y_1_1 += sum((y[i] == 1) for i in indices_1)
        y_1_2 += sum((y[i] == 1) for i in indices_2)

        if not org:
            if model:
                torch.no_grad()
                # Compute forward pass
                yp = model(X)
                L = yp.shape[0]
                y_pred = torch.ones(L).to(device)
                for i in range(L):
                    if yp[i] < 0.5:
                            y_pred[i] = -1
            else:
                print("No Model")
                y_pred = y

            y_n1_1 += sum((y[i] == -1) for i in indices_1)
            y_n1_2 += sum((y[i] == -1) for i in indices_2)
            yp_1_1 += sum((y_pred[i] == 1) for i in indices_1)
            yp_1_2 += sum((y_pred[i] == 1) for i in indices_2)
            y_1_yp_1_1 += sum(((y_pred[i] == 1) and (y[i] == 1)) for i in indices_1)
            y_1_yp_1_2 += sum(((y_pred[i] == 1) and (y[i] == 1)) for i in indices_2)
            y_n1_yp_1_1 += sum(((y_pred[i] == 1) and (y[i] == -1)) for i in indices_1)
            y_n1_yp_1_2 += sum(((y_pred[i] == 1) and (y[i] == -1)) for i in indices_2)
            corr_1 += sum((y_pred[i] == y[i]) for i in indices_1)
            corr_2 += sum((y_pred[i] == y[i]) for i in indices_2)

    if org:
        acc_rate_1 = y_1_1/count_1
        acc_rate_2 = y_1_2/count_2
        print("acc_rate_1: ", acc_rate_1.item())
        print("acc_rate_2: ", acc_rate_2.item())
        return acc_rate_1.item(), acc_rate_2.item()


    if y_1_1 != 0:
        TPR_1 = y_1_yp_1_1/y_1_1
    else:
        TPR_1 = torch.tensor(-1)

    if y_1_2 != 0:
        TPR_2 = y_1_yp_1_2/y_1_2
    else:
        TPR_2 = torch.tensor(-1)

    FPR_1 = y_n1_yp_1_1/y_n1_1
    FPR_2 = y_n1_yp_1_2/y_n1_2

    SPD = abs(yp_1_1/count_1 - yp_1_2/count_2)
    EOD = abs(TPR_2 - TPR_1)
    DEO = 0.5*( abs(y_1_yp_1_1/y_1_1 - y_1_yp_1_2/y_1_2) + abs(y_n1_yp_1_1/y_n1_1 - y_n1_yp_1_2/y_n1_2)   )

    acc_rate_1 = yp_1_1/count_1
    acc_rate_2 = yp_1_2/count_2

    correct_overall = (corr_1+corr_2)/(count_1+count_2)

    df_fair = {"Acc_Total": [round(correct_overall.item(),4)],
               "SPD": [round(SPD.item(),4)],
               "EOD": [round(EOD.item(),4)],
               "DEO": [round(DEO.item(),4)]}

    return pd.DataFrame.from_dict(df_fair)
