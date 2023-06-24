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
# import gdown
# from natsort import natsorted
from PIL import Image
from types import SimpleNamespace
from math import ceil
from typing import Type  

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def calc_vals(sens, y, flag, epsilon, p_vals=False):
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
    n = n_1+n_2
    
    p_1_ten = torch.gather(y,0,indices_1)
    p_1 = (p_1_ten == 1.).sum(dim=0)
    
    p_2_ten = torch.gather(y,0,indices_2)
    p_2 = (p_2_ten == 1.).sum(dim=0)
    
    if p_vals:
        return p_1, p_2

    try:
        # τ_1 = ((n_2*p_1)-(p_2*n_1)-(n_1*n_2*ϵ))/(n_1*(n_1+n_2))
        if flag:
            tau_1 = ((n_2*p_1)-(p_2*n_1)-(n_1*n_2*epsilon))/(n_1*(n_1+n_2))
        else:
            tau_1 = ((p_2*n_1)-(n_2*p_1)-(n_1*n_2*epsilon))/(n_1*(n_1+n_2))
    except: 
        tau_1 = 0
    try:
        # τ_2 = ((n_2*p_1)-(p_2*n_1)-(n_1*n_2*ϵ))/(n_2*(n_1+n_2))
        if flag: 
            tau_2 = ((n_2*p_1)-(p_2*n_1)-(n_1*n_2*epsilon))/(n_2*(n_1+n_2))
        else:
            tau_2 = ((p_2*n_1)-(n_2*p_1)-(n_1*n_2*epsilon))/(n_2*(n_1+n_2))
    except:
        tau_2 = 0
    return indices_1, indices_2, tau_1, tau_2, n_1, n_2 

def get_flag(train_sens, train_y):
    p_2 = 0
    p_1 = 0
    n_2 = 0
    for i in range(len(train_y)):
        if train_sens[i] == -1:
            n_2 += 1
        if train_y[i] == 1:
            if train_sens[i] == 1:
                p_1 += 1
            else:
                p_2 += 1
    n_1 = len(train_y) - n_2
    return (p_1/n_1 > p_2/n_2) 

def proj_z_structured(zt, sens, y, X, x_bar, flag, epsilon, delta, merit_attrs):
    indices_1, indices_2, tau_1, tau_2, n_1, n_2 = calc_vals(sens, y, flag, epsilon)
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

    ### 
    # params = {
    #     "WLSACCESSID": wlsid,
    #     "WLSSECRET": wlssecret,
    #     "LICENSEID": licsid,
    #     }
    # env = gp.Env(params=params) 
    env = gp.Env()
    ####

    m = gp.Model(env=env)
    m.Params.LogToConsole = 0
    z = m.addVars((i for i in range(n)), vtype=GRB.BINARY)
    u = m.addVars((i for i in range(n)))

    # z in Z_{tau_1,tau_2} constraints
    m.addConstr(k_1 == sum(z[i] for i in indices_1))
    m.addConstr(k_2 == sum(z[i] for i in indices_2))
    
    #total number of positives remains the same
    m.addConstr(sum((y[i]+1)/2 for i in range(n)) == sum((y[i]*(1-2*z[i])+1)/2 for i in range(n)))
    if flag: 
        #labels of the more privileged class should be flipped from +1 to -1
        m.addConstr(sum((y[i]+1)/2 for i in indices_1) >= sum((y[i]*(1-2*z[i])+1)/2 for i in indices_1))
        #labels of the less privileged class should be flipped from -1 to +1
        m.addConstr(sum((y[i]+1)/2 for i in indices_2) <= sum((y[i]*(1-2*z[i])+1)/2 for i in indices_2))
    else:
        #labels of the less privileged class should be flipped from -1 to +1
        m.addConstr(sum((y[i]+1)/2 for i in indices_1) <= sum((y[i]*(1-2*z[i])+1)/2 for i in indices_1))
        #labels of the more privileged class should be flipped from +1 to -1
        m.addConstr(sum((y[i]+1)/2 for i in indices_2) >= sum((y[i]*(1-2*z[i])+1)/2 for i in indices_2))
    
    # meritocracy constraints 
    for j in merit_attrs: 
        m.addConstr(sum(X[i,j]*(y[i]*(1-2*z[i])+1) for i in range(n)) <= (x_bar[j]+x_bar[j]*delta)*sum(y[i]*(1-2*z[i])+1 for i in range(n)))
        m.addConstr(sum(X[i,j]*(y[i]*(1-2*z[i])+1) for i in range(n)) >= (x_bar[j]-x_bar[j]*delta)*sum(y[i]*(1-2*z[i])+1 for i in range(n)))

    m.addConstrs(zt[i] - z[i] <= u[i] for i in range(n))
    m.addConstrs(zt[i] - z[i] >= -u[i] for i in range(n))

    m.setObjective(sum(u[i] for i in range(n)), GRB.MINIMIZE)

    m.optimize()

    z_curr = m.getAttr('x', z).values()
    obj = m.getAttr("ObjVal")
    z_new = torch.tensor(z_curr)
    return z_new

def proj_z_unstructured(zt, sens, y, flag, epsilon):
    indices_1, indices_2, tau_1, tau_2, n_1, n_2 = calc_vals(sens, y, flag, epsilon)
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

    ### 
    # params = {
    #     "WLSACCESSID": wlsid,
    #     "WLSSECRET": wlssecret,
    #     "LICENSEID": licsid,
    #     }
    # env = gp.Env(params=params) 
    env = gp.Env()
    ####

    m = gp.Model(env=env)
    m.Params.LogToConsole = 0
    z = m.addVars((i for i in range(n)), vtype=GRB.BINARY)
    u = m.addVars((i for i in range(n)))

    # z in Z_{tau_1,tau_2} constraints
    m.addConstr(k_1 == sum(z[i] for i in indices_1))
    m.addConstr(k_2 == sum(z[i] for i in indices_2))
    
    #total number of positives remains the same
    m.addConstr(sum((y[i]+1)/2 for i in range(n)) == sum((y[i]*(1-2*z[i])+1)/2 for i in range(n)))
    if flag: 
        #labels of the more privileged class should be flipped from +1 to -1
        m.addConstr(sum((y[i]+1)/2 for i in indices_1) >= sum((y[i]*(1-2*z[i])+1)/2 for i in indices_1))
        #labels of the less privileged class should be flipped from -1 to +1
        m.addConstr(sum((y[i]+1)/2 for i in indices_2) <= sum((y[i]*(1-2*z[i])+1)/2 for i in indices_2))
    else:
        #labels of the less privileged class should be flipped from -1 to +1
        m.addConstr(sum((y[i]+1)/2 for i in indices_1) <= sum((y[i]*(1-2*z[i])+1)/2 for i in indices_1))
        #labels of the more privileged class should be flipped from +1 to -1
        m.addConstr(sum((y[i]+1)/2 for i in indices_2) >= sum((y[i]*(1-2*z[i])+1)/2 for i in indices_2))

    m.addConstrs(zt[i] - z[i] <= u[i] for i in range(n))
    m.addConstrs(zt[i] - z[i] >= -u[i] for i in range(n))

    m.setObjective(sum(u[i] for i in range(n)), GRB.MINIMIZE)

    m.optimize()

    z_curr = m.getAttr('x', z).values()
    obj = m.getAttr("ObjVal")
    z_new = torch.tensor(z_curr)
    return z_new 

def compute_error(output, y_true):
    L = output.shape[0]  
    y_pred = torch.ones(L).to(device) 
    for i in range(L): 
        if output[i] < 0.5: 
            y_pred[i] = -1 
    return (y_pred != y_true).sum().item()

def epoch_nominal(loader, model, lr=1e-3, opt=None):
    total_loss, total_err = 0.,0. 
    for batch_idx, (X, _, y, idx) in enumerate(tqdm(loader)):
        X, y, idx = X.to(device), y.to(device), idx.to(device)
        yp = model(X)[:,0]
        loss = torch.nn.SoftMarginLoss()(yp, y)

        if opt:
            opt.zero_grad()
            # Compute gradients
            loss.backward()
            # Update parameters
            opt.step()

        total_err += compute_error(yp, y)
        total_loss += loss.item() * X.shape[0]

    return total_err / len(loader.dataset), total_loss / len(loader.dataset) 


def epoch_flipped_structured(loader, model, z, train_sens, train_y, train_X, train_x_bar, flag, merit_attrs, epsilon, delta, lr_z=1e-3, opt=None):

    total_loss, total_err = 0.,0.
    zt = z.to(torch.float64)
    zt = zt.to(device)
    zt = zt.clone().detach().requires_grad_(True)

    for batch_idx, (X, sens, y, idx) in enumerate(tqdm(loader)):
        X, sens, y, idx = X.to(device), sens.to(device), y.to(device), idx.to(device)
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
            total_err += compute_error(y_hat, y)
        else:
            loss = torch.nn.SoftMarginLoss()(yp, y)
            total_err += compute_error(yp, y)

        total_loss += loss.item() * X.shape[0]

    z_new = proj_z_structured(zt, train_sens, train_y, train_X, train_x_bar, flag, epsilon, delta, merit_attrs)

    return total_err / len(loader.dataset), total_loss / len(loader.dataset), z_new

def epoch_flipped_unstructured(loader, model, z, train_sens, train_y, epsilon=0.01, lr_z=1e-3, opt=None):

    flag = get_flag(train_sens, train_y)
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
            
            total_err += compute_error(y_hat, y)
            
        else:
            loss = torch.nn.SoftMarginLoss()(yp, y) 
            total_err += compute_error(yp, y)

        total_loss += loss.item() * X.shape[0] 

    z_new = proj_z_unstructured(zt, train_sens, train_y, flag, epsilon)

    return total_err / len(loader.dataset), total_loss / len(loader.dataset), z_new
