"""
Python implementation of Simulation 1 for Nxt PDF.
"""
import numpy as np
import scipy.integrate as integrate
import scipy.stats as stats
import statsmodels.api as sm
import random
from scipy.interpolate import interp1d
from statsmodels.tsa.stattools import adfuller
from collections import Counter
import pandas as pd
from scipy.stats import t
import torch
from scipy.stats import shapiro
import json

sample_sizes = [1000, 3000, 5000]
num_simulations = 100


#  Generate samples using the acceptance-rejection method
def generate_noise_data(k, miu, s, alpha, num):
    samples = []
    while len(samples) < num:
        std = np.sqrt(1 / alpha)
        x = np.random.normal(0, std)
        y = x * s + miu
        if np.random.uniform(0, 1) <= (1 / ((1 + x ** 2) ** k)):
            samples.append(y)
    return np.array(samples)


# Vectorized computation of A_t, B_t, W_t
def compute_tensors(sample, miu, s0, alpha0, k0):
    sample = torch.tensor(sample, dtype=torch.float32)
    eps = (sample - miu) / s0
    alpha = alpha0
    k = k0
    eps_sq = eps ** 2
    denom = 1 + eps_sq
    A_t = (-alpha * eps - (2 * k * eps) / denom) / s0
    B_t = (-alpha - (2 * k * (1 - eps_sq)) / (denom ** 2)) / (s0 ** 2)
    W_t = -A_t ** 2 - 2 * B_t
    return A_t, B_t, W_t


# GIC
def GIC(sample, miu, s0, alpha0, k0):
    A_t, B_t, W_t = compute_tensors(sample, miu, s0, alpha0, k0)
    return torch.sum(W_t) / len(sample)


# MGICE by gradient descent
def gradient_descent(sample):
    np.random.seed(100)
    torch.manual_seed(100)
    miu = torch.tensor(sample.mean(), dtype=torch.float32, requires_grad=True)
    s0 = torch.tensor(sample.std(), dtype=torch.float32, requires_grad=True)
    alpha0 = torch.tensor(1.0, dtype=torch.float32, requires_grad=True)
    k0 = torch.tensor(2.0, dtype=torch.float32, requires_grad=True)
    optimizer = torch.optim.Adam([miu, s0, alpha0, k0], lr=0.01)
    for _ in range(1000):
        optimizer.zero_grad()
        loss = -GIC(sample, miu, s0, alpha0, k0)
        loss.backward()
        optimizer.step()
        alpha0.data = alpha0.data.clamp(min=0.01)
        s0.data = s0.data.clamp(min=0.01)
        k0.data = k0.data.clamp(min=0.01)
    final_loss = -GIC(sample, miu, s0, alpha0, k0)
    final_loss = final_loss.detach().numpy()
    return {'loss': final_loss, 'miu': miu, 's': s0, 'alpha': alpha0, 'k': k0}


# Mean and std of MGICE in replications
def value_hat_average(value_hat_list):
    average_result_dic = {}
    losses = []
    miu_values = []
    s_values = []
    alpha_values = []
    k_values = []
    estimate_combined_array = []
    for each_value_dic in value_hat_list:
        each_value_dic['loss'] = float(each_value_dic['loss'])
        each_value_dic['miu'] = float(each_value_dic['miu'])
        each_value_dic['s'] = float(each_value_dic['s'])
        each_value_dic['alpha'] = float(each_value_dic['alpha'])
        each_value_dic['k'] = float(each_value_dic['k'])
        losses.append(each_value_dic['loss'])
        miu_values.append(each_value_dic['miu'])
        s_values.append(each_value_dic['s'])
        alpha_values.append(each_value_dic['alpha'])
        k_values.append(each_value_dic['k'])

        miu = np.array(each_value_dic['miu']).flatten()
        s = np.array(each_value_dic['s']).flatten()
        alpha = np.array(each_value_dic['alpha']).flatten()
        k = np.array(each_value_dic['k']).flatten()

        estimate_combined_array.append(np.concatenate([miu, s, alpha, k]))

    mean_loss = np.mean(losses)
    std_loss = np.std(losses)

    mean_miu = np.mean(miu_values)
    std_miu = np.std(miu_values)

    mean_s = np.mean(s_values)
    std_s = np.std(s_values)

    mean_alpha = np.mean(alpha_values)
    std_alpha = np.std(alpha_values)

    mean_k = np.mean(k_values)
    std_k = np.std(k_values)

    average_result_dic['mean_loss,std'] = (mean_loss, std_loss)
    average_result_dic['mean_miu,std'] = (mean_miu, std_miu)
    average_result_dic['mean_s,std'] = (mean_s, std_s)
    average_result_dic['mean_alpha,std'] = (mean_alpha, std_alpha)
    average_result_dic['mean_k,std'] = (mean_k, std_k)

    return average_result_dic, estimate_combined_array


# Run the program
final_result_dic = {}
for i in sample_sizes:
    true_order_value_hat_list = []
    result_dic = {}
    for j in range(num_simulations):
        idx = 0 + j
        np.random.seed(idx)
        data = generate_noise_data(k=1.5, miu=0.3, s=0.5, alpha=0.5, num=i)
        np.random.seed(100 + idx)
        true_order_value_hat_list.append(gradient_descent(data))

    true_order_value_hat = value_hat_average(true_order_value_hat_list)
    result_dic['true_order_value'] = true_order_value_hat[0]
    result_dic['true_order_all_values'] = np.array(true_order_value_hat[1]).tolist()
    final_result_dic[str(i)] = result_dic

with open('./S1_Nxt_PDF_result.json', 'w') as w:
    json.dump(final_result_dic, w, indent=4)
