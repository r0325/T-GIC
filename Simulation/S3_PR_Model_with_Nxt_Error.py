"""
Python implementation of Simulation 3 of the T-GIC method for polynomial regression model with Nxt error.
"""
import numpy as np
import scipy.integrate as integrate
import scipy.stats as stats
import statsmodels.api as sm
import random
from scipy.interpolate import interp1d
from statsmodels.tsa.stattools import adfuller
from collections import Counter
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import pandas as pd
import torch
from torch.optim import Adam
from scipy.stats import t
import json

sample_sizes = [1000, 3000, 5000]
num_simulations = 100


#  Generate noise using the acceptance-rejection method
def generate_noise_data(k, alpha, s, num):
    samples = []
    while len(samples) < num:
        std = np.sqrt(1 / alpha)
        x = np.random.normal(0, std)
        y = x * s
        if np.random.uniform(0, 1) <= (1 / ((1 + x ** 2) ** k)):
            samples.append(y)
    return np.array(samples)


#  Generate samples from polynomial regression model
def generate_pr_data(sample_num, ar_params, true_k, true_alpha, true_s, true_constant, non_gaussian=False,
                     outliers=False):
    if non_gaussian:
        noise = generate_noise_data(true_k, true_alpha, true_s, sample_num)
    else:
        noise = np.random.normal(scale=true_k, size=sample_num)

    if outliers:
        num_outliers = int(0.05 * all_num)
        outlier_indices = np.random.choice(all_num, num_outliers, replace=False)
        noise[outlier_indices] = noise[outlier_indices] * 10
    data_x = np.random.randn(sample_num)
    data0 = np.zeros(sample_num)
    for d in range(len(ar_params)):
        data0 += ar_params[d] * (data_x ** (d + 1))
    data_y = data0 + noise + true_constant
    return data_x, data_y


# Vectorized computation of W_t
def compute_tensors(sample, A, k0, alpha0, p, constant, s0):
    sample = (torch.tensor(sample[0], dtype=torch.float32), torch.tensor(sample[1], dtype=torch.float32))
    k = k0
    s = s0
    alpha = alpha0
    data_x, data_y = sample
    x_t = data_x
    y_t = data_y
    y_t_hat = sum(A[d] * (x_t ** (d + 1)) for d in range(p))
    eps_t = (y_t - y_t_hat - constant) / s
    A_t = (-alpha * eps_t - (2 * k * eps_t) / (1 + eps_t ** 2)) / s
    B_t = (-alpha - (2 * k * (1 - eps_t ** 2)) / ((1 + eps_t ** 2) ** 2)) / (s ** 2)
    W_t = -A_t ** 2 - 2 * B_t
    return W_t


# GIC
def GIC(sample, A, k0, alpha0, p, constant, s0):
    W_t = compute_tensors(sample, A, k0, alpha0, p, constant, s0)
    return W_t.mean()


# MGICE by gradient descent
def gradient_descent(sample, order, lr=0.1, epochs=1000):
    np.random.seed(1000)
    torch.manual_seed(1000)
    p = order
    A = torch.zeros(p, requires_grad=True)
    alpha0 = torch.tensor(1.0, requires_grad=True)
    k0 = torch.tensor(2.0, requires_grad=True)
    constant = torch.tensor(sample[1].mean(), requires_grad=True)
    s0 = torch.tensor(sample[1].std(), requires_grad=True)
    optimizer = Adam([A, alpha0, k0, constant, s0], lr=lr)
    for _ in range(epochs):
        optimizer.zero_grad()
        loss = -GIC(sample, A, k0, alpha0, p, constant, s0)
        loss.backward()
        optimizer.step()
        alpha0.data = alpha0.data.clamp(min=0.01)
        s0.data = s0.data.clamp(min=0.01)
        k0.data = k0.data.clamp(min=0.01)

    A = A.detach().numpy()
    alpha = alpha0.item()
    k = k0.item()
    constant = constant.item()
    s = s0.item()
    loss = -GIC(sample, torch.tensor(A), k0, alpha0, p, constant, s0).item()

    return {'loss': loss, 'A': A, 'alpha': alpha, 'k': k, 'constant': constant, 's': s}


# Model selection by TGIC
def estimate_pr_order(sample, max_order):
    TGIC1_values = np.zeros(max_order)
    TGIC2_values = np.zeros(max_order)
    n_size = len(sample[0])
    true_order_value = {}
    for order in range(1, max_order + 1):
        c_n2 = n_size ** (-order / n_size)
        c_n1 = np.exp(-(2 * order) / n_size)
        value_hat = gradient_descent(sample, order)
        GIC_hat = -value_hat['loss']
        TGIC1 = c_n1 * GIC_hat
        TGIC2 = c_n2 * GIC_hat
        TGIC1_values[order - 1] = TGIC1
        TGIC2_values[order - 1] = TGIC2
        if order == 3:
            true_order_value = value_hat
    return np.argmax(TGIC1_values) + 1, np.argmax(TGIC2_values) + 1, true_order_value


# Mean and std of MGICE in replications
def value_hat_average(value_hat_list):
    average_result_dic = {}
    losses = []
    A_values = []
    alpha_values = []
    k_values = []
    constant_values = []
    s_values = []
    estimate_combined_array = []
    for each_value_dic in value_hat_list:
        each_value_dic['loss'] = float(each_value_dic['loss'])
        each_value_dic['A'] = np.array([float(a) for a in each_value_dic['A']])
        each_value_dic['k'] = float(each_value_dic['k'])
        each_value_dic['alpha'] = float(each_value_dic['alpha'])
        each_value_dic['s'] = float(each_value_dic['s'])
        each_value_dic['constant'] = float(each_value_dic['constant'])
        losses.append(each_value_dic['loss'])
        A_values.append(each_value_dic['A'])
        alpha_values.append(each_value_dic['alpha'])
        k_values.append(each_value_dic['k'])
        constant_values.append(each_value_dic['constant'])
        s_values.append(each_value_dic['s'])
        A = np.array(each_value_dic['A']).flatten()
        alpha = np.array(each_value_dic['alpha']).flatten()
        k = np.array(each_value_dic['k']).flatten()
        constant = np.array(each_value_dic['constant']).flatten()
        s = np.array(each_value_dic['s']).flatten()
        estimate_combined_array.append(np.concatenate([A, alpha, k, constant, s]))

    A_values = np.array(A_values)

    mean_loss = np.mean(losses)
    std_loss = np.std(losses)

    mean_A = np.mean(A_values, axis=0)
    std_A = np.std(A_values, axis=0)

    mean_alpha = np.mean(alpha_values)
    std_alpha = np.std(alpha_values)

    mean_k = np.mean(k_values)
    std_k = np.std(k_values)

    mean_constant = np.mean(constant_values)
    std_constant = np.std(constant_values)

    mean_s = np.mean(s_values)
    std_s = np.std(s_values)

    average_result_dic['mean_loss,std'] = (mean_loss, std_loss)
    average_result_dic['mean_A,std'] = (list(mean_A), list(std_A))
    average_result_dic['mean_alpha,std'] = (mean_alpha, std_alpha)
    average_result_dic['mean_k,std'] = (mean_k, std_k)
    average_result_dic['mean_constant,std'] = (mean_constant, std_constant)
    average_result_dic['mean_s,std'] = (mean_s, std_s)
    return average_result_dic, estimate_combined_array


# Run the program
final_result_dic = {}
for i in sample_sizes:
    order_results = {'TGIC1': [], 'TGIC2': []}
    true_order_value_hat_list = []
    freq_result_dic1 = {}
    freq_result_dic2 = {}
    result_dic = {}

    for j in range(num_simulations):
        idx = 0 + j
        np.random.seed(idx)
        data = generate_pr_data(i, ar_params=[-1.5, 2, 5], true_k=1.5, true_alpha=0.5, true_s=0.5, true_constant=3.0,
                                non_gaussian=True)
        np.random.seed(1000 + idx)
        estimate_ar_order_model = estimate_pr_order(data, max_order=10)
        TGIC1_order = estimate_ar_order_model[0]
        TGIC2_order = estimate_ar_order_model[1]
        true_order_value_hat_list.append(estimate_ar_order_model[2])
        order_results['TGIC1'].append(TGIC1_order)
        order_results['TGIC2'].append(TGIC2_order)

    true_order_value_hat_result = value_hat_average(true_order_value_hat_list)

    TGIC_counts1 = Counter(order_results['TGIC1'])
    for GIC_order, count in sorted(TGIC_counts1.items()):
        freq_result_dic1[str(GIC_order)] = count
    TGIC_counts2 = Counter(order_results['TGIC2'])
    for GIC_order, count in sorted(TGIC_counts2.items()):
        freq_result_dic2[str(GIC_order)] = count
    result_dic['freq_result_dic1'] = freq_result_dic1
    result_dic['freq_result_dic2'] = freq_result_dic2
    result_dic['true_order_value'] = true_order_value_hat_result[0]
    result_dic['true_order_all_values'] = np.array(true_order_value_hat_result[1]).tolist()
    final_result_dic[str(i)] = result_dic

with open('./S3_PR_Model_with_Nxt_Error_result.json', 'w') as w:
    json.dump(final_result_dic, w, indent=4)
