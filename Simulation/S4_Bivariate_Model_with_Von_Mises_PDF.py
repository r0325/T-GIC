"""
Python implementation of Simulation 4 of the T-GIC method for bivariate model with a von Mises PDF.
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
import json

sample_sizes = [300, 500, 1000]
num_simulations = 100


def bvm_density(x1, x2, k1, k2, miu1, miu2, lam):
    return np.exp(k1 * np.cos(x1 - miu1) + k2 * np.cos(x2 - miu2) + lam * np.sin(x1 - miu1) * np.sin(x2 - miu2))


#  Generate samples using the acceptance-rejection method
def generate_bvm_samples(k1, k2, miu1, miu2, lam, num_samples):
    samples = []
    max_density = np.exp(k1 + k2 + lam)
    while len(samples) < num_samples:
        x1 = np.random.uniform(0, 2 * np.pi)
        x2 = np.random.uniform(0, 2 * np.pi)
        density = bvm_density(x1, x2, k1, k2, miu1, miu2, lam)
        if np.random.uniform(0, 1) <= density / max_density:
            samples.append((x1, x2))
    return np.array(samples)


# Computation of A_t, B_t, W_t
def A_t(sample, t, k10, k20, miu10, miu20, lam):
    x_t = sample[t]
    x_t_1 = x_t[0]
    x_t_2 = x_t[1]
    k1 = np.exp(k10)
    k2 = np.exp(k20)
    miu1 = (2 * np.pi) / (1 + np.exp(-miu10))
    miu2 = (2 * np.pi) / (1 + np.exp(-miu20))
    A1 = -k1 * np.sin(x_t_1 - miu1) + lam * np.cos(x_t_1 - miu1) * np.sin(x_t_2 - miu2)
    A2 = -k2 * np.sin(x_t_2 - miu2) + lam * np.sin(x_t_1 - miu1) * np.cos(x_t_2 - miu2)
    return A1, A2


def B_t(sample, t, k10, k20, miu10, miu20, lam):
    x_t = sample[t]
    x_t_1 = x_t[0]
    x_t_2 = x_t[1]
    k1 = np.exp(k10)
    k2 = np.exp(k20)
    miu1 = (2 * np.pi) / (1 + np.exp(-miu10))
    miu2 = (2 * np.pi) / (1 + np.exp(-miu20))
    B1 = -k1 * np.cos(x_t_1 - miu1) - lam * np.sin(x_t_1 - miu1) * np.sin(x_t_2 - miu2)
    B2 = -k2 * np.cos(x_t_2 - miu2) - lam * np.sin(x_t_1 - miu1) * np.sin(x_t_2 - miu2)
    return B1, B2


def W_t(sample, t, k10, k20, miu10, miu20, lam):
    a_t = A_t(sample, t, k10, k20, miu10, miu20, lam)
    b_t = B_t(sample, t, k10, k20, miu10, miu20, lam)
    return -(a_t[0] ** 2 + a_t[1] ** 2) - 2 * (b_t[0] + b_t[1])


# GIC
def GIC(sample, k10, k20, miu10, miu20, lam):
    m = len(sample)
    F = np.sum([W_t(sample, t, k10, k20, miu10, miu20, lam) for t in range(m)])
    return F / m


# MGICE by gradient descent
def gradient_descent(sample, order):
    para = [0.0, 0.0, 0.0, 0.0, 0.0]
    from scipy.optimize import minimize
    if order == 5:
        k10 = para[0]
        k20 = para[1]
        miu10 = para[2]
        miu20 = para[3]
        lam = para[4]

        def loss_fn(params):
            k10_ = params[0]
            k20_ = params[1]
            miu10_ = params[2]
            miu20_ = params[3]
            lam_ = params[4]
            return -GIC(sample, k10_, k20_, miu10_, miu20_, lam_)

        optimizer = minimize(loss_fn, np.r_[k10, k20, miu10, miu20, lam], method='BFGS')
        k10 = optimizer.x[0]
        k20 = optimizer.x[1]
        miu10 = optimizer.x[2]
        miu20 = optimizer.x[3]
        lam = optimizer.x[4]
    else:
        k10 = para[0]
        k20 = para[1]
        miu10 = para[2]
        miu20 = para[3]

        def loss_fn(params):
            k10_ = params[0]
            k20_ = params[1]
            miu10_ = params[2]
            miu20_ = params[3]
            lam_ = 0.0
            return -GIC(sample, k10_, k20_, miu10_, miu20_, lam_)

        optimizer = minimize(loss_fn, np.r_[k10, k20, miu10, miu20], method='BFGS')
        k10 = optimizer.x[0]
        k20 = optimizer.x[1]
        miu10 = optimizer.x[2]
        miu20 = optimizer.x[3]
        lam = 0.0
    loss = -GIC(sample, k10, k20, miu10, miu20, lam)
    k1 = np.exp(k10)
    k2 = np.exp(k20)
    miu1 = (2 * np.pi) / (1 + np.exp(-miu10))
    miu2 = (2 * np.pi) / (1 + np.exp(-miu20))
    return {'loss': loss, 'k1': k1, 'k2': k2, 'miu1': miu1, 'miu2': miu2, 'lam': lam}


# Model selection by TGIC
def estimate_order(sample, order_list):
    TGIC1_values = np.zeros(len(order_list))
    TGIC2_values = np.zeros(len(order_list))
    n_size = len(sample)
    true_order_value = {}
    for order in order_list:
        c_n2 = n_size ** (-order / n_size)
        c_n1 = np.exp(-(2 * order) / n_size)
        value_hat = gradient_descent(sample, order)
        GIC_hat = -value_hat['loss']
        TGIC1 = c_n1 * GIC_hat
        TGIC2 = c_n2 * GIC_hat
        TGIC1_values[order - 4] = TGIC1
        TGIC2_values[order - 4] = TGIC2
        if order == 5:
            true_order_value = value_hat
    return np.argmax(TGIC1_values) + 4, np.argmax(TGIC2_values) + 4, true_order_value


# Mean and std of MGICE in replications
def value_hat_average(value_hat_list):
    average_result_dic = {}
    losses = []
    k1_values = []
    k2_values = []
    miu1_values = []
    miu2_values = []
    lam_values = []
    for each_value_dic in value_hat_list:
        losses.append(each_value_dic['loss'])
        k1_values.append(each_value_dic['k1'])
        k2_values.append(each_value_dic['k2'])
        miu1_values.append(each_value_dic['miu1'])
        miu2_values.append(each_value_dic['miu2'])
        lam_values.append(each_value_dic['lam'])

    mean_loss = np.mean(losses)
    std_loss = np.std(losses)

    mean_k1 = np.mean(k1_values)
    std_k1 = np.std(k1_values)

    mean_k2 = np.mean(k2_values)
    std_k2 = np.std(k2_values)

    mean_miu1 = np.mean(miu1_values)
    std_miu1 = np.std(miu1_values)

    mean_miu2 = np.mean(miu2_values)
    std_miu2 = np.std(miu2_values)

    mean_lam = np.mean(lam_values)
    std_lam = np.std(lam_values)

    average_result_dic['mean_loss,std'] = (mean_loss, std_loss)
    average_result_dic['mean_k1,std'] = (mean_k1, std_k1)
    average_result_dic['mean_k2,std'] = (mean_k2, std_k2)
    average_result_dic['mean_miu1,std'] = (mean_miu1, std_miu1)
    average_result_dic['mean_miu2,std'] = (mean_miu2, std_miu2)
    average_result_dic['mean_lam,std'] = (mean_lam, std_lam)
    return average_result_dic


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
        np.random.seed(0 + idx)
        data = generate_bvm_samples(k1=2.0, k2=1.0, miu1=1.5, miu2=2.5, lam=3, num_samples=i)
        estimate_order_model = estimate_order(data, order_list=[4, 5])
        TGIC1_order = estimate_order_model[0]
        TGIC2_order = estimate_order_model[1]
        true_order_value_hat_list.append(estimate_order_model[2])
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
    result_dic['true_order_value'] = true_order_value_hat_result
    final_result_dic[str(i)] = result_dic

with open('./S4_Bivariate_Model_with_Von_Mises_PDF_result.json', 'w') as w:
    json.dump(final_result_dic, w, indent=4)
