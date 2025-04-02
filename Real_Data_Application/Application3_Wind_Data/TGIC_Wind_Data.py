"""
Python implementation of Application 3 of the T-GIC method on Wind data using bivariate model with a von Mises PDF.
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
    for order in order_list:
        c_n2 = n_size ** (-order / n_size)
        c_n1 = np.exp(-(2 * order) / n_size)
        value_hat = gradient_descent(sample, order)
        print(value_hat)
        GIC_hat = -value_hat['loss']
        TGIC1 = c_n1 * GIC_hat
        TGIC2 = c_n2 * GIC_hat
        print('TGIC1', TGIC1)
        print('TGIC2', TGIC2)
        TGIC1_values[order - 4] = TGIC1
        TGIC2_values[order - 4] = TGIC2
    return np.argmax(TGIC1_values) + 4, np.argmax(TGIC2_values) + 4


# Data process
def process_wind_direction(file1, file2):
    direction_to_radian = {
        "N": 0,
        "NNE": np.pi / 8,
        "NE": 2 * np.pi / 8,
        "ENE": 3 * np.pi / 8,
        "E": 4 * np.pi / 8,
        "ESE": 5 * np.pi / 8,
        "SE": 6 * np.pi / 8,
        "SSE": 7 * np.pi / 8,
        "S": 8 * np.pi / 8,
        "SSW": 9 * np.pi / 8,
        "SW": 10 * np.pi / 8,
        "WSW": 11 * np.pi / 8,
        "W": 12 * np.pi / 8,
        "WNW": 13 * np.pi / 8,
        "NW": 14 * np.pi / 8,
        "NNW": 15 * np.pi / 8
    }
    df1 = pd.read_csv(file1)
    df2 = pd.read_csv(file2)
    wind_direction_00 = df1['value']
    wind_direction_12 = df2['value']
    x1 = wind_direction_00.map(direction_to_radian)
    x2 = wind_direction_12.map(direction_to_radian)
    samples = list(zip(x1, x2))
    return samples


# Run the program
if __name__ == '__main__':
    order_results = {'TGIC1': [], 'TGIC2': []}

    data = process_wind_direction(file1='./Data/wind_data_value_2023_00.csv',
                                  file2='./Data/wind_data_value_2023_12.csv')
    print(len(data))
    np.random.seed(0)
    estimate_order_model = estimate_order(data, order_list=[4, 5])
    TGIC1_order = estimate_order_model[0]
    TGIC2_order = estimate_order_model[1]
    order_results['TGIC1'].append(TGIC1_order)
    order_results['TGIC2'].append(TGIC2_order)
    print(order_results)
