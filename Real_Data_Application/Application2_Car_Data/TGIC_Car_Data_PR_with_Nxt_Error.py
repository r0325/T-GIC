"""
Python implementation of Application 2 of the T-GIC method on Car data using polynomial regression model with Nxt error.
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
def gradient_descent(sample, order, lr=0.01, epochs=1000):
    np.random.seed(1845)
    torch.manual_seed(1845)

    p = order
    A = torch.zeros(p, requires_grad=True)
    alpha0 = torch.tensor(0.25, requires_grad=True)
    k0 = torch.tensor(3.0, requires_grad=True)
    constant = torch.tensor(sample[1].mean(), requires_grad=True)
    s0 = torch.tensor(sample[1].std(), requires_grad=True)

    optimizer = Adam([A, alpha0, k0, constant, s0], lr=lr)
    for _ in range(epochs):
        optimizer.zero_grad()
        loss = -GIC(sample, A, k0, alpha0, p, constant, s0)
        loss.backward()
        optimizer.step()
        alpha0.data = alpha0.data.clamp(min=0.0001)
        s0.data = s0.data.clamp(min=0.0001)
        k0.data = k0.data.clamp(min=0.0001)

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
    for order in range(1, max_order + 1):
        c_n2 = n_size ** (-order / n_size)
        c_n1 = np.exp(-(2 * order) / n_size)
        value_hat = gradient_descent(sample, order)
        print(value_hat)
        GIC_hat = -value_hat['loss']
        TGIC1 = c_n1 * GIC_hat
        TGIC2 = c_n2 * GIC_hat
        print('TGIC1', TGIC1)
        print('TGIC2', TGIC2)
        TGIC1_values[order - 1] = TGIC1
        TGIC2_values[order - 1] = TGIC2

    return np.argmax(TGIC1_values) + 1, np.argmax(TGIC2_values) + 1


# Run the program
if __name__ == '__main__':
    order_results = {}

    df = pd.read_csv('./Data/Auto_data.csv')
    data_1 = df['horsepower']
    data_2 = df['mpg']
    data_2_log = np.log(data_2)

    mean_data_1 = data_1.mean()
    std_data_1 = data_1.std()
    data_1_standardized = (data_1 - mean_data_1) / std_data_1
    data = (data_1_standardized, data_2_log)
    print(len(data_1))

    estimate_pr_order_model = estimate_pr_order(data, max_order=10)
    TGIC1_order = estimate_pr_order_model[0]
    TGIC2_order = estimate_pr_order_model[1]
    order_results['TGIC1'] = TGIC1_order
    order_results['TGIC2'] = TGIC2_order
    print(order_results)
