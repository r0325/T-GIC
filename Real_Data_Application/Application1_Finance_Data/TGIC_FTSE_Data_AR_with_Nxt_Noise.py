"""
Python implementation of Application 1 of the T-GIC method on finance data using AR model with Nxt noise.
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


# Vectorized computation of W_t
def compute_tensors(sample, A, k0, alpha0, p, constant, s0, max_order):
    sample = torch.tensor(sample, dtype=torch.float32)
    n = len(sample)
    sample = sample - constant
    X_matrix = torch.stack([sample[max_order - m - 1: n - m - 1] for m in range(p)])
    X_t_hat = torch.matmul(A, X_matrix)
    eps_t = (sample[max_order:] - X_t_hat) / s0
    eps_t_squared = eps_t ** 2
    A_t = (-alpha0 * eps_t - (2 * k0 * eps_t) / (1 + eps_t_squared)) / s0
    B_t = (-alpha0 - (2 * k0 * (1 - eps_t_squared)) / ((1 + eps_t_squared) ** 2)) / (s0 ** 2)
    W_t = -A_t ** 2 - 2 * B_t
    return W_t


# GIC
def GIC(sample, A, k0, alpha0, p, constant, s0, max_order):
    W_t = compute_tensors(sample, A, k0, alpha0, p, constant, s0, max_order)
    return W_t.mean()


# MGICE by gradient descent
def gradient_descent(sample, order, max_order, lr=0.01, epochs=1000):
    np.random.seed(1000)
    torch.manual_seed(1000)
    sample = np.array(sample)

    p = order
    A = torch.zeros(p, requires_grad=True)
    alpha0 = torch.tensor(0.5, requires_grad=True)
    k0 = torch.tensor(3.5, requires_grad=True)
    constant = torch.tensor(sample.mean(), requires_grad=True)
    s0 = torch.tensor(sample.std(), requires_grad=True)

    optimizer = Adam([A, alpha0, k0, constant, s0], lr=lr)

    for _ in range(epochs):
        optimizer.zero_grad()
        loss = -GIC(sample, A, k0, alpha0, p, constant, s0, max_order)
        loss.backward()
        optimizer.step()
        alpha0.data = alpha0.data.clamp(min=0.001)
        s0.data = s0.data.clamp(min=0.001)
        k0.data = k0.data.clamp(min=0.001)

    A = A.detach().numpy()
    alpha = alpha0.item()
    k = k0.item()
    constant = constant.item()
    s = s0.item()
    loss = -GIC(sample, torch.tensor(A), k0, alpha0, p, constant, s0, max_order).item()

    return {'loss': loss, 'A': A, 'alpha': alpha, 'k': k, 'constant': constant, 's': s}


# Model selection by TGIC
def estimate_ar_order(sample, max_order):
    TGIC1_values = np.zeros(max_order)
    TGIC2_values = np.zeros(max_order)
    n_size = len(sample)
    for order in range(1, max_order + 1):
        c_n2 = n_size ** (-order / n_size)
        c_n1 = np.exp(-(2 * order) / n_size)
        value_hat = gradient_descent(sample, order, max_order)
        print(value_hat)
        GIC_hat = -value_hat['loss']
        TGIC1 = c_n1 * GIC_hat
        TGIC2 = c_n2 * GIC_hat
        print('TGIC1', TGIC1)
        print('TGIC2', TGIC2)
        TGIC1_values[order - 1] = TGIC1
        TGIC2_values[order - 1] = TGIC2

    return np.argmax(TGIC1_values) + 1, np.argmax(TGIC2_values) + 1


# Calculate logged returns
def calculate_log_returns(file_path):
    df = pd.read_csv(file_path)
    df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
    df = df.dropna(subset=['Close'])
    df['Close'] = df['Close'][::-1].reset_index(drop=True)
    df = df.reset_index(drop=True)
    df['log_returns'] = np.log(df['Close'] / df['Close'].shift(1))
    data_list = df['log_returns'].dropna().tolist()
    return data_list


# Run the program
if __name__ == '__main__':
    order_results = {}

    np.random.seed(0)
    data = calculate_log_returns('./Data/FTSE100data.csv')
    print(len(data))
    np.random.seed(1000)

    estimate_ar_order_model = estimate_ar_order(data, max_order=10)
    TGIC1_order = estimate_ar_order_model[0]
    TGIC2_order = estimate_ar_order_model[1]

    order_results['TGIC1'] = TGIC1_order
    order_results['TGIC2'] = TGIC2_order
    print(order_results)
