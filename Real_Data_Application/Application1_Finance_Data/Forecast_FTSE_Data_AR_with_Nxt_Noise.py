"""
Python implementation of Application 1 for the rolling m-step-ahead forecast
on finance data using AR model with Nxt noise.
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
import torch
from torch.optim import Adam


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


# Rolling m-step-ahead forecast

def ar_m_step_predict(history_sample, para_estimate, select_order, m_step):
    A = para_estimate['A']
    constant = para_estimate['constant']
    sample_and_pred = history_sample
    for t in range(m_step):
        x_t_hat = 0
        for d in range(select_order):
            x_t_hat += A[d] * (sample_and_pred[- d - 1] - constant)
        sample_and_pred = np.append(sample_and_pred, x_t_hat + constant)
    return sample_and_pred


def rolling_forecast(sample, train_data_number, test_data_number, m_step, model_order):
    train_data = sample[:train_data_number]
    test_data = sample[-test_data_number:]
    predictions = []
    actuals = []
    window_size = train_data_number
    for roll_i in range(test_data_number - m_step + 1):
        start_idx = max(0, train_data_number + roll_i - window_size)
        history = np.concatenate([train_data[start_idx:], test_data[:roll_i]])
        np.random.seed(1000)
        para_value = gradient_descent(history, model_order, model_order)
        forecast = ar_m_step_predict(history, para_value, model_order, m_step)
        predictions.append(forecast[-1])
        actuals.append(test_data[roll_i + m_step - 1])

    print(len(predictions))
    print(len(actuals))
    test_mse = np.mean(np.array(np.array(predictions) - np.array(actuals)) ** 2)

    return test_mse, predictions, actuals


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
    data = calculate_log_returns('./Data/FTSE100data.csv')
    print(len(data))
    test_number = 100
    train_number = len(data) - test_number
    forecast_results = rolling_forecast(data, train_number, test_number,
                                        m_step=1, model_order=2)  # m_step=1,2,3,4,5; model_order=2,6
    print('forecast_results', forecast_results)
