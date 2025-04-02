"""
Python implementation of Application 2 for calculating the residual skewness and excess kurtosis
on Car data using fitted polynomial regression models, with Nxt and Gaussian errors.
"""
import numpy as np
from scipy.stats import skew, kurtosis, t
from scipy.interpolate import interp1d
import pandas as pd
from sklearn.utils import resample


# Calculate the residual
def per_resid(sample, A, p, t, constant):
    data_x = sample[0]
    data_y = sample[1]
    x_t = data_x[t]
    y_t = data_y[t]
    y_t_hat = 0
    for d in range(p):
        y_t_hat += A[d] * (x_t ** (d + 1))
    eps_t = y_t - y_t_hat - constant
    return eps_t


def all_resid(sample, A, p, constant):
    resid_list = [per_resid(sample, A, p, i, constant) for i in range(len(sample[0]))]
    return np.array(resid_list)


# SE of the sample estimates for skewness and excess kurtosis by a Bootstrap procedure
def bootstrap_skew_kurt_with_ci(samples, num_experiments=1000):
    n = len(samples)
    skew_values = []
    kurt_values = []

    for _ in range(num_experiments):
        subsample = np.random.choice(samples, size=n, replace=True)
        skew_values.append(skew(subsample))
        kurt_values.append(kurtosis(subsample, fisher=True))

    skew_se = np.std(skew_values)
    kurt_se = np.std(kurt_values)

    return skew_se, kurt_se


# Sample estimate and SE
def calculate_result(sample, A, p, constant):
    res_samples = all_resid(sample, A, p, constant)
    np.random.seed(500)
    # residual skewness and excess kurtosis
    sample_skewness = skew(res_samples)
    sample_kurtosis = kurtosis(res_samples, fisher=True)
    print(f"Skewness: {sample_skewness:.2f}")
    print(f"Kurtosis (excess): {sample_kurtosis:.2f}")
    results = bootstrap_skew_kurt_with_ci(res_samples)
    print(f"Skewness SE: {results[0]:.2f}")
    print(f"Kurtosis SE: {results[1]:.2f}")


# Theoretical value and the std of fitted polynomial regression model with Nxt error
def generate_noise_data(k, alpha, s, num):
    samples = []
    while len(samples) < num:
        std = np.sqrt(1 / alpha)
        x = np.random.normal(0, std)
        y = x * s
        if np.random.uniform(0, 1) <= (1 / ((1 + x ** 2) ** k)):
            samples.append(y)

    return np.array(samples)


def theoretical_value(k, alpha, s, num):
    sample_kurtosis_list = []
    for i in range(1000):
        np.random.seed(500 + i)
        samples = generate_noise_data(k, alpha, s, num)

        sample_kurtosis = kurtosis(samples, fisher=True)
        sample_kurtosis_list.append(sample_kurtosis)

    mean_sample_kurtosis = np.mean(sample_kurtosis_list)
    std_sample_kurtosis = np.std(sample_kurtosis_list)
    print(f"\nAverage Sample Kurtosis: {mean_sample_kurtosis:.2f}")
    print(f"\nStd Sample Kurtosis: {std_sample_kurtosis:.2f}")


# Run the program
if __name__ == '__main__':
    df = pd.read_csv('./Data/Auto_data.csv')
    data_1 = df['horsepower']
    data_2 = df['mpg']
    data_2_log = np.log(data_2)
    mean_data_1 = data_1.mean()
    std_data_1 = data_1.std()
    data_1_standardized = (data_1 - mean_data_1) / std_data_1
    data = (data_1_standardized, data_2_log)

    # fitted 7-degree polynomial regression model with Gaussian error
    A_hat1 = [-0.296597, 0.072299, -0.123224, 0.041173, 0.048936, -0.032108, 0.005132]
    constant_hat1 = 3.037590
    p_hat1 = 7
    # fitted 2-degree polynomial regression model with Gaussian error
    A_hat2 = [-0.344804, 0.057793]
    constant_hat2 = 3.040667
    p_hat2 = 2
    # fitted 2-degree polynomial regression model with Nxt error
    A_hat3 = [-0.38378966, 0.07995221]
    constant_hat3 = 3.0287626955229086
    p_hat3 = 2

    print('Sample estimates and SE of fitted 7-degree polynomial regression model with Gaussian error')
    calculate_result(data, A_hat1, p_hat1, constant_hat1)
    print('Sample estimates and SE of fitted 2-degree polynomial regression model with Gaussian error')
    calculate_result(data, A_hat2, p_hat2, constant_hat2)
    print('Sample estimates and SE of fitted 2-degree polynomial regression model with Nxt error')
    calculate_result(data, A_hat3, p_hat3, constant_hat3)

    print('Theoretical value and the std of fitted 2-degree polynomial regression model with Nxt error')
    alpha_true = 0.4973334074020386
    k_true = 3.238938808441162
    s_true = 0.3757480986690269
    num_samples = 10000
    theoretical_value(k_true, alpha_true, s_true, num_samples)
