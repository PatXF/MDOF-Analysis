import numpy as np


def CalculateEigenVector(m, k, tolerance, initial_lambda):
    x_i = np.ones((m.shape[0], 1))
    lambda_i = initial_lambda
    found = False
    while not found:
        k_inv = np.linalg.inv(k - lambda_i * m)
        x_j = k_inv @ m @ x_i
        numerator = (x_j.T @ m @ x_i)[0][0]
        denominator = (x_j.T @ m @ x_j)[0][0]
        lambda_j = (numerator / denominator) + lambda_i
        norm_den = np.sqrt(x_j.T @ m @ x_j)
        x_i = x_j / norm_den
        if abs(lambda_j - lambda_i) / lambda_j <= tolerance:
            return np.round(lambda_j, 3), np.round(x_i, 3)

        lambda_i = lambda_j


def CalculateModalMatrix(m, k, tolerance=0.001):
    prev_lambda = 0
    n = m.shape[0]
    lambdas = []
    modal_matrix = np.empty((n, 0))
    for i in range(n):
        converged = False
        if len(lambdas) != 0:
            prev_lambda = 2 * lambdas[-1]
        while not converged:
            lambda_i, eigen_i = CalculateEigenVector(m, k, tolerance, prev_lambda)
            if len(lambdas) == 0 or lambda_i not in lambdas:
                lambdas.append(float(lambda_i))
                modal_matrix = np.hstack((modal_matrix, eigen_i))
                converged = True

            prev_lambda += 1

    return lambdas, modal_matrix


def CalculateModalNumpy(m, k):
    A = np.linalg.inv(m) @ k
    return np.linalg.eig(A)


