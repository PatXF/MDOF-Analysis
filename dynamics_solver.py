import numpy as np
from InverseVectorIteration import *


def solver(m, c, k, u0, u_0, del_t, l, b, end_time, loader, total_time_loading, eval_mode, slope=0, bias=0, load_type="linear", amplitudes=0):
    lambdas, modal_matrix = CalculateModalNumpy(m, k)
    n = modal_matrix.shape[0]
    temp_mode = []
    for i in range(eval_mode):
        eigen_vec = modal_matrix[:, n - i - 1]
        temp_mode.append(eigen_vec)
    modal_matrix = np.array(temp_mode).T

    M = modal_matrix.T @ m @ modal_matrix

    for i in range(eval_mode):
        modal_matrix[:, i] /= np.sqrt(M[i][i])

    M = np.round(modal_matrix.T @ m @ modal_matrix, 3)
    C = np.round(modal_matrix.T @ c @ modal_matrix, 3)
    K = np.round(modal_matrix.T @ k @ modal_matrix, 3)

    P = np.zeros((eval_mode, 1))

    q0 = []
    q_0 = []
    for i in range(eval_mode):
        phi_n = modal_matrix[:, i].reshape((n, 1))
        numerator_q0 = phi_n.T @ m @ u0
        denominator_q0 = phi_n.T @ m @ phi_n
        q0.append(numerator_q0 / denominator_q0)

    q0 = np.round(np.array(q0).reshape((eval_mode, 1)), 3)

    for i in range(eval_mode):
        phi_n = modal_matrix[:, i].reshape((n, 1))
        numerator_q_0 = phi_n.T @ m @ u_0
        denominator_q_0 = phi_n.T @ m @ phi_n
        q_0.append(numerator_q_0 / denominator_q_0)

    q_0 = np.round(np.array(q_0).reshape((eval_mode, 1)), 3)

    q__0 = np.round(np.linalg.inv(M) @ (P - C @ q_0 - K @ q0), 3)
    K_hat = K + (l / (b * del_t)) * C + M / (b * (del_t ** 2))
    K_hat_inv = np.linalg.inv(K_hat)

    a = np.round(M / (b * del_t) + (l / b) * C, 3)
    B = np.round(M / (2 * b) + del_t * (l / (2 * b) - 1) * C, 3)

    curr_time = del_t
    response = []

    while curr_time <= end_time:
        if load_type == "linear":
            Pi = modal_matrix.T @ loader(slope, curr_time, bias)
        else:
            Pi = modal_matrix.T @ loader(amplitudes, curr_time, load_type)
        n = Pi.shape[0]
        if curr_time > total_time_loading:
            Pi = np.zeros((n, 1))

        del_Pi = Pi - P
        del_Pi_hat = del_Pi + a @ q_0 + B @ q__0
        del_qi = K_hat_inv @ del_Pi_hat
        del_q_i = ((l / (b * del_t)) * del_qi) - ((l / b) * q_0) + ((del_t * (1 - l / (2 * b))) * q__0)
        del_q__i = del_qi / (b * (del_t ** 2)) - q_0 / (b * del_t) - q__0 / (2 * b)
        q0 += del_qi
        q_0 += del_q_i
        q__0 += del_q__i
        ui = np.round(modal_matrix @ q0, 4)
        response.append(ui)
        curr_time += del_t
        P = Pi

    return response