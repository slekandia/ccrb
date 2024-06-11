import numpy as np
import tensor_operations as to
import scipy
import sys


def constrained_cramer_rao_bound(U, S, sigma):
    """
    This is the code for calculating the constrained cramer rao bound and the oracle bound for a low multilinear rank
    tensor estimation problem.
    Input:
    U: a list with n factor matrices
    S: core tensor
    sigma: the zero mean gaussian noise standard deviation
    Output: FIM, CCRB
    """
    # Calculate the fisher information size
    sz_original_tensor = []
    total_number_of_facs = 0
    for i, U_i in enumerate(U):
        total_number_of_facs += 1
        sz_original_tensor.append(U_i.shape[0])

    grads_fac = []
    # Calculate the first n * (n+1) / 2 blocks regarding the factor derivatives
    for i, U_i_1 in enumerate(U):
        for j, U_i_2 in enumerate(U):
            if i > j:
                continue
            if i == j:
                S_unfold_i = to.tens2mat(S, i)
                S_unfold_j = to.tens2mat(S, j)
                tmp = np.zeros((np.prod(U_i_1.shape), np.prod(U_i_2.shape)))
                for beta in range(U_i_1.shape[1]):
                    for alpha in range(U_i_1.shape[0]):
                        for eta in range(U_i_2.shape[1]):
                            for gamma in range(U_i_2.shape[0]):
                                if(alpha == gamma):
                                    e_beta = np.zeros((U_i_1.shape[1], 1))
                                    e_beta[beta] = 1
                                    e_eta = np.zeros((U_i_2.shape[1], 1))
                                    e_eta[eta] = 1
                                    tmp[beta * U_i_1.shape[0] + alpha, eta * U_i_2.shape[0] + gamma] = 1 / sigma ** 2 * e_beta.T @ S_unfold_i @ S_unfold_j.T @ e_eta
                grads_fac.append(tmp)
            else:
                S_unfold_i = to.tens2mat(S, i)
                S_unfold_j = to.tens2mat(S, j)
                tmp = np.zeros((np.prod(U_i_1.shape), np.prod(U_i_2.shape)))

                remaining_facs_U_1 = list(np.arange(0, total_number_of_facs))[::-1]
                del remaining_facs_U_1[total_number_of_facs - i - 1]

                remaining_facs_U_2 = list(np.arange(0, total_number_of_facs))[::-1]
                del remaining_facs_U_2[total_number_of_facs - j - 1]
                for beta in range(U_i_1.shape[1]):
                    for alpha in range(U_i_1.shape[0]):
                        for eta in range(U_i_2.shape[1]):
                            for gamma in range(U_i_2.shape[0]):
                                e_beta = np.zeros((U_i_1.shape[1], 1))
                                e_beta[beta] = 1
                                e_eta = np.zeros((U_i_2.shape[1], 1))
                                e_eta[eta] = 1
                                kron_remaining_facs_U_1 = U[remaining_facs_U_1[0]].copy()
                                for k in remaining_facs_U_1[1:]:
                                    kron_remaining_facs_U_1 = np.kron(kron_remaining_facs_U_1, U[k])
                                kron_remaining_facs_U_2 = U[remaining_facs_U_2[0]].copy()
                                for k in remaining_facs_U_2[1:]:
                                    kron_remaining_facs_U_2 = np.kron(kron_remaining_facs_U_2, U[k])
                                tmp[beta * U_i_1.shape[0] + alpha, eta * U_i_2.shape[0] + gamma] = 1 / sigma ** 4 \
                                    * e_beta.T @  S_unfold_i @ kron_remaining_facs_U_1.T @ (get_noise_covariance_fac2fac(alpha, gamma, i, j, sz_original_tensor, sigma)) @ kron_remaining_facs_U_2 @ S_unfold_j.T @ e_eta
                grads_fac.append(tmp)
    grads_fac_to_S = []
    kron_all_facs = U[-1].copy()
    for i in range(2, len(U) + 1):
        kron_all_facs = np.kron(kron_all_facs, U[-i])

    # Calculate the cross derivatives of factors and S
    for i, U_i_1 in enumerate(U):

        remaining_facs = list(np.arange(0, total_number_of_facs))[::-1]
        del remaining_facs[total_number_of_facs - i - 1]

        kron_remaining_facs = U[remaining_facs[0]].copy()
        for k in remaining_facs[1:]:
            kron_remaining_facs = np.kron(kron_remaining_facs, U[k])

        S_unfold_i = to.tens2mat(S, i)
        tmp = np.zeros((np.prod(U_i_1.shape), np.prod(S.shape)))
        for beta in range(U_i_1.shape[1]):
            for alpha in range(U_i_1.shape[0]):
                e_beta = np.zeros((U_i_1.shape[1], 1))
                e_beta[beta] = 1
                tmp[beta * U_i_1.shape[0] + alpha, :] = 1 / sigma ** 4 \
                    * e_beta.T @ S_unfold_i @ kron_remaining_facs.T @ (get_noise_covariance_fac2core(alpha, i, sz_original_tensor, sigma)) @ kron_all_facs
        grads_fac_to_S.append(tmp)

    # Calculate the gradient wrt to S
    grads_S_to_S = 1 / sigma ** 2 * np.eye(np.prod(S.shape), np.prod(S.shape))
    d_u = []
    for i, U_i in enumerate(U):
        d_u.append(np.asarray([[2 * U_i[0, 0], 2 * U_i[1, 0], 2 * U_i[2, 0], 0, 0, 0],
                                    [U_i[0, 1], U_i[1, 1], U_i[2, 1], U_i[0, 0], U_i[1, 0], U_i[2, 0]],
                                    [0, 0, 0, 2 * U_i[0, 1], 2 * U_i[1, 1], 2 * U_i[2, 1]]]))

    d_s = [
                np.asarray([S[1, 0, 0], S[0, 0, 0], S[1, 1, 0], S[0, 1, 0], S[1, 0, 1], S[0, 0, 1], S[1, 1, 1], S[0, 1, 1]]),
                np.asarray([S[0, 1, 0], S[1, 1, 0], S[0, 0, 0], S[1, 0, 0], S[0, 1, 1], S[1, 1, 1], S[0, 0, 1], S[1, 0, 1]]),
                np.asarray([S[0, 0, 1], S[1, 0, 1], S[0, 1, 1], S[1, 1, 1], S[0, 0, 0], S[1, 0, 0], S[0, 1, 0], S[1, 1, 0]])]
    C = np.block([
        [d_u[0], np.zeros(d_u[1].shape), np.zeros(d_u[2].shape), np.zeros((3,len(d_s[0])))],
        [np.zeros(d_u[0].shape), d_u[1], np.zeros(d_u[2].shape), np.zeros((3,len(d_s[0])))],
        [np.zeros(d_u[0].shape), np.zeros(d_u[1].shape), d_u[2], np.zeros((3,len(d_s[0])))],
        [np.zeros(d_u[0].shape[1]), np.zeros(d_u[1].shape[1]), np.zeros(d_u[2].shape[1]), d_s[0]],
        [np.zeros(d_u[0].shape[1]), np.zeros(d_u[1].shape[1]), np.zeros(d_u[2].shape[1]), d_s[1]],
        [np.zeros(d_u[0].shape[1]), np.zeros(d_u[1].shape[1]), np.zeros(d_u[2].shape[1]), d_s[2]],
    ])

    V = scipy.linalg.null_space(C)
    if(C.shape[1]-C.shape[0] is not V.shape[1]):
        print("ERROR")
    # Merge everything back to the fisher information matrix
    fisher_info_mat = np.block([
        [grads_fac[0], grads_fac[1], grads_fac[2], grads_fac_to_S[0]],
        [grads_fac[1].T, grads_fac[3], grads_fac[4], grads_fac_to_S[1]],
        [grads_fac[2].T, grads_fac[4].T, grads_fac[5], grads_fac_to_S[2]],
        [grads_fac_to_S[0].T, grads_fac_to_S[1].T, grads_fac_to_S[2].T, grads_S_to_S]
    ])
    print("The condition number is " + str(np.linalg.cond(V.T @ fisher_info_mat @ V)))
    inv_fish_info_mat = np.linalg.pinv(fisher_info_mat)
    if np.linalg.cond(V.T @ fisher_info_mat @ V) < 1 / sys.float_info.epsilon:
        inv_ccrb_fish_info_mat = V @ np.linalg.inv(V.T @ fisher_info_mat @ V) @ V.T
    else:
        print("CCRB does not exist")
        inv_ccrb_fish_info_mat = V @ np.linalg.pinv(V.T @ fisher_info_mat @ V) @ V.T
    return inv_fish_info_mat, inv_ccrb_fish_info_mat


def get_noise_covariance_fac2fac(i, j, mode_i, mode_j, shape, sigma):
    """
    shape: shape of the original tensor
    i: index of the column selection vector corresponding to mode_i
    j: index of the column selection vector corresponding to mode_j
    mode_i: ith unfolding
    mode_j: jth unfolding
    """
    total_size = np.prod(shape)
    indices = to.vec2tens(np.arange(0, total_size), shape)
    mode_i_flatten = to.tens2mat(indices, mode_i)
    mode_j_flatten = to.tens2mat(indices, mode_j)
    e_i = np.zeros(shape[mode_i],).reshape(-1, 1)
    e_i[i] = 1
    e_j = np.zeros(shape[mode_j],).reshape(-1, 1)
    e_j[j] = 1
    select_col_i = mode_i_flatten.T @ e_i
    select_row_j = e_j.T @ mode_j_flatten
    noise_cov_fac2fac = np.zeros((select_col_i.shape[0], select_row_j.shape[1]))
    for k in range(select_col_i.shape[0]):
        for m in range(select_row_j.shape[1]):
            if select_col_i[k, 0] == select_row_j[0, m]:
                noise_cov_fac2fac[k, m] = sigma**2
    return noise_cov_fac2fac


def get_noise_covariance_fac2core(i, mode_i, shape, sigma):
    total_size = np.prod(shape)
    vec_indices = np.arange(0, total_size)
    indices = to.vec2tens(vec_indices, shape)
    mode_i_flatten = to.tens2mat(indices, mode_i)
    e_i = np.zeros(shape[mode_i], ).reshape(-1, 1)
    e_i[i] = 1
    select_col_i = mode_i_flatten.T @ e_i
    noise_cov_fac2core = np.zeros((select_col_i.shape[0], len(vec_indices)))
    for k in range(len(select_col_i)):
        for m in range(len(vec_indices)):
            if( select_col_i[k, 0] == vec_indices[m]):
                noise_cov_fac2core[k, m] = sigma**2
    return noise_cov_fac2core


def resolve_sign_ambiguity(U_est, U_true, S_est):
    eps = 1
    for unfold_i, u_est in enumerate(U_est):
        for u_i in range(u_est.shape[1]):
            if np.linalg.norm(u_est[:, u_i] + U_true[unfold_i][:, u_i]) < eps and np.linalg.norm(u_est[:, u_i] - U_true[unfold_i][:, u_i]) > eps:
                U_est[unfold_i][:, u_i] = -1 * U_est[unfold_i][:, u_i]
                tmp = to.tens2mat(S_est, unfold_i)
                tmp[u_i, :] = -1 * tmp[u_i, :]
                S_est = to.mat2tens(tmp, S_est.shape, unfold_i)
    return U_est, S_est
