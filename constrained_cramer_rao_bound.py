import numpy as np
import tensor_operations as to
import scipy
import itertools as it
import sys


def get_noise_covariance_fac2fac_w_perm2(i, j, mode_i, mode_j, shape, sigma):
    """
    shape: shape of the original tensor
    i: index of the column selection vector corresponding to mode_i
    j: index of the column selection vector corresponding to mode_j
    mode_i: ith unfolding
    mode_j: jth unfolding
    """
    # Calculate the total size of the tensor
    total_size = np.prod(shape)

    idx_x = list(shape)
    idx_y = list(shape)
    idx_x[mode_i] = 1
    idx_y[mode_j] = 1


    return


def get_noise_covariance_fac2fac_w_perm(i, j, mode_i, mode_j, shape, sigma):
    """
    shape: shape of the original tensor
    i: index of the column selection vector corresponding to mode_i
    j: index of the column selection vector corresponding to mode_j
    mode_i: ith unfolding
    mode_j: jth unfolding
    """
    total_size = np.prod(shape)
    dim = len(shape)
    noise_cov_fac2fac = np.zeros(((total_size / shape[mode_i]).astype('int'), (total_size / shape[mode_j]).astype('int')))
    idx_x = list(shape)
    idx_y = list(shape)
    idx_x[mode_i] = 1
    idx_y[mode_j] = 1
    ranges_x = []
    ranges_y = []
    for dim_i in range(dim):
        if dim_i == 0:
            if dim_i == mode_j:
                bias_x = j
            else:
                ranges_x.append(range(0, idx_x[dim_i]))
        else:
            step = np.prod(np.array(idx_x[0:dim_i]))
            if dim_i == mode_j:
                bias_x = j * step
            else:
                ranges_x.append(step * range(0, idx_x[dim_i]))
    for dim_i in range(dim):
        if dim_i == 0:
            if dim_i == mode_i:
                bias_y = i
            else:
                ranges_y.append(range(0, idx_y[dim_i]))
        else:
            step = np.prod(np.array(idx_y[0:dim_i]))
            if dim_i == mode_i:
                bias_y = i * step
            else:
                ranges_y.append(step * range(0, idx_y[dim_i]))
    for x, y in it.zip_longest(it.product(*ranges_x), it.product(*ranges_y)):
        noise_cov_fac2fac[np.sum(x).astype('int') + bias_x, np.sum(y).astype('int') + bias_y] = sigma**2
    return noise_cov_fac2fac

def get_perm_vec(mode_i, mode_j, tensor_shape):

    """
    Construct the permutation matrix P such that A = PB, where A and B
    are the column-wise vectorizations of the tensor unfolded in mode_i and mode_j.
    """
    # Calculate the size of the tensor
    size = np.prod(tensor_shape)

    # Initialize the permutation matrix P
    P = np.zeros((size, size), dtype=int)

    # Generate all possible indices for the N-dimensional tensor
    indices = np.indices(tensor_shape).reshape(len(tensor_shape), -1, order='F').T

    for idx in indices:
        # Calculate index_A for unfolding in mode_i
        idx_A = list(idx)
        i = idx_A.pop(mode_i)
        idx_A = [i] + idx_A
        shape_A = [tensor_shape[mode_i]] + list(tensor_shape[:mode_i]) + list(tensor_shape[mode_i + 1:])

        index_A = np.ravel_multi_index(idx_A, shape_A, order='F')

        # Calculate index_B for unfolding in mode_j
        idx_B = list(idx)
        j = idx_B.pop(mode_j)
        idx_B = [j] + idx_B
        shape_B = [tensor_shape[mode_j]] + list(tensor_shape[:mode_j]) + list(tensor_shape[mode_j + 1:])

        index_B = np.ravel_multi_index(idx_B, shape_B, order='F')

        # Set the corresponding entry in P
        P[index_A.astype('int'), index_B.astype('int')] = 1

    return P


def get_triangular_parts(matrix, diagonal):
    """
    Extracts the strict upper and lower triangular parts of a matrix.

    Parameters:
    matrix (numpy.ndarray): A 2D numpy array (matrix).

    Returns:
    tuple: (strict_upper, strict_lower) where each is a 2D numpy array.
    """
    if not isinstance(matrix, np.ndarray):
        raise TypeError("Input must be a numpy.ndarray.")

    if matrix.ndim != 2:
        raise ValueError("Input must be a 2D matrix.")

    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError("Matrix must be square.")

    # Create masks for the strict lower triangular parts
    if(diagonal):
        # Lower triangular values (including diagonal)
        c_idx = [np.ravel_multi_index(mask_i, matrix.shape) for mask_i in zip(*np.tril_indices(matrix.shape[0]))]
        mask = np.sort(np.ravel_multi_index(np.unravel_index(np.array(c_idx, dtype='int'), matrix.shape)[::-1],
                                    matrix.shape[::-1]))
        triangular = matrix[np.unravel_index(mask, matrix.shape, order='F')]
    else:

        c_idx = [np.ravel_multi_index(mask_i, matrix.shape) for mask_i in zip(*np.tril_indices(matrix.shape[0], k=-1))]
        # Lower triangular values (excluding diagonal)
        mask = np.sort(np.ravel_multi_index(np.unravel_index(np.array(c_idx, dtype='int'), matrix.shape)[::-1], matrix.shape[::-1]))
        triangular = matrix[np.unravel_index(mask, matrix.shape, order='F')]
    return np.unravel_index(mask, matrix.shape, order='F'), triangular

def get_fisher(grads_fac, grads_fac_to_S, grads_S_to_S):

    # Check if constr_u and constr_S are non-empty lists

    num_fac_blocks = len(grads_fac_to_S)
    column_hops = np.zeros((num_fac_blocks+1, ), dtype='int')
    for i in range(0,num_fac_blocks):
       column_hops[i] = grads_fac[i].shape[1]
    column_hops[-1] = grads_fac_to_S[0].shape[1]
    column_hops = np.cumsum(column_hops)
    # Determine the total number of rows and columns
    total_rows = column_hops[-1]

    # Initialize the block matrix with zeros
    C = np.zeros((total_rows, total_rows))


    row_block_number = num_fac_blocks - 1
    hop_count = 0
    current_row = 0
    block_num_increase = row_block_number
    col_hop = 0
    current_col = 0
    for count_grad_i, fac_block in enumerate(grads_fac):
        fac_shape = fac_block.shape
        C[col_hop + current_row:col_hop + current_row + fac_shape[0], col_hop + current_col:col_hop + current_col + fac_shape[1]] = fac_block
        C[col_hop + current_col:col_hop + current_col + fac_shape[1], col_hop + current_row:col_hop + current_row + fac_shape[0]] = fac_block.T
        current_col += fac_shape[1]
        if (count_grad_i == row_block_number):
            # Fill in the grads_fac_to_S blocks
            fac_to_S_shape = grads_fac_to_S[hop_count].shape
            C[col_hop + current_row:col_hop + current_row + fac_to_S_shape[0], col_hop + current_col:col_hop + current_col + fac_to_S_shape[1]] = grads_fac_to_S[hop_count]
            C[col_hop + current_col:col_hop + current_col + fac_to_S_shape[1], col_hop + current_row:col_hop + current_row + fac_to_S_shape[0]] = grads_fac_to_S[hop_count].T
            row_block_number = row_block_number + block_num_increase
            block_num_increase -= 1
            current_col = 0
            col_hop = column_hops[hop_count]
            hop_count = hop_count + 1


    C[col_hop:, col_hop:] = grads_S_to_S
    return C


def get_noise_covariance_fac2core_w_perm(i, mode_i, tensor_shape, sigma):
    """
    shape: shape of the original tensor
    i: index of the column selection vector corresponding to mode_i
    j: index of the column selection vector corresponding to mode_j
    mode_i: ith unfolding
    mode_j: jth unfolding
    """
    total_size = np.prod(tensor_shape)
    P = np.zeros(((total_size / tensor_shape[mode_i]).astype('int'), (total_size).astype('int')))

    # Generate all possible indices for the N-dimensional tensor
    indices = np.indices(tensor_shape).reshape(len(tensor_shape), -1, order='F').T

    for idx in indices:
        if(idx[mode_i] == i):
            idx_A = list(idx)
            idx_A.pop(mode_i)
            shape_A = list(tensor_shape[:mode_i]) + list(tensor_shape[mode_i + 1:])

            index_A = np.ravel_multi_index(idx_A, shape_A, order='F')

            # Calculate index_B for unfolding in mode_j
            index_B = np.ravel_multi_index(idx, tensor_shape, order='F')
            # Set the corresponding entry in P
            P[index_A.astype('int'), index_B.astype('int')] = sigma**2
    return P


def get_C(constr_u, constr_S):
    # Check if constr_u and constr_S are non-empty lists
    if not constr_u or not constr_S:
        raise ValueError("constr_u and constr_S must be non-empty lists")

    # Determine the total number of rows and columns
    total_rows = sum(u.shape[0] for u in constr_u) + sum(s.shape[0] for s in constr_S)
    total_cols = sum(u.shape[1] for u in constr_u) + constr_S[0].shape[1]

    # Initialize the block matrix with zeros
    C = np.zeros((total_rows, total_cols))

    # Fill in the constr_u blocks
    current_row = 0
    current_col = 0
    for u_block in constr_u:
        u_shape = u_block.shape
        C[current_row:current_row + u_shape[0], current_col:current_col + u_shape[1]] = u_block
        current_row += u_shape[0]
        current_col += u_shape[1]
    # Fill in the constr_S blocks
    for s_block in constr_S:
        s_shape = s_block.shape
        C[current_row:current_row + s_shape[0], current_col:] = s_block
        current_row += s_shape[0]
    return C


def get_derivative_S_ST(matrix, mode, tensor_shape):
    SS_T = np.zeros((int((matrix.shape[0] * (matrix.shape[0] - 1)) / 2), int(np.prod(matrix.shape))))
    idx, _ = get_triangular_parts(matrix @ matrix.T, False)
    #  Generate all possible indices for the unfolded S matrix
    for j in range(matrix.shape[1]):
        for i in range(matrix.shape[0]):
            J_ij = np.zeros((matrix.shape))
            J_ji = np.zeros((matrix.shape[::-1]))
            J_ij[i, j] = 1
            J_ji[j, i] = 1
            _, SS_T[:, i + j * matrix.shape[0]] = get_triangular_parts(matrix @ J_ji + J_ij @ matrix.T, False)
    P = get_perm_vec(0, mode, tensor_shape)
    for i in range(len(idx[0])):
        SS_T[i, :] = (P @ np.array(SS_T[i, :]).reshape(-1, 1)).reshape(-1, )
    return SS_T

def get_derivative_U_TU(matrix):
    derivative_U_TU = np.zeros((int((matrix.shape[1] * (matrix.shape[1] + 1)) / 2), int(np.prod(matrix.shape))))

    for j in range(matrix.shape[1]):
        for i in range(matrix.shape[0]):
            J_ij = np.zeros((matrix.shape))
            J_ji = np.zeros((matrix.shape[::-1]))
            J_ij[i, j] = 1
            J_ji[j, i] = 1
            _, derivative_U_TU[:, i + j * matrix.shape[0]] = get_triangular_parts(matrix.T @ J_ij + J_ji @ matrix, True)
    return derivative_U_TU


def constrained_cramer_rao_bound_w_perm(U, S, sigma):
    """
    U: a list with n factor matrices
    S: core tensor
    sigma: the zero mean gaussian noise standard deviation
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
                                #print(np.linalg.norm((get_noise_covariance_fac2fac_w_perm(alpha, gamma, i, j, sz_original_tensor, sigma)) - (get_noise_covariance_fac2fac_w_perm2(alpha, gamma, i, j, sz_original_tensor, sigma))))
                                tmp[beta * U_i_1.shape[0] + alpha, eta * U_i_2.shape[0] + gamma] = 1 / sigma ** 4 \
                                    * e_beta.T @  S_unfold_i @ kron_remaining_facs_U_1.T @ (get_noise_covariance_fac2fac_w_perm(alpha, gamma, i, j, sz_original_tensor, sigma)) @ kron_remaining_facs_U_2 @ S_unfold_j.T @ e_eta
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
                    * e_beta.T @ S_unfold_i @ kron_remaining_facs.T @ (get_noise_covariance_fac2core_w_perm(alpha, i, sz_original_tensor, sigma)) @ kron_all_facs
        grads_fac_to_S.append(tmp)

    # Calculate the gradient wrt to S
    grads_S_to_S = 1 / sigma ** 2 * np.eye(np.prod(S.shape), np.prod(S.shape))

    constr_u = []
    constr_S = []
    for i, U_i in enumerate(U):
        constr_u.append(get_derivative_U_TU(U_i))

    for dim_i in range(total_number_of_facs):
        constr_S.append(get_derivative_S_ST(to.tens2mat(S, dim_i), dim_i, S.shape))


    C = get_C(constr_u, constr_S)
    V_null = scipy.linalg.null_space(C)
    if(~np.isclose(C.shape[1]-C.shape[0],V_null.shape[1])):
        print("CCRB exists.")
    # Merge everything back to the fisher information matrix
    fisher_info_mat = get_fisher(grads_fac, grads_fac_to_S, grads_S_to_S)
    # print(np.linalg.cond(V_null.T @ fisher_info_mat @ V_null))
    print("Rank of FIM is " + str(np.linalg.matrix_rank(fisher_info_mat)) + " and N_theta - N_c is " + str((C.shape[1] - C.shape[0])))
    if np.linalg.cond(V_null.T @ fisher_info_mat @ V_null) < 1 / sys.float_info.epsilon:
        inv_fish_info_mat_constrained = V_null @ np.linalg.inv(V_null.T @ fisher_info_mat @ V_null) @ V_null.T
    else:
        "Non singular"
        inv_fish_info_mat_constrained = V_null @ np.linalg.pinv(V_null.T @ fisher_info_mat @ V_null) @ V_null.T
    inv_fish_info_mat_unconstrained = np.linalg.pinv(fisher_info_mat)
    #return np.trace(inv_fish_info_mat)
    return inv_fish_info_mat_unconstrained, inv_fish_info_mat_constrained

def resolve_sign_ambiguity(U_est, U_true, S_est):
    eps = 1
    for unfold_i, u_est in enumerate(U_est):
        for u_i in range(u_est.shape[1]):
            if np.linalg.norm(u_est[:, u_i] - U_true[unfold_i][:, u_i]) > np.linalg.norm(u_est[:, u_i] + U_true[unfold_i][:, u_i]):
                U_est[unfold_i][:, u_i] = -1 * U_est[unfold_i][:, u_i]
                tmp = to.tens2mat(S_est, unfold_i)
                tmp[u_i, :] = -1 * tmp[u_i, :]
                S_est = to.mat2tens(tmp, S_est.shape, unfold_i)
    return U_est, S_est
