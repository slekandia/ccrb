import numpy as np


def reorder(indices, mode):
    """Reorders the elements
    """
    indices = list(indices)
    element = indices.pop(mode)
    return ([element] + indices[::-1])


def mat2tens(unfolded, shape, mode):
    """Returns the folded tensor of shape `shape` from the `mode`-mode unfolding `unfolded`.
    """
    unfolded_indices = reorder(range(len(shape)), mode)
    original_shape = [shape[i] for i in unfolded_indices]
    unfolded = unfolded.reshape(original_shape)

    folded_indices = list(range(len(shape) - 1, 0, -1))
    folded_indices.insert(mode, 0)
    return np.transpose(unfolded, folded_indices)


def tens2mat(tensor, mode):
    """
    Contracts a tens according to the n-th mode.

    Input: tens of size
           mode is the axis at which the tensor will be contracted
    Output: the tensor matrix product where the ith dimension is replaced by the row dimension of the matrix
    """

    d = tensor.shape
    nd = len(tensor.shape)
    assert mode < nd, "The mode should be less than the dimension of the tensor"

    row_d = d[mode]
    return np.transpose(tensor, reorder(range(tensor.ndim), mode)).reshape((row_d, -1))


def tmprod(tensor, mat, mode):
    """
    Computes the mode-n product of a tensor and a matrix.

    Input: Tensor an n-dimensional tensor
           mat a matrix
    Output: The resulting tensor matrix product.
    """
    if (1 in mat.shape):
        out_shape = list(tensor.shape)
        out_shape[mode] = 1
        result = np.zeros(out_shape)
        # Iterate over each mode-n slice and perform dot product with the vector
        for idx in range(tensor.shape[mode]):
            result = result + np.take(tensor, idx, mode) * mat[idx]
        return result
    else:
        out_n = np.matmul(mat, tens2mat(tensor, mode))
        out_shape = list(tensor.shape)
        out_shape[mode] = mat.shape[0]
        return mat2tens(out_n, out_shape, mode)


def khatri_rao(A, B):
    """
    Computes the column wise kronecker product

    """
    sz_A = np.shape(A)
    sz_B = np.shape(B)
    C = np.zeros((sz_A[0] * sz_B[0], sz_A[1]))
    for r in range(sz_A[1]):
        C[:, r] = np.kron(A[:, r], B[:, r])
    return C


def mlsvd(tensor, ranks=None):
    """
    Computes the multilinear singular value decomposition of a tensor and returns the core matrices and the factor
    matrices.

    Input: an N dimensional tensor
           args stand for a list with ranks for each N dimensions that will be used to truncate the tensor

    Output: the core matrices, a list of factor matrices, and the singular values in each unfolding
    """
    factors = []
    singular_values = []
    nd = len(tensor.shape)

    for n in range(nd):
        tensor_n = tens2mat(tensor, n)
        U, S, Vt = np.linalg.svd(tensor_n, full_matrices=False)
        if ranks is None:
            factors.append(U)
            singular_values.append(S)
        else:
            factors.append(U[:, 0:ranks[n]])
            singular_values.append(S[0:ranks[n]])
        if n == 0:
            core = tmprod(tensor, factors[n].conj().T, n)
        else:
            core = tmprod(core, factors[n].conj().T, n)
    return factors, core, singular_values


def tens2vec(tensor):
    vec_indices = list(range(tensor.ndim - 1, -1, -1))
    return np.transpose(tensor, vec_indices).flatten()


def vec2tens(vec, shape):
    tens = vec.reshape(shape[::-1])
    return np.transpose(tens,list(range(len(shape) - 1, -1, -1)))


def lmlragen(U, S):
    """
    Returns the tensor T that multiplies each factor matrix in U in the corresponding mode with the core matrix S

    Input:
        U : List of the factor matrices
        S : The core tensor
    Output:
        T: The tensor that is the multiplication each factor matrix in U in the corresponding mode with
         the core matrix S
    """
    nd = len(U)
    for n in range(nd):
        if n == 0:
            T = tmprod(S, U[n], n)
        else:
            T = tmprod(T, U[n], n)
    return T


def generate(size_tens, rank_tens):
    """
    Returns a random tensor of size "size_tens" with the rank "rank_tens". The core and the factor matrices are sampled
    from the normal distribution with variance 1. In addition, the factor matrices are randomized according to the Haar
    measure.

    Input:
        size_tens: the size of the tensor
        rank_tens: the n-rank of the tensor
    Output:
        the random tensor of size "size_tens" with the rank "rank_tens"
    """
    s = np.random.normal(0, 1, size=rank_tens)
    u_list = []
    for i in range(len(size_tens)):
        u = np.random.normal(0, 1, size=[size_tens[i],
                                         rank_tens[i]])
        # Haar measure
        q, r = np.linalg.qr(u)
        u = q @ np.diag(np.sign(np.diag(r)))
        u_list.append(u)
    return lmlragen(u_list, s)


def generate_core_fac(size_tens, rank_tens):
    """
    Returns a random tensor of size "size_tens" with the rank "rank_tens". The core and the factor matrices are sampled
    from the normal distribution with variance 1. In addition, the factor matrices are randomized according to the Haar
    measure.

    Input:
        size_tens: the size of the tensor
        rank_tens: the n-rank of the tensor
    Output:
        the random tensor of size "size_tens" with the rank "rank_tens"
    """
    s = np.random.normal(0, 1, size=rank_tens)
    u_list = []
    for i in range(len(size_tens)):
        u = np.random.normal(0, 1, size=[size_tens[i],
                                         rank_tens[i]])
        # Haar measure
        q, r = np.linalg.qr(u)
        u = q @ np.diag(np.sign(np.diag(r)))
        u_list.append(u)
    return u_list, s
