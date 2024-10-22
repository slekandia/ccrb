import numpy as np
import pickle
import tensorly as tl
from tensor_operations import *
from constrained_cramer_rao_bound_w_perm import constrained_cramer_rao_bound_w_perm, resolve_sign_ambiguity
import matplotlib.pyplot as plt
# This is the test for constrained with S
monte_c = 10000
SNR_range = [0, 5, 10, 15, 20, 25, 30]
tr_crb_constrained = np.zeros((monte_c, len(SNR_range)))
tr_crbs_unconstrained = np.zeros((monte_c, len(SNR_range)))

errs = np.zeros((monte_c, len(SNR_range)))
errs_hooi = np.zeros((monte_c, len(SNR_range)))
errs_reconstruction_mlsvd = np.zeros((monte_c, len(SNR_range)))
errs_reconstruction_hooi = np.zeros((monte_c, len(SNR_range)))
ranks = [3,5,7,7]
shape_tens_t = [4, 7, 10, 10]
tens_t = generate(shape_tens_t, ranks)
u, s, sv = mlsvd(tens_t, ranks)
sv1_true = tens2mat(s, 0) @ np.transpose(tens2mat(s, 0))
sv2_true = tens2mat(s, 1) @ np.transpose(tens2mat(s, 1))
sv3_true = tens2mat(s, 2) @ np.transpose(tens2mat(s, 2))
sz = tens_t.shape
import time
start_time = time.perf_counter()
count = 0

for i, SNR_i in enumerate(SNR_range):
    wgn = np.random.normal(0, 1, size=tens_t.shape)
    c = (np.var(tens_t) / 10 ** (SNR_i / 10)) / np.var(wgn)
    wgn = np.sqrt(c) * wgn
    noisy_t = tens_t + wgn

    crao_unconstrained, crao_constrained = constrained_cramer_rao_bound_w_perm(u, s, np.std(wgn))
    tr_crb_constrained[:, i] = np.trace(crao_constrained)
    tr_crbs_unconstrained[:, i] = np.trace(crao_unconstrained)
    for mon_c in range(monte_c):
        print(mon_c)
        wgn = np.random.normal(0, 1, size=tens_t.shape)
        c = (np.var(tens_t) / 10 ** (SNR_i / 10)) / np.var(wgn)
        wgn = np.sqrt(c) * wgn
        noisy_t = tens_t + wgn

        U_noisy, S_noisy, singular_values = mlsvd(noisy_t, ranks)
        noisy_t_tensorly = tl.tensor(noisy_t)
        S_hooi, U_hooi = tl.decomposition.tucker(noisy_t_tensorly, rank=ranks, n_iter_max=2000, tol=0.000001)
        U_noisy, S_noisy = resolve_sign_ambiguity(U_noisy, u, S_noisy)
        U_hooi, S_hooi = resolve_sign_ambiguity(U_hooi, u, S_hooi)

        theta_MLSVD = np.hstack((np.hstack([(U_noisy[i].flatten()) for i in range(len(ranks))]), S_noisy.flatten()))
        theta_hooi = np.hstack((np.hstack([(U_hooi[i].flatten()) for i in range(len(ranks))]), S_hooi.flatten()))
        theta = np.hstack((np.hstack([(u[i].flatten()) for i in range(len(ranks))]), s.flatten()))
        errs[mon_c, i] = np.sum((theta - theta_MLSVD) ** 2)
        errs_hooi[mon_c, i] = np.sum((theta - theta_hooi)**2)
        if(np.trace(crao_constrained) < np.trace(crao_unconstrained)):
            print(str(mon_c) + ' problem')
            count = count + 1
        errs_reconstruction_mlsvd[mon_c, i] = np.linalg.norm(lmlragen(U_noisy, S_noisy) - tens_t) / np.linalg.norm(tens_t)
        errs_reconstruction_hooi[mon_c, i] = np.linalg.norm(lmlragen(U_hooi, S_hooi) - tens_t) / np.linalg.norm(tens_t)
end_time = time.perf_counter()
total_time = end_time - start_time
a = {"crb_constrained": tr_crb_constrained,
     "error_mlsvd": errs,
     "error_hooi": errs_hooi,
     "true_tensor": tens_t,
     "crao_unconstrained": tr_crbs_unconstrained,
     "reconstrerr_mlsvd": errs_reconstruction_mlsvd,
     "reconstrerr_hooi": errs_reconstruction_hooi,
     }

with open('crb_4D_size_3_5_7_7', 'wb') as handle:
    pickle.dump(a, handle, protocol=pickle.HIGHEST_PROTOCOL)

print(f'Total execution time: {total_time} seconds')
fig = plt.figure()
plt.rcParams.update({'font.size': 16})
plt.semilogy(SNR_range, np.mean(tr_crb_constrained,axis=0), marker='o')
plt.semilogy(SNR_range, np.mean(tr_crbs_unconstrained,axis=0), marker='*')
plt.semilogy(SNR_range, np.mean(errs,axis=0), marker='x')
plt.semilogy(SNR_range, np.mean(errs_hooi,axis=0), marker='+')
plt.legend(['CCRB', 'OB','tr-MLSVD', 'HOOI'])
plt.grid()
plt.ylabel('MSE')
plt.xlabel("SNR")
plt.tight_layout()
plt.savefig('CRB_plot.png')

print('done')

