import tensorly as tl
from tensor_operations import *
from constrained_cramer_rao_bound import constrained_cramer_rao_bound, resolve_sign_ambiguity
import matplotlib.pyplot as plt

monte_c = 10**3
SNR_range = [0, 5, 10, 15, 20, 25, 30]
tr_crb_constrained = np.zeros((len(SNR_range),))
tr_oracle_bound = np.zeros((len(SNR_range),))
err_mlsvd = np.zeros((monte_c, len(SNR_range)))
errs_hooi = np.zeros((monte_c, len(SNR_range)))

tens_t = generate([3, 3, 3], [2, 2, 2])
U_true, S_true, sv = mlsvd(tens_t, [2, 2, 2])
theta = np.hstack((U_true[0].flatten(), U_true[1].flatten(), U_true[2].flatten(), S_true.flatten()))
print(" The singular value of the first mode unfolding is: " + str(sv[0]))
print(" The singular value of the second mode unfolding is: " + str(sv[1]))
print(" The singular value of the third mode unfolding is: " + str(sv[2]))
for i, SNR_i in enumerate(SNR_range):
    for mon_c in range(monte_c):
        wgn = np.random.normal(0, 1, size=tens_t.shape)
        wgn = (wgn - np.mean(wgn)) / np.std(wgn)
        c = (np.var(tens_t) / 10 ** (SNR_i / 10)) / np.var(wgn)
        wgn = np.sqrt(c) * wgn
        noisy_t = tens_t + wgn

        if(mon_c == 0):
            oracle_bound, crao_constrained = constrained_cramer_rao_bound(U_true, S_true, np.std(wgn))
            tr_crb_constrained[i] = np.trace(crao_constrained)
            tr_oracle_bound[i] = np.trace(oracle_bound)
        # Calculate the estimates
        U_noisy, S_noisy, singular_values = mlsvd(noisy_t, [2, 2, 2])
        noisy_t_tensorly = tl.tensor(noisy_t)
        S_hooi, U_hooi = tl.decomposition.tucker(noisy_t_tensorly, rank=[2, 2, 2], n_iter_max=500, tol=0.000001)

        U_noisy, S_noisy = resolve_sign_ambiguity(U_noisy, U_true, S_noisy)
        U_hooi, S_hooi = resolve_sign_ambiguity(U_hooi, U_true, S_hooi)

        theta_MLSVD = np.hstack((U_noisy[0].flatten(), U_noisy[1].flatten(), U_noisy[2].flatten(), S_noisy.flatten()))
        theta_hooi = np.hstack((U_hooi[0].flatten(), U_hooi[1].flatten(), U_hooi[2].flatten(), S_hooi.flatten()))

        err_mlsvd[mon_c, i] = np.sum((theta - theta_MLSVD) ** 2)
        errs_hooi[mon_c, i] = np.sum((theta - theta_hooi) ** 2)
    print("SNR " + str(SNR_range[i]) + " dB is done")
fig = plt.figure()
plt.rcParams.update({'font.size': 16})
plt.semilogy(SNR_range, tr_crb_constrained, marker='o')
plt.semilogy(SNR_range, tr_oracle_bound, marker='*')
plt.semilogy(SNR_range, np.mean(err_mlsvd, axis=0), marker='x')
plt.semilogy(SNR_range, np.mean(errs_hooi, axis=0), marker='+')
plt.legend(['CCRB', 'OB','tMLSVD', 'HOOI'])
plt.grid()
plt.ylabel('MSE')
plt.xlabel("SNR (dB)")
plt.tight_layout()
plt.show()
plt.savefig('CRB_plot.png')
print('Simulation is finished and "CRB_plot.png" is created')