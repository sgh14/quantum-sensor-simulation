#%%
import numpy as np
from scipy import linalg
from scipy import constants as cts

from hamiltonians import sigmas, extend_operator, get_H_s


def initialize_state(nqubits, gamma_s, B, T, entanglement):
    c_basis = np.identity(2**nqubits)
    ket_0_c, ket_1_c = c_basis[0, np.newaxis], c_basis[-1, np.newaxis]
    # System density matrix (thermal state: rho = exp{-beta*H_s}/Tr(exp{-beta*H_s}))
    beta = 1/(T*cts.Boltzmann)
    S_a = extend_operator(cts.hbar/2*sigmas, dim_left=2**0, dim_right=2**1)
    S_b = extend_operator(cts.hbar/2*sigmas, dim_left=2**1, dim_right=2**0)
    H_s = get_H_s(S_a, S_b, gamma_s, B)
    rho_s = linalg.expm(-beta*H_s)
    rho_s = rho_s/np.trace(rho_s)
    # Camera density matrix (rho = ket{GHZ}bra{GHZ} or ket{0}bra{0})
    c_state = (ket_0_c + ket_1_c)/np.sqrt(2) if entanglement else ket_0_c
    rho_c = np.dot(c_state.T, c_state)
    # Global density matrix
    rho = np.kron(rho_s, rho_c)

    return rho

# %%
