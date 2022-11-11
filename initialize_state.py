import numpy as np
from scipy import linalg
from scipy.constants import Boltzmann

from hamiltonians import get_particles_spin_operators, get_system_hamiltonian


def initialize_state(nqubits, gamma_s, B, T, entanglement):
    ket_0_c = np.zeros((1, 2**nqubits)); ket_0_c[0, 0] = 1.0
    ket_1_c = np.zeros((1, 2**nqubits)); ket_1_c[0,-1] = 1.0
    # System density matrix (thermal state: rho = exp{-beta*H_s}/Tr(exp{-beta*H_s}))
    beta = 1/(T*Boltzmann)
    S_a, S_b = get_particles_spin_operators(0)
    H_s = get_system_hamiltonian(S_a, S_b, gamma_s, B)
    rho_s = linalg.expm(-beta*H_s)
    rho_s = rho_s/np.trace(rho_s)
    # Camera density matrix (rho = ket{GHZ}bra{GHZ} or ket{0}bra{0})
    c_state = (ket_0_c + ket_1_c)/np.sqrt(2) if entanglement else ket_0_c
    rho_c = np.dot(c_state.T, c_state)
    # Global density matrix
    rho = np.kron(rho_s, rho_c)

    return rho

