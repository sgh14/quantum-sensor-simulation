import numpy as np
import itertools as it
from numba import njit, objmode

from initialize_state import * 
from hamiltonians import *
from time_evolution import *
from measure import *


@njit(parallel=True)
def simulate(
    D_vals,
    theta_vals,
    d_s_vals,
    d_c,
    gamma_s,
    gamma_c,
    B,
    t,
    nmeasures,
    nqubits,
    T,
    entanglement,
    nconfigs,
    progress_bar
):
    spin_levels = np.array([0, 1])  # 0 = +, 1 = -
    with objmode(camera_basis='int64[:, :]'):
        camera_basis = np.array(list(it.product(spin_levels, repeat=nqubits)))

    rho_0 = initialize_state(nqubits, gamma_s, B, T, entanglement)    
    S_a, S_b = get_particles_spin_operators(nqubits)
    S = get_qubits_spin_operators(nqubits)
    H_c = get_camera_hamiltonian(S, gamma_c, B)
    H_s = get_system_hamiltonian(S_a, S_b, gamma_s, B)
    measures = np.empty((nconfigs, nmeasures, nqubits))
    for i in range(nconfigs):
        D, theta, d_s = D_vals[i], theta_vals[i], d_s_vals[i]
        H_cs = get_interaction_hamiltonian(d_c, d_s, D, theta, S_a, S_b, S, gamma_s, gamma_c)
        H_total = H_c + H_s + H_cs
        rho_t = time_evolution(rho_0, H_total, t)
        with objmode():
            probabilities = get_probabilities(rho_t)
            measures[i] = measure(camera_basis, probabilities, nmeasures)

        progress_bar.update(1)

    return measures