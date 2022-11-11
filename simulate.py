import numpy as np
from numpy import random
import itertools as it
from tqdm import tqdm

from initialize_state import * 
from hamiltonians import *
from time_evolution import *
from measure import *


def simulate(
    D,
    theta,
    d_s,
    d_c,
    gamma_s,
    gamma_c,
    B,
    t,
    nmeasures,
    nqubits,
    T,
    entanglement,
    nconfigs
):
    spin_levels = (0, 1)  # 0 = +, 1 = -
    camera_basis = np.array(list(it.product(spin_levels, repeat=nqubits)))
    rho_0 = initialize_state(nqubits, gamma_s, B, T, entanglement)    
    S_a, S_b = get_particles_spin_operators(nqubits)
    S = get_qubits_spin_operators(nqubits)
    H_c = get_camera_hamiltonian(S, gamma_c, B)
    H_s = get_system_hamiltonian(S_a, S_b, gamma_s, B)
    D_vals = random.uniform(D['low'], D['high'], nconfigs)
    theta_vals = random.uniform(theta['low'], theta['high'], nconfigs)
    d_s_vals = random.uniform(d_s['low'], d_s['high'], nconfigs)
    measures = np.empty((nconfigs, nmeasures, nqubits))
    for i in tqdm(range(nconfigs)):
        D_i, theta_i, d_s_i = D_vals[i], theta_vals[i], d_s_vals[i]
        H_cs = get_interaction_hamiltonian(
            d_c, d_s_i, D_i, theta_i, S_a, S_b, S, gamma_s, gamma_c
        )
        H_total = H_c + H_s + H_cs
        rho_t = time_evolution(rho_0, H_total, t)
        probabilities = get_probabilities(rho_t)
        measures[i] = measure(camera_basis, probabilities, nmeasures)

    return measures