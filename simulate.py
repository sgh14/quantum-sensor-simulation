import numpy as np
from numpy import random
import itertools as it
from tqdm import tqdm

from initialize_state import initialize_state
from hamiltonians import get_H
from time_evolution import get_U, time_evolution
from measure import get_probabilities, measure


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
    # Bais of states of the camera
    c_basis = np.array(list(it.product(spin_levels, repeat=nqubits)))
    projectors = np.array([np.diag(row) for row in np.identity(2**nqubits)])
    # Initialize the global state
    rho_0 = initialize_state(nqubits, gamma_s, B, T, entanglement)
    # Generate combinations of D, theta and d_s and simulate
    measures = np.empty((nconfigs, nmeasures, nqubits))
    D_vals = random.uniform(D['low'], D['high'], nconfigs)
    theta_vals = random.uniform(theta['low'], theta['high'], nconfigs)
    d_s_vals = random.uniform(d_s['low'], d_s['high'], nconfigs)
    for i in tqdm(range(nconfigs)):
        _D, _theta, _d_s = D_vals[i], theta_vals[i], d_s_vals[i]
        # Build the hamiltonian
        H = get_H(nqubits, d_c, _d_s, _D, _theta, gamma_s, gamma_c, B)
        # Time evolution
        U = get_U(H, t)
        rho_t = time_evolution(rho_0, U)
        # Measure
        probabilities = get_probabilities(rho_t, projectors)
        measures[i] = measure(c_basis, probabilities, nmeasures)

    return measures