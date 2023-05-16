import numpy as np
from numpy.random import choice


def partial_trace(A, subsystem_dim, dim_left, dim_right):
    # if ket{psi} = ket{L}ket{P}ket{R}
    # --> A_{ijklmn} = bra{L_i}bra{P_j}bra{R_k} A ket{L_l}ket{P_m}ket{R_n}
    # --> partial_A = delta^{jm} A_{ijklmn}
    partial_A = A.copy()
    new_shape = (dim_left, subsystem_dim, dim_right,
                 dim_left, subsystem_dim, dim_right)
    partial_A = partial_A.reshape(new_shape)
    partial_A = np.trace(partial_A, axis1=1, axis2=4)
    n = dim_left*dim_right
    partial_A = partial_A.reshape(n, n)

    return partial_A


def get_probabilities(rho_f, nparticles, spin_p):
    rho_f_c = rho_f.copy()
    subsystem_dim = int(2*spin_p + 1)
    for _ in range(nparticles):
        dim_left = 1
        dim_right = rho_f_c.shape[0]//subsystem_dim
        rho_f_c = partial_trace(rho_f_c, subsystem_dim, dim_left, dim_right)

    probabilities = np.diag(rho_f_c)

    return probabilities.real


def measure(states, probabilities, nmeasures):
    indices = np.arange(states.shape[0])
    chosen_indices = choice(indices, size=nmeasures, p=probabilities)
    measures = states[chosen_indices]

    return measures
