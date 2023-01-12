import numpy as np
from numpy.random import choice


def partial_trace_with_reshape(A, subsystems):
    partial_A = A.copy()
    n = A.shape[0]
    for k, subsystem in enumerate(sorted(subsystems)):
        i = subsystem - k
        dim_left = 2**i
        dim_right = n//2**(i+1)
        # if ket{psi} = ket{L}ket{P}ket{R}
        # --> A_{ijklmn} = bra{L_i}bra{P_j}bra{R_k} A ket{L_l}ket{P_m}ket{R_n}
        # --> partial_A = delta^{jm} A_{ijklmn}
        new_shape = (dim_left, 2, dim_right, dim_left, 2, dim_right)
        partial_A = partial_A.reshape(new_shape)
        partial_A = np.trace(partial_A, axis1=1, axis2=4)
        n = dim_left*dim_right
        partial_A = partial_A.reshape(n, n)

    return partial_A


def partial_trace_without_reshape(rho, subsystems):
    partial_rho = rho.copy()
    for k, subsystem in enumerate(sorted(subsystems)):
        n = partial_rho.shape[0] 
        i = subsystem - k
        dim_a = 2**i
        dim_b = 2
        dim_c = n//2**(i+1)
        I_a = np.identity(dim_a)
        I_b = np.identity(dim_b)
        I_c = np.identity(dim_c)
        L = np.array([np.kron(np.kron(I_a, b_i), I_c) for b_i in I_b])
        R = np.array([np.kron(np.kron(I_a, b_i.reshape(-1, 1)), I_c) for b_i in I_b])
        partial_rho = np.einsum('ipl, lm, imq -> pq', L, partial_rho, R)

    return partial_rho


def get_probabilities(rho_f):
    rho_f_camera = partial_trace_with_reshape(rho_f, subsystems=[0, 1])
    probabilities = np.diag(rho_f_camera)
    # probabilities = probabilities/np.sum(probabilities)

    return probabilities.real


def measure(states, probabilities, nmeasures):
    # Sample from states according to probabilities
    indices = list(range(states.shape[0]))
    chosen_indices = choice(indices, size=nmeasures, p=probabilities)
    measures = states[chosen_indices]

    return measures
