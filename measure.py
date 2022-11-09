import numpy as np


def partial_trace(rho, subsystems):
    partial_rho = rho.copy()
    n = rho.shape[0]
    for k, subsystem in enumerate(sorted(subsystems)):
        i = subsystem - k
        dim_left = 2**i
        dim_right = n//2**(i+1)
        # if ket{psi} = ket{L}ket{P}ket{R}
        # --> M_{ijklmn} = bra{L_i}bra{P_j}bra{R_k} M ket{L_l}ket{P_m}ket{R_n}
        # --> partial_rho = delta^{jm} rho_{ijklmn}
        new_shape = (dim_left, 2, dim_right, dim_left, 2, dim_right)
        partial_rho = partial_rho.reshape(new_shape)
        partial_rho = np.trace(partial_rho, axis1=1, axis2=4)
        n = dim_left*dim_right
        partial_rho = partial_rho.reshape(n, n)

    return partial_rho


def get_probabilities(rho_f, projectors):
    rho_f_camera = partial_trace(rho_f, subsystems=[0, 1])
    # p_k = Tr{dot(rho_f_camera, P_k)}
    probabilities = np.einsum('ij, kji -> k', rho_f_camera, projectors)
    # probabilities = probabilities/np.sum(probabilities)

    return probabilities.real


def measure(states, probabilities, nmeasures=1):
    # Sample from states according to probabilities
    indices = list(range(states.shape[0]))
    chosen_indices = np.random.choice(indices, size=nmeasures, p=probabilities)
    measures = states[chosen_indices]

    return measures
