import numpy as np


def system_density_matrix(nparticles, spin_p):
    d = int((2*spin_p + 1)**nparticles)
    rho = 1/d*np.identity(d)

    return rho


def camera_density_matrix(nsensors, spin_s):
    ket_0_s = np.array([[0], [1], [0]]) if spin_s == 1 else np.array([[1], [0]])
    ket_0 = 1
    for _ in range(nsensors):
        ket_0 = np.kron(ket_0, ket_0_s)

    rho = np.dot(ket_0, ket_0.T)

    return rho


def initialize_state(nparticles, nsensors, spin_p, spin_s):
    rho_s = system_density_matrix(nparticles, spin_p)
    rho_c = camera_density_matrix(nsensors, spin_s)
    rho = np.kron(rho_s, rho_c)

    return rho

