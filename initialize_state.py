import numpy as np


def product_state(ket, nsystems):
    product_ket = ket.copy()
    for _ in range(nsystems - 1):
        product_ket = np.kron(product_ket, ket)

    return product_ket


def system_density_matrix(nparticles, spin_p):
    d = int((2*spin_p + 1)**nparticles)
    rho = 1/d*np.identity(d)

    return rho


def camera_density_matrix(nsensors, spin_s, init_state):
    d = int(2*spin_s + 1)
    spin_levels = np.arange(spin_s, -(spin_s + 1), -1)
    basis = np.identity(d)
    if init_state == 'GHZ':
        kets = np.array([product_state(basis[i], nsensors) for i in range(d)])
        c_state = 1/np.sqrt(d)*np.sum(kets, axis=0, keepdims=True)
    else:
        c_state = product_state(basis[spin_levels == init_state], nsensors)  

    rho = np.dot(c_state.T, c_state)

    return rho


def initialize_state(nparticles, nsensors, spin_p, spin_s, init_state):
    rho_s = system_density_matrix(nparticles, spin_p)
    rho_c = camera_density_matrix(nsensors, spin_s, init_state)
    rho = np.kron(rho_s, rho_c)

    return rho
