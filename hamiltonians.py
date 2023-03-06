import numpy as np
from scipy.constants import mu_0, hbar


sigma_x = np.array([[0,   1], [1,  0]])
sigma_y = np.array([[0, -1j], [1j, 0]])
sigma_z = np.array([[1,   0], [0, -1]])
sigmas = np.array([sigma_x, sigma_y, sigma_z])


def get_qubits_coordinates(nqubits, d_c):
    L = (nqubits-1)*d_c
    x = np.linspace(-L/2, L/2, nqubits)
    y = np.zeros(nqubits)
    positions = np.stack([x, y], axis=1)

    return positions


def get_particles_coordinates(D, d_s, theta):
    x_a = 0
    y_a = D
    x_b = x_a + d_s*np.cos(theta)
    y_b = y_a + d_s*np.sin(theta)
    positions = np.array([[x_a, y_a], [x_b, y_b]])

    return positions


def cartesian2polar(coordinates):
    r = np.sqrt(np.sum(coordinates**2, axis=1))
    theta = np.arctan2(coordinates[:, 1], coordinates[:, 0])

    return r, theta


def extend_operator(A, dim_left, dim_right):
    I_left = np.identity(dim_left)
    I_right = np.identity(dim_right)
    A_extended = np.kron(np.kron(I_left, A), I_right)

    return A_extended


def get_particles_spin_operators(nqubits):
    S_a = extend_operator(hbar/2*sigmas, dim_left=2**0, dim_right=2**(1+nqubits))
    S_b = extend_operator(hbar/2*sigmas, dim_left=2**1, dim_right=2**nqubits)

    return S_a, S_b


def get_qubits_spin_operators(nqubits):
    nparticles = 2
    n = nparticles + nqubits
    # s = [[s_1x, s_1y, s_1z], ...], [s_nx, s_ny, s_nz]]
    S = np.empty((nqubits, 3, 2**n, 2**n), dtype='cfloat')
    for i in range(nqubits):
        dim_left = 2**(i+nparticles)
        dim_right = 2**(nqubits-1-i)
        S[i] = extend_operator(hbar/2*sigmas, dim_left, dim_right)

    return S
    

def get_camera_hamiltonian(S, gamma_c, B):
    omega = -gamma_c*B
    S_x, S_z = S[:, 0], S[:, 2]
    # TODO: substitute omega/10 by dipole-dipole factors?
    # H_c = sum_{i} omega/2*S_iz + omega/10*dot(S_ix, S_(i+1)x)
    H_c = omega/2*np.einsum('ipq -> pq', S_z)\
        + omega/10*np.einsum('ipr, irq -> pq', S_x[:-1], S_x[1:])

    return H_c


def get_system_hamiltonian(S_a, S_b, gamma_s, B):
    omega = -gamma_s*B
    S_az, S_bz = S_a[2], S_b[2]
    H_s = omega/2*(S_az + S_bz)

    return H_s


def dipole_dipole_factor(r, theta, gamma_1, gamma_2):
    c = -mu_0/(4*np.pi)*gamma_1*gamma_2
    g = c*(3*np.cos(theta)**2 - 1)/r**3

    return g


def get_interaction_hamiltonian(positions_qubits, d_s, D, theta, S_a, S_b, S, gamma_s, gamma_c):
    # Get positions
    position_a, position_b = get_particles_coordinates(D, d_s, theta)
    # Get relative polar coordinates
    r_a_qubits, theta_a_qubits = cartesian2polar(position_a - positions_qubits)
    r_b_qubits, theta_b_qubits = cartesian2polar(position_b - positions_qubits)
    # Build dipole-dipole interaction factors
    g_a = dipole_dipole_factor(r_a_qubits, theta_a_qubits, gamma_s, gamma_c)
    g_b = dipole_dipole_factor(r_b_qubits, theta_b_qubits, gamma_s, gamma_c)
    # H_cs = sum_i g_a(i)dot(S_ax, s_ix) + g_b(i)dot(S_bx, s_ix)
    S_ax, S_bx, S_x = S_a[0], S_b[0], S[:, 0]
    H_cs = np.einsum('i, pr, irq -> pq', g_a, S_ax, S_x)\
         + np.einsum('i, pr, irq -> pq', g_b, S_bx, S_x)

    return H_cs
