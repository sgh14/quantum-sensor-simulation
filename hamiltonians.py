import numpy as np
from scipy import constants as cts


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


def get_H_c(S, gamma_c, B):
    omega = -gamma_c*B
    S_x, S_z = S[:, 0], S[:, 2]
    # H_c = sum_{i, j} omega/2*S_iz + lambdas(i,j)*dot(S_ix, S_jx)
    H_c = omega/2*np.einsum('ipq -> pq', S_z)\
        + omega/10*np.einsum('ipr, jrq -> pq', S_x, S_x)  # TODO: caso i=j???

    return H_c


def get_H_s(S_a, S_b, gamma_s, B):
    omega = -gamma_s*B
    S_az, S_bz = S_a[2], S_b[2]
    H_s = omega/2*(S_az + S_bz)

    return H_s


def get_H_cs(d_c, d_s, D, theta, S_a, S_b, S, gamma_s, gamma_c):
    nqubits = S.shape[0]
    c = -cts.mu_0/(4*np.pi)*gamma_s*gamma_c
    # Get positions
    position_a, position_b = get_particles_coordinates(D, d_s, theta)
    positions_qubits = get_qubits_coordinates(nqubits, d_c)
    # Get relative polar coordinates
    r_a_qubits, theta_a_qubits = cartesian2polar(position_a - positions_qubits)
    r_b_qubits, theta_b_qubits = cartesian2polar(position_b - positions_qubits)
    # Build dipole-dipole interaction factors
    g_a = c*(1-3*np.cos(theta_a_qubits)**2)/r_a_qubits**3
    g_b = c*(1-3*np.cos(theta_b_qubits)**2)/r_b_qubits**3
    # H_cs = sum_i g_a(i)dot(S_ax, s_ix) + g_b(i)dot(S_bx, s_ix)
    S_ax, S_bx, S_x = S_a[0], S_b[0], S[:, 0]
    H_cs = np.einsum('i, pr, irq -> pq', g_a, S_ax, S_x)\
         + np.einsum('i, pr, irq -> pq', g_b, S_bx, S_x)

    return H_cs


def get_H(nqubits, d_c, d_s, D, theta, gamma_s, gamma_c, B):
    nparticles = 2
    n = nparticles + nqubits
    a = cts.hbar/2
    # Build extended spin operators
    S_a = extend_operator(a*sigmas, dim_left=2**0, dim_right=2**(nqubits+1))
    S_b = extend_operator(a*sigmas, dim_left=2**1, dim_right=2**nqubits)
    # s = [[s_1x, s_1y, s_1z], ...], [s_nx, s_ny, s_nz]]
    S = np.empty((nqubits, 3, 2**n, 2**n), dtype='cfloat')
    for i in range(nqubits):
        dim_left = 2**(i+nparticles)
        dim_right = 2**(nqubits-1-i)
        S[i] = extend_operator(a*sigmas, dim_left=dim_left, dim_right=dim_right)

    # Build the hamiltonians
    H_c = get_H_c(S, gamma_c, B)
    H_s = get_H_s(S_a, S_b, gamma_s, B)
    H_cs = get_H_cs(d_c, d_s, D, theta, S_a, S_b, S, gamma_s, gamma_c)
    H = H_c + H_s + H_cs

    return H
