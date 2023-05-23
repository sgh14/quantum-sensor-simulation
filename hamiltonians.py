import numpy as np
from scipy.constants import mu_0, hbar

from constants import D, gamma_e #, sigmas


def sigmas(spin):
    d = int(2*spin + 1)
    s_x, s_y, s_z = np.empty((3, d, d), dtype='cfloat')
    delta = np.identity(d+2)
    for i in range(d):
        for j in range(d):
            a, b = i+1, j+1
            s_x[i, j] = 1/2*(delta[a,b+1] + delta[a+1,b])*np.sqrt(np.abs((spin+1)*(a+b-1)-a*b))
            s_y[i, j] = 1j/2*(delta[a,b+1] - delta[a+1,b])*np.sqrt(np.abs((spin+1)*(a+b-1)-a*b))
            s_z[i, j] = (spin+1-a)*delta[a,b]

    s = np.array([s_x, s_y, s_z])

    return s


def extend_operator(A, dim_left, dim_right):
    I_left = np.identity(dim_left)
    I_right = np.identity(dim_right)
    A_extended = np.kron(np.kron(I_left, A), I_right)

    return A_extended


def particles_spin_operators(nparticles, nsensors, spin_p, spin_s):
    d_p, d_s = int(2*spin_p + 1), int(2*spin_s + 1)
    d = (d_p**nparticles)*(d_s**nsensors)
    s = np.empty((nparticles, 3, d, d), dtype='cfloat')
    for i in range(nparticles):
        dim_left = d_p**i
        dim_right = (d_p**(nparticles - (i + 1)))*(d_s**nsensors)
        s[i] = extend_operator(sigmas(spin_p), dim_left, dim_right)
    
    return s


def sensors_spin_operators(nparticles, nsensors, spin_p, spin_s):
    d_p, d_s = int(2*spin_p + 1), int(2*spin_s + 1)
    d = (d_p**nparticles)*(d_s**nsensors)
    s = np.empty((nsensors, 3, d, d), dtype='cfloat')
    for i in range(nsensors):
        dim_left = (d_p**nparticles)*(d_s**i)
        dim_right = d_s**(nsensors - (i + 1))
        s[i] = extend_operator(sigmas(spin_s), dim_left, dim_right)
    
    return s


def inner_dipolar_term(s, coords, gamma=gamma_e):
    c = mu_0*gamma**2*hbar**2/(4*np.pi)
    H_d = 0
    for i in range(s.shape[0]):
        for j in range(i):
            r_ij = coords[j] - coords[i]
            norm_rij = np.sqrt(np.sum(r_ij**2))
            dot_si_sj = np.sum(s[i] @ s[j], axis=0)
            dot_si_rij = np.einsum('kpq, k -> pq', s[i], r_ij/norm_rij)
            dot_sj_rij = np.einsum('kpq, k -> pq', s[j], r_ij/norm_rij)
            H_d += c*(dot_si_sj - 3*dot_si_rij @ dot_sj_rij)/norm_rij**3

    return H_d


def external_dipolar_term(s_1, s_2, coords_1, coords_2, gamma_1=gamma_e, gamma_2=gamma_e):
    c = mu_0*gamma_1*gamma_2*hbar**2/(4*np.pi)
    H_d = 0
    for i in range(s_1.shape[0]):
        for j in range(s_2.shape[0]):
            r_ij = coords_2[j] - coords_1[i]
            norm_rij = np.sqrt(np.sum(r_ij**2))
            dot_si_sj = np.sum(s_1[i] @ s_2[j], axis=0)
            dot_si_rij = np.einsum('kpq, k -> pq', s_1[i], r_ij/norm_rij)
            dot_sj_rij = np.einsum('kpq, k -> pq', s_2[j], r_ij/norm_rij)
            H_d += c*(dot_si_sj - 3*dot_si_rij @ dot_sj_rij)/norm_rij**3
    
    return H_d


def camera_hamiltonian(s, coords, B, gamma=gamma_e):
    nsensors = s.shape[0]
    H_0_c = hbar*np.sum(D*s[:, 2] @ s[:, 2] - gamma*B*s[:, 2], axis=0)# - 2/3*hbar*D*np.identity(s.shape[-1])*nsensors
    if nsensors > 1:
        H_I_c = inner_dipolar_term(s, coords)
    else:
        H_I_c = 0

    H_c = H_0_c + H_I_c

    return H_c


def system_hamiltonian(s, coords, B, gamma=gamma_e):
    nparticles = s.shape[0]
    H_0_s = -hbar*gamma*B*np.sum(s[:, 2], axis=0)
    if nparticles > 1:
        H_I_s = inner_dipolar_term(s, coords)
    else:
        H_I_s = 0

    H_s = H_0_s + H_I_s

    return H_s


def interaction_hamiltonian(s_s, s_p, coords_s, coords_p):
    H_cs = external_dipolar_term(s_s, s_p, coords_s, coords_p)

    return H_cs
