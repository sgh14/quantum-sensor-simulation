from scipy import linalg
from scipy.constants import hbar
from numba import njit, objmode


# def get_U(H, t):
#     U = linalg.expm(-1j*H*t/hbar)

#     return U


@njit
def time_evolution(rho_i, H, t):
    with objmode(U='complex128[:, :]'):
        U = linalg.expm(-1j*H*t/hbar)

    # rho_f = U rho_i U^dagger
    rho_f = U.dot(rho_i.dot(U.conj().T))

    return rho_f
