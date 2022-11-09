from scipy import linalg
from scipy import constants as cts


def get_U(H, t):
    U = linalg.expm(-1j*H*t/cts.hbar)

    return U


def time_evolution(rho_i, U):
    # rho_f = U rho_i U^dagger
    rho_f = U.dot(rho_i.dot(U.conj().T))

    return rho_f
