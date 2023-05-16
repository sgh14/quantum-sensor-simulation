import numpy as np
from tqdm import tqdm

from initialize_state import * 
from hamiltonians import *
from time_evolution import *
from measure import *


def simulate(coords_p_vals, coords_s, spin_p, spin_s, B, t):
    nparticles, nsensors = coords_p_vals.shape[1], coords_s.shape[0]
    nconfigs = coords_p_vals.shape[0]
    rho_0 = initialize_state(nparticles, nsensors, spin_p, spin_s)    
    s_p = particles_spin_operators(nparticles, nsensors, spin_p, spin_s)
    s_s = sensors_spin_operators(nparticles, nsensors, spin_p, spin_s)
    H_c = camera_hamiltonian(s_s, coords_s, B)
    probabilities = np.empty((nconfigs, int((2*spin_s + 1)**nsensors)))
    for i in tqdm(range(nconfigs)):
        H_s = system_hamiltonian(s_p, coords_p_vals[i], B)
        H_cs = interaction_hamiltonian(s_s, s_p, coords_s, coords_p_vals[i])
        H_total = H_c + H_s + H_cs
        rho_t = time_evolution(rho_0, H_total, t) 
        probabilities[i] = get_probabilities(rho_t, nparticles, spin_p) 

    return probabilities