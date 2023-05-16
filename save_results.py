import numpy as np
import h5py


def save_results(basis, probs, coords_p_vals, output_file):
    if np.any(np.abs(np.sum(probs, axis=1) - 1) > 1e-6):
        print('WARNING: probabilities do not add up to 1')

    nparticles = coords_p_vals.shape[1]
    with h5py.File(output_file, "w") as file:
        file.create_dataset("basis", data=basis)
        file.create_dataset("probabilities", data=probs)
        for i in range(nparticles):
            file.create_dataset(f"coords_{i}", data=coords_p_vals[:, i])
