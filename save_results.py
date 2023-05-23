import numpy as np
import h5py


def save_results(basis, probs, coords_p_vals, output_file):
    largest_diff = np.max(np.abs(np.sum(probs, axis=1) - 1))
    print(f'WARNING: the largest difference between the sums of probabilities and 1 is {largest_diff}')

    nparticles = coords_p_vals.shape[1]
    with h5py.File(output_file, "w") as file:
        file.create_dataset("basis", data=basis)
        file.create_dataset("probabilities", data=probs)
        for i in range(nparticles):
            file.create_dataset(f"coords_{i}", data=coords_p_vals[:, i])
