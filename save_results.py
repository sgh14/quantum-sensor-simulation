import numpy as np
import h5py


def save_results(basis, probs, labels, file_path):
    if np.any(np.abs(np.sum(probs, axis=1) - 1)) > 1e-10:
        print('WARNING: probabilities do not add up to 1')

    file = h5py.File(file_path, "w")
    # Create a dataset in the file
    file.create_dataset("basis", data=basis)
    file.create_dataset("probabilities", data=probs)
    file.create_dataset("labels", data=labels)
    file.close()