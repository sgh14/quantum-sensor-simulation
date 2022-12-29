import numpy as np
import os
import h5py


def save_results(basis, probs, labels, file_path):
    dirname = os.path.dirname(file_path)
    # Create a new HDF5 file
    if dirname:
        os.makedirs(dirname, exist_ok=True)
        
    file = h5py.File(file_path, "w")
    # Create a dataset in the file
    file.create_dataset("basis", data=basis)
    file.create_dataset("probabilities", data=probs)
    file.create_dataset("labels", data=labels)
    file.close()