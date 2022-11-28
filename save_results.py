import numpy as np
import os
import h5py


def save_results(basis, probs, labels, file_path):
    # Create a new HDF5 file
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    file = h5py.File(file_path, "w")
    # Create a dataset in the file
    data_format = h5py.h5t.STD_U8BE
    file.create_dataset("basis", np.shape(basis), data_format, basis)
    file.create_dataset("probabilities", np.shape(probs), data_format, probs)
    file.create_dataset("labels", np.shape(labels), data_format, labels)
    file.close()