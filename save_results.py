import numpy as np
import os
import h5py


def save_results(measures, labels, file_path): #labels
    # Create a new HDF5 file
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    file = h5py.File(file_path, "w")
    # Create a dataset in the file
    data_format = h5py.h5t.STD_U8BE
    file.create_dataset("measures", np.shape(measures), data_format, measures)
    file.create_dataset("labels", np.shape(labels), data_format, labels)
    file.close()