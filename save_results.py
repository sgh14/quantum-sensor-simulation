import numpy as np
import os
import h5py


def save_results(measures, filename='output.h5'): #labels
    # Create a new HDF5 file
    data_dir = 'data'
    os.makedirs(data_dir, exist_ok=True)
    file = h5py.File(os.path.join(data_dir, filename), "w")
    # Create a dataset in the file
    data_format = h5py.h5t.STD_U8BE
    file.create_dataset("measures", np.shape(measures), data_format, measures)
    file.close()