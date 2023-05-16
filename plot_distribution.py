import numpy as np
import os
from matplotlib import pyplot as plt
from scipy.interpolate import griddata


def add_suffix(file, suffix):
    file_name, file_extension = os.path.splitext(file)
    file = file_name + suffix + file_extension

    return file


def plot_probs(basis, probs, coords_p, output_file):
    nsensors, nconfigs = basis.shape[1], probs.shape[0]
    p_0 = np.empty((nconfigs, nsensors))
    for i in range(nconfigs):
        for j in range(nsensors):
            p_0[i, j] = np.sum(probs[i][basis[:, j] == 0])

    vmin, vmax = np.min(p_0), np.max(p_0)
    axis = {0: '$x\; (\\mathrm{m})$', 1: '$y\; (\\mathrm{m})$', 2: '$z\; (\\mathrm{m})$'}
    planes = np.array([[0, 1], [0, 2], [1, 2]])
    nrows, ncols = planes.shape[0], nsensors
    nrows = 1
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(6*ncols, 4*nrows), squeeze=False)
    for i in range(nrows):
        for j in range(ncols):
            c_1, c_2 = coords_p[:, planes[i, 0]], coords_p[:, planes[i, 1]]
            C_1, C_2 = np.meshgrid(np.linspace(min(c_1), max(c_1), 1000),
                                   np.linspace(min(c_2), max(c_2), 1000))
            p_0_j = griddata((c_1, c_2), p_0[:, j], (C_1, C_2), method='nearest')
            pc = axes[i, j].pcolormesh(C_1, C_2, p_0_j, cmap='magma',
                                       vmin=vmin, vmax=vmax)
            axes[i, j].set_aspect('equal')
            axes[i, j].set_xlabel(axis[planes[i, 0]])
            axes[i, j].set_ylabel(axis[planes[i, 1]])
            axes[i, j].ticklabel_format(style='sci', axis='both', scilimits=(-1,1), useMathText=True)
            if i == 0:
                axes[i, j].set_title(f'Sensor {j+1}')
    
    cb = fig.colorbar(pc, ax=axes.ravel().tolist(), label='$P(S_z = 0)$', aspect=40)
    cb.formatter.set_powerlimits((0, 0))
    cb.formatter.set_useMathText(True)
    fig.savefig(output_file, bbox_inches='tight')


def plot_mean_spin(basis, probs, coords_p, output_file):
    mean_spin = np.einsum('jr, ij -> ir', basis, probs)
    vmin, vmax = np.min(mean_spin), np.max(mean_spin)
    nsensors = basis.shape[1]
    axis = {0: '$x\; (\\mathrm{m})$', 1: '$y\; (\\mathrm{m})$', 2: '$z\; (\\mathrm{m})$'}
    planes = np.array([[0, 1], [0, 2], [1, 2]])
    nrows, ncols = planes.shape[0], nsensors
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(16, 12))
    for i in range(nrows):
        for j in range(ncols):
            c_1, c_2 = coords_p[:, planes[i, 0]], coords_p[:, planes[i, 1]]
            C_1, C_2 = np.meshgrid(np.linspace(min(c_1), max(c_1), 1000),
                                   np.linspace(min(c_2), max(c_2), 1000))
            mean_spin_j = griddata((c_1, c_2), mean_spin[:, j], (C_1, C_2), method='nearest')
            pc = axes[i, j].pcolormesh(C_1, C_2, mean_spin_j, cmap='magma',
                                       vmin=vmin, vmax=vmax)
            axes[i, j].set_aspect('equal')
            axes[i, j].set_xlabel(axis[planes[i, 0]])
            axes[i, j].set_ylabel(axis[planes[i, 1]])
            axes[i, j].ticklabel_format(style='sci', axis='both', scilimits=(-1,1), useMathText=True)
            if i == 0:
                axes[i, j].set_title(f'Sensor {j+1}')
    
    cb = fig.colorbar(pc, ax=axes.ravel().tolist(), label='$\\langle S_z\\rangle$', aspect=40)
    cb.formatter.set_powerlimits((0, 0))
    cb.formatter.set_useMathText(True)
    fig.savefig(output_file, bbox_inches='tight')


def plot_distribution(basis, probs, coords_p_vals, output_file):
    nparticles = coords_p_vals.shape[1]
    for i in range(nparticles):
        output_file_i = add_suffix(output_file, f'_particle{i + 1}')
        # plot_mean_spin(basis, probs, coords_p_vals[:, i], output_file_i)
        plot_probs(basis, probs, coords_p_vals[:, i], output_file_i)
