from matplotlib import pyplot as plt
import numpy as np


def plot_data(ax, basis, probabilities, D_vals):
    X, Y = np.meshgrid(np.arange(basis.shape[0]), D_vals)
    pc = ax.pcolormesh(X, Y, probabilities, vmin=0, vmax=1)

    return ax, pc


def plot_distribution(basis, probabilities, D_vals, plot_file):
    fig, ax = plt.subplots(figsize=(10, 8))
    ax, pc = plot_data(ax, basis, probabilities, D_vals)
    labels = [''] + ['$|'+str(ket.astype(int))[1:-1]+'\\rangle$' for ket in basis]
    ax.set_xticklabels(labels, rotation=45)
    ax.set_ylabel('$D\;(\mathrm{m})$')
    fig.colorbar(pc)
    ax.set_title('State probability')
    fig.savefig(plot_file)
    