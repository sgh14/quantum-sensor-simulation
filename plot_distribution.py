from matplotlib import pyplot as plt
import numpy as np
from os.path import splitext


def plot_probs(ax, basis, probabilities, D_vals): 
    X, Y = np.meshgrid(np.arange(basis.shape[0]), D_vals)
    pc = ax.pcolormesh(X, Y, probabilities, vmin=0, vmax=1)

    return ax, pc


def plot_mean_vals(ax, basis, probabilities, D_vals):
    mean_vals = np.einsum('jr, ij -> ir', basis, probabilities)
    p = ax.plot(D_vals, mean_vals)

    return ax, p


def plot_distribution(basis, probabilities, D_vals, plot_file):
    file_name, file_extension = splitext(plot_file)
    order = np.argsort(D_vals)
    D_vals, probabilities = D_vals[order], probabilities[order]
    
    fig, ax = plt.subplots(figsize=(10, 8))
    ax, pc = plot_probs(ax, basis, probabilities, D_vals)
    labels = [''] + ['$|'+str(ket.astype(int))[1:-1]+'\\rangle$' for ket in basis]
    ax.set_xticklabels(labels, rotation=45)
    ax.ticklabel_format(style='sci', axis='y', scilimits=(-1,1), useMathText=True)
    ax.set_ylabel('$D\;(\mathrm{m})$')
    fig.colorbar(pc)
    ax.set_title('State probability')
    fig.savefig(file_name + '_probs' + file_extension)

    fig, ax = plt.subplots(figsize=(10, 8))
    ax, p = plot_mean_vals(ax, basis, probabilities, D_vals)
    labels = [f'Qubit {i+1}' for i in range(basis.shape[1])]
    ax.legend(p, labels, bbox_to_anchor=(0.5, 1.025), loc="lower center", ncol=5)
    ax.set_ylim(0, 1) 
    ax.ticklabel_format(style='sci', axis='x', scilimits=(-1,1), useMathText=True)
    ax.set_xlabel('$D\;(\mathrm{m})$')
    ax.set_ylabel('$\\langle\\sigma_z\\rangle$')
    ax.set_title('Spin mean values ($0\equiv \hbar/2,\; 1\equiv-\hbar/2$)', pad=50)
    fig.savefig(file_name + '_mean_vals' + file_extension)
    