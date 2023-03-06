from matplotlib import pyplot as plt
import numpy as np
from os.path import splitext


def add_suffix(file, suffix):
    file_name, file_extension = splitext(file)
    file = file_name + suffix + file_extension

    return file


def plot_state_probs(basis, probabilities, label, label_name, file):
    fig, ax = plt.subplots(figsize=(10, 8)) 
    X, Y = np.meshgrid(np.arange(basis.shape[0]), label)
    pc = ax.pcolormesh(X, Y, probabilities, vmin=0, vmax=1)
    labels = np.array(['$|'+str(ket.astype(int))[1:-1]+'\\rangle$' for ket in basis])
    nticks = len(labels) if len(labels) < 2**5 else 2**5
    xticks = np.arange(0, len(labels), len(labels)//nticks)
    ax.set_xticks(xticks)
    ax.set_xticklabels(labels[xticks], rotation=45)
    ax.ticklabel_format(style='sci', axis='y', scilimits=(-1,1), useMathText=True)
    ax.set_ylabel('$' + label_name + '\;(\mathrm{m})$')
    fig.colorbar(pc, label='State probability')
    fig.savefig(file)


def plot_spin_probs(basis, probabilities, label, label_name, file):
    fig, ax = plt.subplots(figsize=(10, 8))
    # multiply each state by its probability and sum
    probs_equal_0 = 1 - np.einsum('jr, ij -> ir', basis, probabilities)
    p = ax.plot(label, probs_equal_0, alpha=0.6)
    labels = [f'Qubit {i+1}' for i in range(basis.shape[1])]
    ax.legend(p, labels, bbox_to_anchor=(0.5, 1.025), loc="lower center", ncol=6)
    ax.set_ylim(0, 1) 
    ax.ticklabel_format(style='sci', axis='x', scilimits=(-1,1), useMathText=True)
    ax.set_xlabel('$' + label_name + '\;(\mathrm{m})$')
    ax.set_ylabel('$p(S_z=\hbar/2)$')
    fig.savefig(file)


def plot_spin_mean_vals(basis, probabilities, label, label_name, file):
    fig, ax = plt.subplots(figsize=(10, 8))
    basis[basis==1] = -1
    basis[basis==0] = 1
    # multiply each state by its probability and sum
    mean_vals = np.einsum('jr, ij -> ir', basis, probabilities)
    p = ax.plot(label, mean_vals, alpha=0.6)
    labels = [f'Qubit {i+1}' for i in range(basis.shape[1])]
    ax.legend(p, labels, bbox_to_anchor=(0.5, 1.025), loc="lower center", ncol=6)
    ax.set_ylim(-1, 1) 
    ax.ticklabel_format(style='sci', axis='x', scilimits=(-1,1), useMathText=True)
    ax.set_xlabel('$' + label_name + '\;(\mathrm{m})$')
    ax.set_ylabel('$\\langle S_z\\rangle\; (\hbar/2)$')
    fig.savefig(file)


def plot_distribution(basis, probabilities, labels, plot_file):
    labels_names = ['D', 'theta', 'd_s']
    labels = {labels_names[i]: labels.T[i] for i in range(len(labels_names))}
    for label_name, label in labels.items():
        order = np.argsort(label)
        label, probabilities = label[order], probabilities[order]
        data = (basis, probabilities, label, label_name if label_name!='theta' else '\\theta')
        plot_state_probs(*data, add_suffix(plot_file, '_' + label_name + '_state_probs'))
        plot_spin_probs(*data, add_suffix(plot_file, '_' + label_name + '_spin_probs'))    
        # plot_spin_mean_vals(*data, add_suffix(plot_file, '_' + label_name + '_spin_mean_vals'))    
    