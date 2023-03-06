import argparse
import yaml
import numpy as np
from numpy.random import uniform
import os
import itertools as it

from simulate import simulate
from save_results import save_results
from plot_distribution import add_suffix, plot_distribution


def parse_commandline():
    parser = argparse.ArgumentParser(
        description=("Simulate the interaction between a two particle system and a chain of qubits"))

    parser.add_argument(
        "--config_file",
        "-c",
        required=True,
        help="path to YAML configuration file with simulation options")

    parser.add_argument(
        "--output_folder",
        "-o",
        default='data', # os.path.join('data', 'output.h5'),
        help="path to the folder to store the results")


    return parser.parse_args()


def generate_kwargs(config_dict, nqubits, nconfigs):
    kwargs = config_dict.copy()
    kwargs['nqubits'], kwargs['nconfigs'] = nqubits, nconfigs
    for varname in ('D', 'theta', 'd_s'):
        var = kwargs.pop(varname)
        kwargs[varname + '_vals'] = uniform(var['low'], var['high'], nconfigs)

    kwargs['theta_vals'] = np.radians(kwargs['theta_vals'])

    return kwargs


def get_camera_basis(nqubits):
    spin_levels = (0, 1)  # 0 = +, 1 = -
    camera_basis = np.array(list(it.product(spin_levels, repeat=nqubits)))

    return camera_basis


def get_labels(kwargs):
    varnames = ('D', 'theta', 'd_s')
    labels = np.stack([kwargs[vname + '_vals'] for vname in varnames], axis=1)
    
    return labels


def main():
    args = parse_commandline()
    os.makedirs(args.output_folder, exist_ok=True)
    o_file = os.path.join(args.output_folder, 'output.h5')
    p_file = os.path.join(args.output_folder, 'dist.png')
    with open(args.config_file, 'r') as config_file:
        c = yaml.safe_load(config_file)

    nqubits_vals, nconfigs_vals = c['nqubits'], c['nconfigs']
    nsims = len(nqubits_vals)*len(nconfigs_vals)
    for nqubits in nqubits_vals:
        for nconfigs in nconfigs_vals:
            print(f'Simulating {nconfigs} configurations for {nqubits} qubits:')
            kwargs = generate_kwargs(c, nqubits, nconfigs)
            probabilities = simulate(**kwargs)
            camera_basis = get_camera_basis(nqubits)
            labels = get_labels(kwargs)
            suffix = '' if nsims == 1 else f'_{nqubits}_{nconfigs}'
            results = (camera_basis, probabilities, labels)
            save_results(*results, add_suffix(o_file, suffix))
            plot_distribution(*results, add_suffix(p_file, suffix))
    

if __name__ == "__main__":
    main()
