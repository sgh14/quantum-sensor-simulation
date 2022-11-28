import argparse
import yaml
import numpy as np
from numpy.random import uniform
import os
import itertools as it

from simulate import simulate
from save_results import save_results
from plot_distribution import plot_distribution


def parse_commandline():
    parser = argparse.ArgumentParser(
        description=("Simulate the interaction between a two particle system and a chain of qubits"))

    parser.add_argument(
        "--config_file",
        "-c",
        required=True,
        help="path to YAML configuration file with simulation options")

    parser.add_argument(
        "--output_file",
        "-o",
        default=os.path.join('data', 'output.h5'),
        help="path to the .h5 file to store the results")

    parser.add_argument(
        "--plot_file",
        "-p",
        default=os.path.join('data', 'distribution.png'),
        help="path to the file to save the plot of the distribution")

    return parser.parse_args()


def main():
    args = parse_commandline()
    with open(args.config_file, 'r') as config_file:
        c = yaml.safe_load(config_file)

    nconfigs = c['nconfigs']
    c['theta']['low'] = np.radians(c['theta']['low'])
    c['theta']['high'] = np.radians(c['theta']['high'])
    labels = np.empty((nconfigs, 3))
    for i, varname in enumerate(('D', 'theta', 'd_s')):
        var = c.pop(varname)
        c[varname + '_vals'] = uniform(var['low'], var['high'], nconfigs)
        labels[:, i] = c[varname + '_vals']
        
    probabilities = simulate(**c)
    spin_levels = (0, 1)  # 0 = +, 1 = -
    camera_basis = np.array(list(it.product(spin_levels, repeat=c['nqubits'])))
    save_results(camera_basis, probabilities, labels, args.output_file)
    plot_distribution(camera_basis, probabilities, c['D_vals'], args.plot_file)
    

if __name__ == "__main__":
    main()
