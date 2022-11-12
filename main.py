import argparse
import yaml
import numpy as np
from numpy.random import uniform
import os

from simulate import simulate
from save_results import save_results


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

    return parser.parse_args()


def main():
    args = parse_commandline()
    with open(args.config_file, 'r') as config_file:
        config = yaml.safe_load(config_file)

    nconfigs = config['nconfigs']
    config['theta']['low'] = np.radians(config['theta']['low'])
    config['theta']['high'] = np.radians(config['theta']['high'])
    labels = np.empty((nconfigs, 3))
    for i, varname in enumerate(('D', 'theta', 'd_s')):
        var = config.pop(varname)
        config[varname + '_vals'] = uniform(var['low'], var['high'], nconfigs)
        labels[:, i] = config[varname + '_vals']
        
    measures = simulate(**config)
    save_results(measures, labels, args.output_file)
    

if __name__ == "__main__":
    main()
