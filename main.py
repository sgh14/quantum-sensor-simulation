import argparse
import yaml
import numpy as np

from simulate import simulate
from save_results import save_results


def parse_commandline():
    parser = argparse.ArgumentParser(
        description=("Simulate the interaction between a two particle system and a chain of qubits"))

    parser.add_argument(
        'config_file',
        help="path to YAML configuration file with simulation options")

    # TODO: add --output_file

    return parser.parse_args()


def main():
    args = parse_commandline()
    with open(args.config_file, 'r') as config_file:
        config = yaml.safe_load(config_file)

    config['theta']['low'] = np.radians(config['theta']['low'])
    config['theta']['high'] = np.radians(config['theta']['high'])
    # measures, labels = 
    measures = simulate(**config)
    # parameters = 
    # filename = f'.h5'
    save_results(measures) #labels, parameters, filename=filename
    

if __name__ == "__main__":
    main()
