import argparse
import yaml
import numpy as np
import os
import itertools as it
from numpy.random import uniform

from simulate import simulate
from save_results import save_results


def parse_commandline():
    parser = argparse.ArgumentParser(
        description=(
            "Simulate the interaction between system of particles and camera with multiple sensors"
        )
    )

    parser.add_argument(
        "--config_file",
        "-c",
        required=True,
        help="path to YAML configuration file with simulation options",
    )

    parser.add_argument(
        "--output_folder",
        "-o",
        default="data",
        help="path to the folder to store the results",
    )

    return parser.parse_args()


def add_suffix(file, suffix):
    file_name, file_extension = os.path.splitext(file)
    file = file_name + suffix + file_extension

    return file


def generate_coords(coord_p_ranges, nconfigs):
    nparticles = coord_p_ranges.shape[0]
    coords_p_vals = np.empty((nconfigs, nparticles, 3))
    for i in range(nparticles):
        for j in range(3):
            low = coord_p_ranges[i, j, 0]
            high = coord_p_ranges[i, j, 1]
            coords_p_vals[:, i, j] = uniform(low, high, nconfigs)

    return coords_p_vals


def get_camera_basis(nsensors, spin_s):
    spin_levels = np.arange(spin_s, -(spin_s + 1), -1)
    camera_basis = np.array(list(it.product(spin_levels, repeat=nsensors)))

    return camera_basis


def main():
    args = parse_commandline()
    os.makedirs(args.output_folder, exist_ok=True)
    o_file = os.path.join(args.output_folder, "output.h5")
    with open(args.config_file, "r") as config_file:
        c = yaml.safe_load(config_file)

    nconfigs_vals = c["nconfigs"]
    nsims = len(nconfigs_vals)
    spin_p, spin_s = c['particles']['spin'], c['sensors']['spin']
    coords_s = np.array(c['sensors']['coordinates'])
    nsensors = coords_s.shape[0]
    for nconfigs in nconfigs_vals:
        print(f"Simulating {nconfigs} configurations:")
        coords_p_vals = generate_coords(np.array(c['particles']['coordinates']), nconfigs)
        probs = simulate(coords_p_vals, coords_s, spin_p, spin_s,
                         c["B"], c["t"], c['sensors']['init_state'])
        basis = get_camera_basis(nsensors, spin_s)
        suffix = "" if nsims == 1 else f"_{nconfigs}"
        save_results(basis, probs, coords_p_vals, add_suffix(o_file, suffix))


if __name__ == "__main__":
    main()
