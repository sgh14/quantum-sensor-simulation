# Quantum-sensor-simulation
Simulate the interaction between a two particle system and a chain of qubits.

<p align="center"><img title="Sensor diagram" src="images/qsensor_diagram.png" width=50%></p>

## Structure

<p align="center"><img title="Dependencies" src="images/dependencies.png" width=75%></p>

## Set up

```
$ conda env create -f environment.yml
```

## Usage

Specify the parameters of the simulation using a configuration YAML file and run

```
$ python main.py -c [config_file] -o [output_file] -p [plot_file]
```

An example of a configuration file is `config.yml`. The output file must have `.h5` extension. The parameters `output_file` and `plot_file` take default values if they are not specified.

## Example

The following plots are obtained when running

```
$ python main.py -c config.yml -p images/distribution.png
```

<p align="center"><img title="distribution_probs" src="images/distribution_probs.png" width=75%></p>

<p align="center"><img title="distribution_mean_vals" src="images/distribution_mean_vals.png" width=75%></p>