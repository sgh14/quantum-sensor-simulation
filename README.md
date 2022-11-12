# Quantum-sensor-simulation
Simulate the interaction between a two particle system and a chain of qubits.

## Set up

```
$ conda env create -f environment.yml
```

## Usage

Specify the parameters of the simulation using a configuration YAML file and run

```
$ python main.py -c [config_file] -o [output_file]
```

An example of a configuration file is `config.yml`. The output file must have `.h5` extension.