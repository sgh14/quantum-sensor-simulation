# Quantum-sensor-simulation
Simulate the interaction between a two particle system and a chain of qubits.

<p align="center"><img title="Sensor diagram" src="images/qsensor_diagram.png" width=50%></p>

## Structure

<!-- TODO: update image -->

<p align="center"><img title="Dependencies" src="images/dependencies.png" width=75%></p>

## Set up

```
$ conda env create -f environment.yml
```

## Usage

Specify the parameters of the simulation using a configuration YAML file and run

```
$ python main.py -c [config_file] -o [output_folder]
```

An example of a configuration file is `config.yml`. The parameter `output_folder` takes a default value if it is not specified.

## Example

The following plots are obtained when running

```
$ python main.py -c config.yml
```

<p align="center">
<img title="dist_d_s_state_probs" src="images/dist_d_s_state_probs.png" width=49%>
<img title="dist_d_s_spin_probs" src="images/dist_d_s_spin_probs.png" width=49%>
</p>
<p align="center">
<img title="dist_D_state_probs" src="images/dist_D_state_probs.png" width=49%>
<img title="dist_D_spin_probs" src="images/dist_D_spin_probs.png" width=49%>
</p>
<p align="center">
<img title="dist_theta_state_probs" src="images/dist_theta_state_probs.png" width=49%>
<img title="dist_theta_spin_probs" src="images/dist_theta_spin_probs.png" width=49%>
</p>