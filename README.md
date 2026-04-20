# Diff-HyMD-HhPF
∂-HyMD-HhPF (read diff-HyMD) is [HyMD](https://github.com/Cascella-Group-UiO/HyMD) rewritten in [JAX](https://github.com/google/jax).
This is the Hamiltonian hybrid Particle-Field version of [∂-HyMD](https://github.com/Cascella-Group-UiO/Diff-HyMD).
The main goal is to automatically learn force field parameters while running differentiable molecular dynamics simulations.

The paper describing ∂-HyMD-HhPF with application to train a force field for lipids is available [here](https://pubs.acs.org/doi/full/10.1021/acs.jcim.4c00564).

## Installation
> **Note**:
> If installing on Saga or Betzy you need to first load the `python` and `openmpi` modules
> ```terminal
> module load Python/3.11.3-GCCcore-12.3.0
> module load OpenMPI/4.1.5-GCC-12.3.0
> ```
> and then proceed with the installation.

Clone the repo on your machine and create a virtual enviroment inside a directory `<dir>` of your choice
```terminal
cd Diff-HyMD-HhPF
python -m venv --upgrade-deps <dir>
```
Then you can simply install the package with
```terminal
source <dir>/bin/activate
pip install .[v1]
```
## Example usage
To run a simple MD simulation you can use
```terminal
cd examples
diff_hymd mdrun -f dppc/input.h5 -p dppc/topol.toml -c dppc/options.toml -o dppc/simulation -v
```
Instead, to optimize force field parameters you can run
```terminal
cd examples
sed -i -e s/10000/200/ dppc/options.toml # Use smaller number of steps when training
diff_hymd optimize -f input.h5 -p topol.toml -c options.toml -o dppc/train -m dppc/training.toml -v
```
## Optimization
Optimization requires a bit more work, so carefully check `dppc/training.toml` for all the available options.

Inside `training.toml` we need to specify a `system` list.
The elements of this list are directories that each contain the inputs to `diff_hymd`, with the same name provided in the command line
(in the example above, these are `input.h5`, `topol.toml`, and `options.toml`).
The system directories paths are relative to the working directory path from which you call `diff_hymd`.

It is also possible to run multiple replicas of the optimzation in parallel, by using `mpirun`
```terminal
mpirun -n 4 diff_hymd optimize ...
```
Finally, the program automatically checkpoints the state of the gradients and the parameters after each epoch.
These checkpoints are saved in the `step_#/cpt` directories.
It's possible to restart the optimization from a given checkpoint by simply passing that directory to `diff_hymd`:
```terminal
diff_hymd optimize ... --restart step_600/cpt
```
