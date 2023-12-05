#!/bin/bash
#SBATCH --job-name=<JOB_NAME>    # create a short name for your job
#SBATCH --output=<OUTPUT>        # output file
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=4        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem-per-cpu=4G         # memory per cpu-core (4G is default)
#SBATCH --time=01:00:00          # total run time limit (HH:MM:SS)

module purge
module load python/3.10

# Some packages (e.g., PyTorch), are already installed and configured to use
# GPU. Yet, some other packages should be installed in an environment.
# TODO. Set the environment properly; some packages should be inherited.

source .venv/bin/activate
python -m drafts.inverted_pendulum <BS1> <LR1> <BS2> <LR2>

# deactivate
# rm -rf .venv/
