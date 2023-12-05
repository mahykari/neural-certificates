#!/bin/bash
#SBATCH --job-name=invpend       # create a short name for your job
#SBATCH --output=invpend-100-0.0003-100-1.out    # output file
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=4        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem-per-cpu=4G         # memory per cpu-core (4G is default)
#SBATCH --time=01:00:00          # total run time limit (HH:MM:SS)
#SBATCH --mail-type=fail         # send email if job fails
#SBATCH --mail-user=mahyar.karimi@ist.ac.at

module purge
module load python/3.10

# Some packages (e.g., PyTorch), are already installed and configured to use
# GPU. Yet, some other packages should be installed in an environment.
# TODO. Set the environment properly; some packages should be inherited.

# TODO. Change this command, so it actually does something.
source .venv/bin/activate
python -m drafts.inverted_pendulum 000100 0.000300 000100 1.000000

# deactivate
# rm -rf .venv/
