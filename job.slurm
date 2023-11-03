#!/bin/bash
#SBATCH --job-name=test-learner  # create a short name for your job
#SBATCH --output=test-learner    # output file
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=4        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem-per-cpu=4G         # memory per cpu-core (4G is default)
#SBATCH --time=01:00:00          # total run time limit (HH:MM:SS)
#SBATCH --mail-type=fail         # send email if job fails
#SBATCH --mail-user=mahyar.karimi@ist.ac.at

#SBATCH --partition=gpu          # define the "gpu" partition for GPU-accelerated jobs
#SBATCH --gres=gpu:1             # define the number of GPUs used by your job

module purge
module load cuda/11.7
module load python/3.10

# Some packages (e.g., PyTorch), are already installed and configured to use
# GPU. Yet, some other packages should be installed in an environment.
# TODO. Set the environment properly; some packages should be inherited.

# TODO. Change this command, so it actually does something.
python -m ...

# deactivate
# rm -rf .venv/

