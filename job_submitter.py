from pathlib import Path
import shutil
import time
import random
import subprocess


def product(a, b):
  return [a1 * b1 for a1 in a for b1 in b]


bs1_vals = product([1], [100, 1000])
lr1_vals = product([1, 3, 7], [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e0])

bs2_vals = product([1], [100, 1000])
lr2_vals = product([1, 3, 7], [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e0])

with open('job.template.slurm') as f:
  template = f.read()

if Path('.jobs').exists():
  shutil.rmtree('./jobs')
if Path('./output').exists():
  shutil.rmtree('./output')
Path('./jobs').mkdir(parents=True)
Path('./output').mkdir(parents=True)

args = [
    (f'{bs1:>06d}', f'{lr1:.6f}', f'{bs2:06d}', f'{lr2:.6f}')
    for bs1 in bs1_vals
    for lr1 in lr1_vals
    for bs2 in bs2_vals
    for lr2 in lr2_vals
]

jobfiles = []

print('Generating job files ...')

for arg in args:
  job = template

  jobname = 'invpend-' + '-'.join(arg)
  job = job.replace('<JOB_NAME>', jobname)

  output = 'output/' + jobname + '.out'
  job = job.replace('<OUTPUT>', output)

  bs1, lr1, bs2, lr2 = arg
  job = job.replace('<BS1>', bs1)
  job = job.replace('<LR1>', lr1)
  job = job.replace('<BS2>', bs2)
  job = job.replace('<LR2>', lr2)

  jobfile = 'jobs/' + jobname + '.slurm'
  jobfiles.append(jobfile)
  with open(jobfile, 'w') as f:
    f.write(job)

print(
    'Press any key to start submitting'
    + f'{len(jobfiles):>6} jobs ...', end='')
input()

for jf in jobfiles:
  subprocess.run(['sbatch', jf], check=True)
  time.sleep(random.random())
